use std::sync::*;

use ordered_float::OrderedFloat;

use crate::hittable::*;
use crate::interval::*;
use crate::aabb::*;
use crate::ray::*;
use crate::common::*;
use crate::random::*;


use rand::Rng;
#[derive(Clone)]
pub struct BVHnode<'a> {
    aabb: AABB,
    left: Arc<Mutex<HittableObject<'a>>>,
    right: Arc<Mutex<HittableObject<'a>>>,
}

impl<'a> Default for BVHnode<'a>{
    fn default() -> Self {
        BVHnode
        {
            aabb: AABB::default(),
            left: Arc::new(Mutex::new(HittableObject::default())),
            right: Arc::new(Mutex::new(HittableObject::default())),
        }
    }
}
impl<'a> Hittable<'a> for BVHnode<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        if !self.aabb.hit(ray, ray_t) {return false;}
        let hit_left = self.left.lock().unwrap().hit(ray, ray_t, rec);
        let hit_right = self.right.lock().unwrap().hit(ray, 
            Interval::new(ray_t.min, if hit_left {rec.t} else {ray_t.max}), rec);
        hit_left || hit_right
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        *aabb = self.aabb;
        println!("BVHnode bounding box");
        true
    }  
}

impl<'a> BVHnode<'a> {
    fn box_val<T: Hittable<'a> + ?Sized>(a: &Arc<Mutex<T>>, axis: u8) -> NumberType
    {
        let mut a_box = AABB::default();
        if !a.lock().unwrap().bounding_box(&mut a_box) {panic!("missing implemenation for AABB");}
        else {a_box.minimum[axis]}
    }

    pub fn construct(objects: Vec<Arc<Mutex<HittableObject<'a>>>>) -> Arc<Mutex<HittableObject>>
    {
        let mut node = BVHnode::default();
        let mut copy = objects.clone();
        let axis: u8 = rng().gen_range(0..3);

        println!("Axis: {axis}");

        let x = move |a:&Arc<Mutex<HittableObject<'a>>>| OrderedFloat(Self::box_val(a,axis));

        let object_span = copy.len();
        println!("object_span = {object_span}");

        if object_span == 0 {
            panic!("no elements when running construct");
        }
        if object_span == 1 {
            return copy[copy.len()-1].clone();
            //node.left  = copy[copy.len()-1].clone();
            //node.right = copy[copy.len()-1].clone();

            //println!("SINGLE OBJECT");
        }
        else if object_span == 2 {
            node.left  = copy[copy.len()-1].clone();
            node.right = copy[copy.len()-2].clone();
            //if x(&self.left) > x(&self.right)
            //{
            //    mem::swap(&mut self.left, &mut self.right);
            //}
        }
        else {
            copy.sort_by_key(x);
            let mid = object_span/2;


            let left_node = Self::construct(copy[mid..].to_vec());
            node.left = left_node;

            let right_node = Self::construct(copy[..mid].to_vec());
            node.right = right_node;
        }

        let mut box_left = AABB::default(); 
        let mut box_right = AABB::default(); 

        let has_left = node.left.lock().unwrap().bounding_box(&mut box_left);
        let has_right = node.right.lock().unwrap().bounding_box(&mut box_right);
        if !has_left || !has_right
        {
            panic!("AABB missing");
        }
        node.aabb = box_left.surrounding_box(box_right);
        Arc::new(Mutex::new(HittableObject::BVHnode(node)))
    }
}

#[derive(Clone)]

//impl Copy for BVHEnum {}
enum BVHEnum<'a> {
    BVHHeapNode(BVHHeapNode),
    HittableObjectSimple(HittableObjectSimple<'a>),
}
impl<'a> Default for BVHEnum<'a> {
    fn default() -> Self {BVHEnum::BVHHeapNode(BVHHeapNode::default())}
}
impl<'a> Hittable<'a> for BVHEnum<'a> {
    fn hit(&self, _ray: &Ray, _ray_t: Interval, _rec: &mut HitRecord<'a>) -> bool {
        unimplemented!(); 
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        match self {
            BVHEnum::BVHHeapNode(n) => {*aabb = n.aabb; true},
            BVHEnum::HittableObjectSimple(s) => s.bounding_box(aabb),
        }
    }  
}

struct BVH;

impl BVH {
    fn top() -> usize {0}
    fn left(index: usize) -> usize {(index+1)*2-1}
    fn right(index: usize) -> usize {(index+1)*2+1-1}
    fn parent(index: usize) -> Option<usize> {
        if index == 0 {None}
        else {Some(index/2)}
    }
}

// switch node -> enum to store actual objects too
pub struct BVHHeap<'a, const LEN: usize> {
    arr: [BVHEnum<'a>; LEN],
}
impl<'a, const LEN: usize> Default for BVHHeap<'a, LEN>
{
    fn default() -> Self {
        unsafe {
            let mut arr: [BVHEnum; LEN] = core::mem::zeroed();
            for x in &mut arr {*x = BVHEnum::default();}
            BVHHeap{arr}
        }
    }
}
impl<'a, const LEN: usize> Hittable<'a> for BVHHeap<'a, LEN> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        self.hit_recursive(ray, ray_t, rec, 0)
    }
    fn bounding_box(&self, _aabb: &mut AABB) -> bool {
        unimplemented!();
    }  
}
impl<'a, const LEN: usize> BVHHeap<'a, LEN> {
    fn hit_recursive(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>, index: usize) -> bool {
        match &self.arr[index] {
            BVHEnum::HittableObjectSimple(s) => s.hit(ray, ray_t, rec),
            BVHEnum::BVHHeapNode(n) => {
                if !n.aabb.hit(ray, ray_t) {false}
                else {
                    let hit_left = self.hit_recursive(ray, ray_t, rec, BVH::left(index));
                    let new_interval = Interval::new(ray_t.min, if hit_left {rec.t} else {ray_t.max});
                    let hit_right = self.hit_recursive(ray, new_interval, rec, BVH::right(index));
                    hit_left || hit_right
                }
            }
             
        }
    }
    pub fn construct_new(objects: &mut [HittableObjectSimple<'a>]) -> Self {
        let mut bvh = Self::default();
        bvh.construct(objects, 0);
        bvh
        
    }
    pub fn construct(&mut self, objects: &mut [HittableObjectSimple<'a>], index: usize) {
        let cardinality = objects.len();
        assert!(cardinality!=0, "empty list of hittable objects");
        
        if cardinality == 1 {
            *self.at(index) = BVHEnum::HittableObjectSimple(objects[0].clone());
        }
        else {

            if cardinality == 2 {
                *self.left(index) = BVHEnum::HittableObjectSimple(objects[0].clone());
                *self.right(index) = BVHEnum::HittableObjectSimple(objects[1].clone());

            }
            else {

                let axis = rng().gen_range(0..3);
                let x = 
                    move |a: &HittableObjectSimple, b: &HittableObjectSimple| {
                    let a_box = AABB::default();
                    let b_box = AABB::default();
                    a_box.compare(b_box, axis)
                };
                objects.sort_unstable_by(x);
                
                let mid = cardinality/2;
                
                self.construct(&mut objects[..mid], BVH::left(index));
                self.construct(&mut objects[mid..], BVH::right(index));
            }

            // left and right has been constucted at this point.

            let mut box_left = AABB::default();
            let mut box_right = AABB::default();

            let has_left = self.left(index).bounding_box(&mut box_left);
            let has_right = self.right(index).bounding_box(&mut box_right);

            assert!(has_left && has_right);
            
            let aabb = box_left.surrounding_box(box_right);

            *self.at(index) = BVHEnum::BVHHeapNode(BVHHeapNode::new(aabb));

        }
    }

    fn at(&mut self, index: usize) -> &mut BVHEnum<'a> {
        &mut self.arr[index]
    }
    fn left(&mut self, index: usize) -> &mut BVHEnum<'a> {
        self.at(BVH::left(index))
    }
    fn right(&mut self, index: usize) -> &mut BVHEnum<'a> {
        self.at(BVH::left(index))
    }
}


impl<'a, const LEN: usize> BVHHeap<'a, LEN> {
}

// to be put in an array
#[derive(Default,Copy,Clone)]
struct BVHHeapNode {
    aabb: AABB,
}
impl BVHHeapNode {
    fn new(aabb: AABB) -> Self {
        BVHHeapNode{ aabb }
    }
//    fn hit(self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
//    }
//    fn bounding_box(&self, aabb: &mut AABB) -> bool {
//    }  
}




// needs hittable
#[test]
fn test_bvh() {
    //use std::mem::size_of;
    
    assert_eq!(BVH::left(0),1);
    assert_eq!(BVH::right(0),2);
    assert_eq!(BVH::parent(0),None);
    assert_eq!(BVH::parent(1),Some(0));
    assert_eq!(BVH::parent(1),Some(0));
    
    //let s = size_of::<HittableObject>();
    //let t = size_of::<BVHHeapNode>();
    ////let v = size_of::<BVHEnum>();
    //panic!("{s}, {t}");
    
    //let bvh: BVHHeap<3> = BVHHeap::new();
}
