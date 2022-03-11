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
        if !self.aabb.hit(ray, ray_t.min, ray_t.max) {return false;}
        let hit_left = self.left.lock().unwrap().hit(ray, ray_t, rec);
        let hit_right = self.right.lock().unwrap().hit(ray, Interval::new(ray_t.min, if hit_left {rec.t} else {ray_t.max}), rec);
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

struct BVH;

impl BVH {
    fn left(index: usize) -> usize {(index+1)*2-1}
    fn right(index: usize) -> usize {(index+1)*2+1-1}
    fn parent(index: usize) -> Option<usize> {
        if index == 0 {None}
        else {Some(index/2)}
    }
}

// switch node -> enum to store actual objects too
struct BVHHeap<const LEN: usize> {
    array: [BVHHeapNode; LEN],
}
impl<const LEN: usize> Default for BVHHeap<LEN>
{
    fn default() -> Self {
        BVHHeap{ array: [BVHHeapNode::default();LEN]}     
    }
}
impl<const LEN: usize> BVHHeap<LEN> {
    fn new() -> Self {
        Self::default() 
    }
}

// to be put in an array
#[derive(Default,Copy,Clone)]
struct BVHHeapNode {
    aabb: AABB,
}
// needs hittable



#[test]
fn test_bvh() {
    
    assert_eq!(BVH::left(0),1);
    assert_eq!(BVH::right(0),2);
    assert_eq!(BVH::parent(0),None);
    assert_eq!(BVH::parent(1),Some(0));
    assert_eq!(BVH::parent(1),Some(0));
    
    //let bvh: BVHHeap<3> = BVHHeap::new();
    

    
}