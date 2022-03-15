

use crate::hittable::*;
use crate::interval::*;
use crate::aabb::*;
use crate::ray::*;
use crate::random::*;


use rand::Rng;


#[derive(Clone)]

//impl Copy for BVHEnum {}
#[derive(Debug)]
enum BVHEnum<'a> {
    BVHHeapNode(BVHHeapNode),
    HittableObject(HittableObject<'a>),
}
impl<'a> Default for BVHEnum<'a> {
    fn default() -> Self {BVHEnum::BVHHeapNode(BVHHeapNode::default())}
}
impl<'a> Hittable<'a> for BVHEnum<'a> {
    fn hit(&self, _ray: &Ray, _ray_t: Interval) -> Option<HitRecord> {
        unimplemented!(); 
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        match self {
            BVHEnum::BVHHeapNode(n) => {*aabb = n.aabb; true},
            BVHEnum::HittableObject(s) => s.bounding_box(aabb),
        }
    }  
}

struct Bvh;

impl Bvh {
    #[allow(dead_code)]
    fn top() -> usize {0}

    fn left(index: usize) -> usize {(index+1)*2-1}
    fn right(index: usize) -> usize {(index+1)*2+1-1}

    #[allow(dead_code)]
    fn parent(index: usize) -> Option<usize> {
        if index == 0 {None}
        else {Some(index/2)}
    }
}

// switch node -> enum to store actual objects too
#[derive(Debug)]
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
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        self.hit_recursive(ray, ray_t, 0)
    }
    fn bounding_box(&self, _aabb: &mut AABB) -> bool {
        unimplemented!();
    }  
}
impl<'a, const LEN: usize> BVHHeap<'a, LEN> {
    fn hit_recursive(&self, ray: &Ray, ray_t: Interval, index: usize) -> Option<HitRecord>{
        //println!("hit_recursive({index})");
        match &self.arr[index] {
            BVHEnum::HittableObject(s) => s.hit(ray, ray_t),
            BVHEnum::BVHHeapNode(n) => {
                if !n.aabb.hit(ray, ray_t) {
                    //println!("AABB missed");
                    None}
                else {
                    //println!("AABB hit");
                    //println!("Calling hit_recursive({}) from hit_recursive({})",BVH::left(index), index);
                    let hit_left = self.hit_recursive(ray, ray_t, Bvh::left(index));
                    let new_interval = Interval::new(ray_t.min, 
                                                     match hit_left {
                                                         None => ray_t.max,
                                                         Some(rec) => {
                                                             rec.t 
                                                         }
                                                     });
                    //println!("Calling hit_recursive({}) from hit_recursive({})",BVH::right(index), index);
                    let hit_right = self.hit_recursive(ray, new_interval, Bvh::right(index));
                    match hit_right {
                        None => hit_left,
                        Some(rec) => Some(rec),

                    }
                }
            }
             
        }
    }
    pub fn construct_new(objects: &mut [HittableObject<'a>]) -> Self {
        let mut bvh = Self::default();
        bvh.construct(objects, 0, 0);
        bvh
        
    }
    fn construct(&mut self, objects: &mut [HittableObject<'a>], index: usize, axis: u8) {

        let cardinality = objects.len();
        assert!(cardinality!=0, "empty list of hittable objects");
        
        //println!("construct({index})");

        if cardinality == 1 {
            *self.at(index) = BVHEnum::HittableObject(objects[0].clone());
        }
        else {

            if cardinality == 2 {
                *self.left(index) = BVHEnum::HittableObject(objects[0].clone());
                *self.right(index) = BVHEnum::HittableObject(objects[1].clone());

            }
            else {

                //let axis = rng().gen_range(0..3);
                let x = 
                    move |a: &HittableObject, b: &HittableObject| {
                    let mut a_box = AABB::default();
                    let mut b_box = AABB::default();

                    a.bounding_box(&mut a_box);
                    b.bounding_box(&mut b_box);
                    a_box.compare(b_box, axis)
                };
                objects.sort_unstable_by(x);
                
                let mid = cardinality/2;
                
                self.construct(&mut objects[..mid], Bvh::left(index), (axis+1)%3);
                self.construct(&mut objects[mid..], Bvh::right(index), (axis-1)%3);
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
        self.at(Bvh::left(index))
    }
    fn right(&mut self, index: usize) -> &mut BVHEnum<'a> {
        self.at(Bvh::right(index))
    }
}


impl<'a, const LEN: usize> BVHHeap<'a, LEN> {
}

// to be put in an array
#[derive(Default,Copy,Clone)]
#[derive(Debug)]
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
    
    assert_eq!(Bvh::left(0),1);
    assert_eq!(Bvh::right(0),2);
    assert_eq!(Bvh::parent(0),None);
    assert_eq!(Bvh::parent(1),Some(0));
    assert_eq!(Bvh::parent(1),Some(0));
    
    //let s = size_of::<HittableObject>();
    //let t = size_of::<BVHHeapNode>();
    ////let v = size_of::<BVHEnum>();
    //panic!("{s}, {t}");
    
    //let bvh: BVHHeap<3> = BVHHeap::new();
}
