use crate::common::*;
use crate::ray::*;
use crate::vector::*;
use crate::bvh::*;


pub trait Light {
    fn calc_light<'a, const LEN: usize>(point: Vec3, normal: Vec3, bvh: &BVHHeap<'a, LEN>) -> Vec3 {
        //! Calulate amount of light that hits this point
        Vec3::default() 
    }
}
