

use crate::vector::Vec3;
use crate::common::*;
use crate::ray::*;
use std::mem;


    use core::cmp::Ordering;

#[derive(Copy, Clone, Default)]
pub struct AABB {
    pub maximum: Vec3,
    pub minimum: Vec3,
}

impl AABB {
    pub fn hit(&self, ray: &Ray, mut t_min: NumberType, mut t_max: NumberType) -> bool
    {
        for a in 0..3 {
            let invd = 1.0/ray.rd[a];
            let mut t0 = (self.minimum[a] - ray.ro[a])*invd;
            let mut t1 = (self.maximum[a] - ray.ro[a])*invd;
            if invd<0.0 {mem::swap(&mut t0,&mut t1);}
            t_min = t_min.max(t0);
            t_max = t_max.max(t1);
            if t_max <= t_min {return false;}
        }
        true
    }
    pub fn surrounding_box(&self, other: AABB) -> AABB
    {
        let min = Vec3::new(
            self.minimum.x.min(other.minimum.x),
            self.minimum.y.min(other.minimum.y),
            self.minimum.z.min(other.minimum.z));
        let max = Vec3::new(
            self.maximum.x.max(other.maximum.x),
            self.maximum.y.max(other.maximum.y),
            self.maximum.z.max(other.maximum.z));
        AABB {minimum: min, maximum: max}
    }
    pub fn compare(&self, other: AABB, axis: u8) -> Ordering {
        self.minimum[axis].partial_cmp(&self.minimum[axis]).unwrap()
    }
}
