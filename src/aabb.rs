

use crate::vector::Vec3;
use crate::interval::*;
use crate::ray::*;
use core::mem;
use core::cmp::Ordering;

#[derive(Copy, Clone, Default)]
#[derive(Debug)]
pub struct AABB {
    pub max: Vec3,
    pub min: Vec3,
}

impl AABB {

    pub fn new(min: Vec3, max: Vec3) -> Self {
        AABB {min, max}
    }

    pub fn hit(&self, ray: &Ray, mut ray_t: Interval) -> bool
    {
        for a in 0..3 {
            //let invd = 1.0/ray.rd[a];
            let invd = ray.rd_inv[a];
            let mut t0 = (self.min[a] - ray.ro[a])*invd;
            let mut t1 = (self.max[a] - ray.ro[a])*invd;
            if invd<0.0 {mem::swap(&mut t0,&mut t1);}
            ray_t.min = ray_t.min.max(t0);
            ray_t.max = ray_t.max.max(t1);
            if ray_t.max <= ray_t.min {return false;}
        }
        true
    }
    pub fn surrounding_box(&self, other: AABB) -> AABB {
        //! Calc AABB that surrounds this AABB
        let min = Vec3::new(
            self.min.x.min(other.min.x),
            self.min.y.min(other.min.y),
            self.min.z.min(other.min.z));
        let max = Vec3::new(
            self.max.x.max(other.max.x),
            self.max.y.max(other.max.y),
            self.max.z.max(other.max.z));
        AABB {min, max}
    }
    pub fn compare(&self, other: AABB, axis: u8) -> Ordering {
        //! Compare along given axis (for BVH)
        self.min[axis].partial_cmp(&other.min[axis]).unwrap()
    }
    pub fn pad(&mut self) {
        //! Make sure no direction is infinitely small.
        let delta = 0.0001;
        for i in 0..3 {
            let size = self.min[i]-self.max[i];
            if size < delta {
                self.max[i] += delta-size;
            }
        }
    }
}
