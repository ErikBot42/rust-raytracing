use crate::common::*;
use crate::ray::*;
use crate::vector::*;
use crate::bvh::*;
use crate::interval::*;
use crate::hittable::*;


pub trait Light {
    /// Calc (light color, light direction)
    fn calc_light<'a, const LEN: usize>(&self, _world: &BVHHeap<'a, LEN>, _point: Vec3) -> (Vec3, Vec3) {
        (Vec3::default(), Vec3::default())
    }

    // calc: world, point
    // -> NumberType
    // light location
}

pub struct PointLight {
    position: Vec3,
    color: Vec3,
}
impl PointLight {
    pub fn new(position: Vec3, color: Vec3) -> Self {
        Self {position, color}
    }
}


fn point_light<const LEN: usize>(point: Vec3, light_ro: Vec3, world: &BVHHeap<LEN> ) -> NumberType {
    let mut light_rd = light_ro-point;
    let light_dist = light_rd.length();

    let mut light_rd = light_rd/light_dist;
    //let fac = 500.0;
    //light_rd.x = (light_rd.x*fac).round()/fac;
    //light_rd.y = (light_rd.y*fac).round()/fac;
    //light_rd = light_rd.normalized();

    //if light_rd.x+light_rd.y+light_rd.z>0.0 {return 0.0;}
    
    //if light_rd.normalized().dot(-Vec3::new(0.0,-1.0,0.0)) < 0.8 {return 0.0;}
    let light_ray = Ray::new(light_ro, -light_rd);

    let q = 5.0;


    let rd_norm =light_ray.rd.normalized(); 

    let off = 0.5;

    let fac = if ((rd_norm.x+1.0)*q).fract()>0.5 {1.0} else {off};
    let fac = fac*if ((rd_norm.y+1.0)*q).fract()>0.5 {1.0} else {off};
    let fac = fac*if ((rd_norm.z+1.0)*q).fract()>0.5 {1.0} else {off};
    let fac = fac*3.0;

    let min_rad = 00.0;

    //let interval = Interval::new(min_rad+0.001,0.99*light_dist);
    let interval = Interval::new(min_rad+0.001,0.999*light_dist);
    //if light_dist < min_rad {return 0.0;}

    let shadow = match world.hit(&light_ray, interval) {
        None => 1.0,
        Some(_) => {0.0}
    };

    100000.0/(light_dist*light_dist)*
        shadow*
        fac*
        1.0
}

impl Light for PointLight {
    fn calc_light<'a, const LEN: usize>(&self, world: &BVHHeap<'a, LEN>, point: Vec3) -> (Vec3, Vec3) {
    let light_rd = self.position - point;

    let rd_norm = light_rd.normalized();
    let q = 5.0;
    let off = 0.2;
    let on = 1.0;
    let fac = Vec3::new( if ((rd_norm.x+1.0)*q).fract()>0.5 {on} else {off},
                if ((rd_norm.y+1.0)*q).fract()>0.5 {on} else {off},
                if ((rd_norm.z+1.0)*q).fract()>0.5 {on} else {off});

    //(self.color*point_light(point, self.position, world), light_rd)
    (fac*point_light(point, self.position, world), light_rd)
    }

}
