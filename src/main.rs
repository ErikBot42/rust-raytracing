
extern crate image;
extern crate lazy_static;
extern crate rand;
extern crate smallvec;

pub mod common;
pub mod interval;
pub mod vector;
pub mod random;
pub mod ray;
pub mod texture;
pub mod onb;
pub mod pdf;
pub mod hittable;
pub mod material;
pub mod aabb;
pub mod bvh;
pub mod render;

//use crate::HittableObject::*;
use crate::vector::Vec3;
use crate::random::rng_seed;
use crate::texture::{SolidColor, CheckerTexture};
use crate::hittable::*;
use crate::material::*;
use crate::bvh::*;
use crate::render::*;




//fn cornell_box<'a>() -> [HittableObjectSimple<'a>; 8]{
//
//
//    let red = Lambertian::col(Vec3::new(0.65,0.05,0.05));
//    let green = Lambertian::col(Vec3::new(0.12,0.45,0.15));
//    let light = MaterialEnum::Emissive(Emissive{light: Vec3::new(15.0,15.0,15.0)});
//
//
//    let white = Lambertian::col(Vec3::one(0.73));
//
//    
//    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,330.0,165.0), white));
//    let cube_rot = HittableObject::RotateY(RotateY::new(&cube,  15.0));
//    let cube_trans = HittableObjectSimple::Translate(Translate{object: &cube, offset: Vec3::new(265.0,0.0,295.0)});
//
//    let cube2 = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,165.0,165.0), white));
//    let cube2 = HittableObject::RotateY(RotateY::new(&cube2, -18.0));
//    let cube2 = HittableObjectSimple::Translate(Translate{object: &cube2, offset: Vec3::new(130.0,0.0,65.0)});
//
//    let cornell_box = [
//        cube,
//        HittableObjectSimple::XZRect(XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }),
//        cube2,
//        HittableObjectSimple::YZRect(YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
//        HittableObjectSimple::YZRect(YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
//        HittableObjectSimple::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
//        HittableObjectSimple::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
//        HittableObjectSimple::XYRect(XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, })
//    ];
//
//    cornell_box
//
//}



fn main() {
    rng_seed();

    let big_light = false;
    let c1 = SolidColor::new(Vec3::one(0.8));
    let c2 = SolidColor::new(Vec3::new(0.1,0.1,0.6));
    let checker = 
        CheckerTexture::new(& c1, & c2 );
    let checker = Lambertian::new(checker);

    //let glass = MaterialEnum::Dielectric(Dielectric{ir:1.5});
    let white = checker;//Lambertian::col(Vec3::one(0.73));
    let red = Lambertian::col(Vec3::new(0.65,0.05,0.05));
    let green = Lambertian::col(Vec3::new(0.12,0.45,0.15));
    let light = MaterialEnum::Emissive(Emissive{light:
        if big_light {Vec3::new(4.0,4.0,4.0)}
        else{Vec3::new(15.0,15.0,15.0)}});

    let lights =  if big_light {XZRect{material: light, x0: 113.0, x1: 443.0, z0: 127.0, z1: 432.0, k: 554.0, }}
    else {XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }};

    let lights = HittableObject::XZRect(lights);
    
    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,330.0,165.0), white));
    let cube = HittableObject::RotateY(RotateY::new(&cube,  15.0));
    let cube = HittableObjectSimple::Translate(Translate{object: &cube, offset: Vec3::new(265.0,0.0,295.0)});

    let cube2 = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,165.0,165.0), white));
    let cube2 = HittableObject::RotateY(RotateY::new(&cube2, -18.0));
    let cube2 = HittableObjectSimple::Translate(Translate{object: &cube2, offset: Vec3::new(130.0,0.0,65.0)});
    //object_list.push(Arc::new(Mutex::new(cube2)));

    //let cube2 = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::one(165.0), white));
    
    const SIZE_CORNELL: usize = 8;
    let cornell_box = [
        cube,
        HittableObjectSimple::XZRect(XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }),
        cube2,
        HittableObjectSimple::YZRect(YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
        HittableObjectSimple::YZRect(YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
        HittableObjectSimple::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
        HittableObjectSimple::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
        HittableObjectSimple::XYRect(XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, })
    ];
    const MAX_SIZE: usize = 100; 
    const HEAP_SIZE: usize = MAX_SIZE*2;
    let mut hittable_list = cornell_box;
    let bvh2: BVHHeap<HEAP_SIZE> = {BVHHeap::construct_new(&mut hittable_list[0..8])};

    render(&lights, &bvh2);

}


