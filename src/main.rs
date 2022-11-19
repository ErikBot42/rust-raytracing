
//#![no_std]
extern crate image;
//extern crate lazy_static;
extern crate rand;
//extern crate smallvec;
//extern crate thread_local;

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
pub mod light;

//use crate::HittableObject::*;
use crate::vector::Vec3;
use crate::texture::{SolidColor, CheckerTexture};
use crate::hittable::*;
use crate::material::*;
use crate::bvh::*;
use crate::render::*;



//fn cornell_box<'a>() -> [HittableObject<'a>; 8]{
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
//    let cube_trans = HittableObject::Translate(Translate{object: &cube, offset: Vec3::new(265.0,0.0,295.0)});
//
//    let cube2 = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,165.0,165.0), white));
//    let cube2 = HittableObject::RotateY(RotateY::new(&cube2, -18.0));
//    let cube2 = HittableObject::Translate(Translate{object: &cube2, offset: Vec3::new(130.0,0.0,65.0)});
//
//    let cornell_box = [
//        cube,
//        HittableObject::XZRect(XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }),
//        cube2,
//        HittableObject::YZRect(YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
//        HittableObject::YZRect(YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
//        HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
//        HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
//        HittableObject::XYRect(XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, })
//    ];
//
//    cornell_box
//
//}
//use std::env;

fn main() {

    //let args: Vec<String> = env::args().collect();
    //
    //let samples = args[1].parse::<u32>().unwrap();
    let samples = 8192;//256;

    let big_light = false;
    //let c1 = SolidColor::create(Vec3::one(0.8));
    //let c2 = SolidColor::create(Vec3::new(0.1,0.1,0.6));
    //let checker = CheckerTexture::create(& c1, & c2 );
    //let checker = Lambertian::create(checker);

    //let aluminum = Metal::col(Vec3::new(0.8, 0.85, 0.88)*0.5, 0.5);
    //let gold = Metal::col(Vec3::new(255.0,215.0,0.0)/255.0, 0.3);
    let glass = Dielectric::create(1.5);

    //let glass = MaterialEnum::Dielectric(Dielectric{ir:1.5});
    let white = Lambertian::col(Vec3::one(0.73));
    let red = Lambertian::col(Vec3::new(0.65,0.05,0.05));
    let green = Lambertian::col(Vec3::new(0.12,0.45,0.15));
    let light = MaterialEnum::Emissive(Emissive{light:
        if big_light {Vec3::new(4.0,4.0,4.0)}
        else{Vec3::new(15.0,15.0,15.0)}});
        

    let sph = HittableObject::Sphere(Sphere::new(glass, Vec3::new(200.0,100.0,250.0),100.0));

    let top_light =  if big_light {XZRect{material: light, x0: 113.0, x1: 443.0, z0: 127.0, z1: 432.0, k: 554.0, }}
    else {XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }};

    let lights = HittableObject::XZRect(top_light);
    //let lights = sph.clone();
    
    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,330.0,165.0), white));
    let cube = HittableObject::RotateY(RotateY::new(&cube,  15.0));
    let cube = HittableObject::Translate(Translate{object: &cube, offset: Vec3::new(265.0,00.0,295.0)});

    //let cube2 = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,165.0,165.0), white));
    //let cube2 = HittableObject::RotateY(RotateY::new(&cube2, -18.0));
    //let cube2 = HittableObject::Translate(Translate{object: &cube2, offset: Vec3::new(130.0,50.0,65.0)});

    //object_list.push(Arc::new(Mutex::new(cube2)));

    //let cube2 = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::one(165.0), white));
    //fn new(q: Vec3, u: Vec3, v: Vec3, material: MaterialEnum<'a>) -> Self {
    const SIZE_CORNELL: usize = 8;
    let cornell_box = [
        cube,
        HittableObject::XZRect(top_light),
        //HittableObject::XZRect(XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }),
        //cube2,
        //HittableObject::YZRect(YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
        HittableObject::Quad(Quad::new(Vec3::new(555.0, 0.0, 0.0), Vec3::new(0.0,555.0,0.0), Vec3::new(0.0,0.0,555.0), green)), 
        //HittableObject::YZRect(YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
        HittableObject::Quad(Quad::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0,555.0,0.0), Vec3::new(0.0,0.0,555.0), red)), 
        HittableObject::Quad(Quad::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(555.0,0.0,0.0), Vec3::new(0.0,0.0,555.0), white)), 
        HittableObject::Quad(Quad::new(Vec3::new(555.0, 555.0, 555.0), Vec3::new(-555.0,0.0,0.0), Vec3::new(0.0,0.0,-555.0), white)), 
        HittableObject::Quad(Quad::new(Vec3::new(0.0, 0.0, 555.0), Vec3::new(555.0,0.0,0.0), Vec3::new(0.0,555.0,0.0), white)), 
        //HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }),
        //HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }),
        //HittableObject::XYRect(XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, }),
        sph, 
        //HittableObject::Sphere(Sphere::new(metal, Vec3::one(200.0),100.0)),
        //HittableObject::Sphere(Sphere::new(metal, Vec3::one(300.0),100.0)),
        //HittableObject::Sphere(Sphere::new(metal, Vec3::one(400.0),100.0)),
        //HittableObject::Sphere(Sphere::new(metal, Vec3::one(500.0),100.0)),
    ];
    const MAX_SIZE: usize = SIZE_CORNELL; 


    const HEAP_SIZE: usize = MAX_SIZE*2*2;
    let mut hittable_list = cornell_box;
    let bvh2: BVHHeap<HEAP_SIZE> = {BVHHeap::construct_new(&mut hittable_list[0..SIZE_CORNELL])};


    render(&lights, &bvh2, samples);

}


