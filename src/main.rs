
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

//use crate::HittableObject::*;
use crate::vector::Vec3;
use crate::interval::Interval;
use crate::common::*;
use crate::random::{rng_seed,rng};
use crate::ray::Ray;
use crate::texture::{SolidColor, CheckerTexture};
use crate::pdf::*;
use crate::hittable::*;
use crate::material::*;
use crate::bvh::*;

//DivAssign,MulAssign
use rand::Rng;
//use std::rc::Rc;
//use std::cell::RefCell;
use std::time::Instant;
//use micromath::F32Ext;
//use std::thread;
use std::sync::{Arc,Mutex};


//fn rng() -> ThreadRng {
//    thread_rng()
//}





fn ray_color<'a>(
    ray: &Ray,
    world: &HittableObject,
    light: &'a HittableObject<'a>,
    depth: u32,
    acc: Vec3 //TODO iterative + background function
    ) -> Vec3 {
    if depth==0 {return Vec3::default();}

    let mut rec = HitRecord::default();
    if !world.hit(ray, Interval::EPSILON_UNIVERSE, &mut rec) {
        return Vec3::one(0.0) // sky
    }

    let mut scattered = Ray::default();
    let emitted = rec.material.emission();
    let mut pdf = 0.0;
    let mut albedo = Vec3::one(0.0);

    if !rec.material.scatter(ray, &rec, &mut albedo, &mut scattered, &mut pdf) {return emitted;}
   
    //let p = CosinePDF::new(rec.n);
    //scattered = Ray{ro: rec.p, rd: p.generate()};
    //pdf = p.value(scattered.rd);
    
    //let light_pdf = HittablePDF::new(light, rec.p);
    //let scattered = Ray{ro: rec.p, rd:light_pdf.generate()};
    //let pdf = light_pdf.value(scattered.rd);

    let pdf_light = HittablePDF::new(light, rec.p);
    let pdf_cosine = CosinePDF::new(rec.n);
    //let mix_pdf = pdf_light;
    let mix_pdf = MixPDF::new(&pdf_light, &pdf_cosine, 0.5);
    
    //let c1 = {
    ////let mix_pdf = pdf_light;
    //let scattered = Ray{ro: rec.p, rd: mix_pdf.generate()};
    //let pdf = mix_pdf.value(scattered.rd);
    //emitted 
    //    + albedo*rec.material.scattering_pdf(ray, &rec, &scattered)
    //    *ray_color(&scattered, world, light, (depth-1).min(1), acc)/pdf};

    let c2 = {
    //let mix_pdf = pdf_cosine;
    let scattered = Ray{ro: rec.p, rd: mix_pdf.generate()};
    let pdf = mix_pdf.value(scattered.rd);
    emitted 
        + albedo*rec.material.scattering_pdf(ray, &rec, &scattered)
        *ray_color(&scattered, world, light, depth-1, acc)/pdf};

    (c2+c2)/2.0
}





//impl Default for BVHnode2<'_> {fn default() -> Self {BVHnode2::Tail}}
//impl BVHnode2<'a> {
//    fn add(&'a self, aabb: AABB) -> Self {
//        BVHnode2::Node{aabb, left: self, right: self}
//    } 
//}





#[derive(Clone, Copy, Default, Debug)]
struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    lens_radius: NumberType,
}


impl Camera {
    fn new(
        lookfrom: Vec3,
        lookat: Vec3,
        up: Vec3,
        fov: NumberType,
        aspect_ratio: NumberType,
        aperture: NumberType,
        focus_dist: NumberType,
        ) -> Self {

        let theta = deg2rad(fov);
        let h = (theta/2.0).tan();
        let viewport_height = 2.0*h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom-lookat).normalized();
        let u = up.cross(w).normalized();
        let v = w.cross(u);

        let mut cam: Camera = Camera::default();
        cam.origin = lookfrom;
        cam.horizontal = u*viewport_width*focus_dist;
        cam.vertical = v*viewport_height*focus_dist;
        cam.lower_left_corner = cam.origin - cam.horizontal/2.0 - cam.vertical/2.0 - w*focus_dist;
        //println!("{cam:?}");
        cam.lens_radius = aperture / 2.0;

        cam
    }
    fn get_ray(&self, u: NumberType, v: NumberType) -> Ray {
        let rd = Vec3::random_in_unit_disk()*self.lens_radius;
        let offset = rd*Vec3::new(u,v,0.0);
        //let offset = u*rd.x + v*rd.y;
        //let offset = Vec3::new(offset,0.0,0.0);
        //

        // rd is mostly a matrix multiplication, (u,v)*[horizontal,vertical]
        Ray {
            ro: self.origin + offset, 
            rd: self.lower_left_corner + self.horizontal*u + self.vertical*v - self.origin - offset,
        }
    }
}







//#[derive(Clone,Copy)]
//struct Interval {
//    min: NumberType,
//    max: NumberType,
//}



fn main() {
    rng_seed();

    let aspect_ratio = 1.0;//16.0/9.0;
    let imgx         = 600;
    let imgy         = ((imgx as NumberType)/aspect_ratio) as u32;

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);


    let mut object_list: Vec<Arc<Mutex<HittableObject>>> = Vec::new();
    let big_light = false;

    //let red   = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.65,0.05,0.05)});
    //let white = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.73,0.73,0.73)});
    //let green = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.12,0.45,0.15)});


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

    //let blue  = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.12,0.12,0.45)});
    //let grey  = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.73,0.73,0.73)*0.4});
    //let light_red = MaterialEnum::Emissive(Emissive{light:Vec3::new(7.0,0.0,0.0)});
    //let light_blue = MaterialEnum::Emissive(Emissive{light:Vec3::new(0.0,0.0,7.0)});

    //object_list.push(Arc::new(Mutex::new( HittableObject::Sphere(Sphere{material: glass, center: Vec3::one(250.0), radius: 100.0}))));
    object_list.push(Arc::new(Mutex::new( HittableObject::Sphere(Sphere{material: white, center: Vec3::one(400.0), radius: 100.0}))));

    let lights =  if big_light {XZRect{material: light, x0: 113.0, x1: 443.0, z0: 127.0, z1: 432.0, k: 554.0, }}
    else {XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }};

    object_list.push(Arc::new(Mutex::new(HittableObject::XZRect(lights))));
    let lights = HittableObject::XZRect(lights);

    object_list.push(Arc::new(Mutex::new( HittableObject::YZRect(YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }))));
    object_list.push(Arc::new(Mutex::new( HittableObject::YZRect(YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }))));
    object_list.push(Arc::new(Mutex::new( HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }))));
    object_list.push(Arc::new(Mutex::new( HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }))));
    object_list.push(Arc::new(Mutex::new( HittableObject::XYRect(XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, }))));

    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,330.0,165.0), white));
    let cube = HittableObject::RotateY(RotateY::new(&cube, 15.0));
    let cube = HittableObject::Translate(Translate{object: &cube, offset: Vec3::new(265.0,0.0,295.0)});
    object_list.push(Arc::new(Mutex::new(cube)));

    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::one(165.0), white));
    let cube = HittableObject::RotateY(RotateY::new(&cube, -18.0));
    let cube = HittableObject::Translate(Translate{object: &cube, offset: Vec3::new(130.0,0.0,65.0)});
    object_list.push(Arc::new(Mutex::new(cube)));


    //object_list.push(Arc::new(Mutex::new( HittableObject::Cuboid(
    //                Cuboid::new(Vec3::new(265.0,0.0,295.0), Vec3::new(430.0,330.0,460.0), white)))));

    //object_list.push(Arc::new(Mutex::new(
    //            XZRect{material: light, 
    //                x0: 123.0,
    //                x1: 423.0,
    //                z0: 147.0, 
    //                z1: 412.0,
    //                k:  554.0, 
    //            })));

    //    let material = Isotropic{albedo: Vec3::one(0.5)};
    //    let fog_sphere = Arc::new(Mutex::new(
    //            HittableObject::Sphere(
    //                Sphere{
    //                    material: white,
    //                    radius: 5000.0,
    //                    center: Vec3::one(0.0),
    //                } )));
    //
    //    let fog_sphere = Arc::new(Mutex::new(
    //                ConstantMedium{
    //                    material: MaterialEnum::Isotropic(material),
    //                    boundary: fog_sphere,
    //                    neg_inv_denisty: -1.0/0.0001,
    //                } ));
    //    object_list.push(fog_sphere);

    //let material = Lambertian{albedo: Vec3::new(0.5,0.7,0.2)};
    //object_list.push(Arc::new(Mutex::new(
    //            Sphere{
    //                material: MaterialEnum::Lambertian(material),
    //                radius: 100.0,
    //                center: Vec3::new(555.0/2.0+100.0,555.0/2.0-100.0,555.0/2.0),
    //            } )));



    //if false { 
    //    for i in 0..10 {
    //        let rad = 50.0;
    //        let center = Vec3::new(random_range(0.0,350.0),random_range(0.0,350.0),random_range(0.0,350.0));
    //        let choose_mat = random_val();

    //        if choose_mat < 0.5 {
    //            let albedo = Vec3::random();
    //            let material = MaterialEnum::Lambertian(Lambertian {albedo});
    //            object_list.push(Arc::new(Mutex::new(Sphere{material, radius: rad, center})));
    //        }
    //        //else if choose_mat < 0.5 {
    //        //    let light = Vec3::random()*4.0;
    //        //    let material = MaterialEnum::Emissive(Emissive{light});
    //        //    object_list.push(Arc::new(Mutex::new(Sphere{material: material.clone(), radius: rad, center})));
    //        //}
    //        else if choose_mat < 0.75{
    //            let albedo = Vec3::random();
    //            let blur = random_range(0.0,0.5);
    //            let material = MaterialEnum::Metal(Metal{albedo,blur});
    //            object_list.push(Arc::new(Mutex::new(Sphere{material: material.clone(), radius: rad, center})));
    //        }
    //        else {
    //            let material = MaterialEnum::Dielectric(Dielectric{ir:1.5});
    //            object_list.push(Arc::new(Mutex::new(Sphere{material: material.clone(), radius: rad, center})));
    //        }

    //    }
    //}

    let num_objects = object_list.len();
    let bvh = BVHnode::construct(object_list);
    let bvh = bvh.lock().unwrap();

    //let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 45.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(26.0,3.0,6.0), Vec3::new(0.0,2.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(278.0,278.0,-800.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio, 100.0, 950.0);
    let cam = Camera::new(Vec3::new(278.0,278.0,-800.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio, 1.0, 200.0);
    //let cam = Camera::new(Vec3::new(478.0,278.0,-600.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio);

    let start = Instant::now();


    let rng = rng();
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        //let tr = thread::spawn(move || {
        if x==0 {let f = y as NumberType/imgy as NumberType * 100.0;println!("row {y}/{imgy}: {f}%");}

        let samples   = 8;//16;//32;//256;
        //let samples   = if imgy/2 < y {1024} else {8};//16;//32;//256;
        let max_depth = 8;

        let mut col = Vec3::default();

        for _ in 0..samples
        {
            let u =     (x as NumberType+rng.gen::<NumberType>()) / (imgx as NumberType - 1.0);
            let v = 1.0-(y as NumberType+rng.gen::<NumberType>()) / (imgy as NumberType - 1.0);

            let ray = cam.get_ray(u,v);
            col += ray_color(&ray, &bvh, &lights, max_depth, Vec3::one(0.0));
        }
        col=col/(samples as NumberType);

        let r = (col.x.sqrt()*255.999) as u8;
        let g = (col.y.sqrt()*255.999) as u8;
        let b = (col.z.sqrt()*255.999) as u8;

        *pixel = image::Rgb([r, g, b]);
        //});
    }
    let duration = (start.elapsed().as_millis() as NumberType)/1000.0;
    println!("Rendered {num_objects} objects in {duration} seconds");


    imgbuf.save("output.png").unwrap();
}

