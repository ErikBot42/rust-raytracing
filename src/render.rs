

use rand::Rng;
use std::time::Instant;


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


pub fn render<'a, const LEN: usize>(lights: &'a HittableObject<'a>, bvh2: &BVHHeap<'a, LEN>) {



    let start = Instant::now();
    let rng = rng();
    let aspect_ratio = 1.0;//16.0/9.0;
    let imgx         = 600;
    let imgy         = ((imgx as NumberType)/aspect_ratio) as u32;
    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
    let cam = Camera::new(Vec3::new(278.0,278.0,-800.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio, 0.0, 200.0);
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        //let tr = thread::spawn(move || {
        if x==0 {let f = y as NumberType/imgy as NumberType * 100.0;println!("row {y}/{imgy}: {f}%");}

        let samples   = 2;//16;//32;//256;
        //let samples   = if imgy/2 < y {1024} else {8};//16;//32;//256;
        let max_depth = 8;

        let mut col = Vec3::default();

        for _ in 0..samples
        {
            let u =     (x as NumberType+rng.gen::<NumberType>()) / (imgx as NumberType - 1.0);
            let v = 1.0-(y as NumberType+rng.gen::<NumberType>()) / (imgy as NumberType - 1.0);

            let ray = cam.get_ray(u,v);
            col += ray_color(&ray, &bvh2, &lights, max_depth, Vec3::one(0.0));
        }
        col=col/(samples as NumberType);
        


        let r = (col.x.sqrt()*255.999) as u8;
        let g = (col.y.sqrt()*255.999) as u8;
        let b = (col.z.sqrt()*255.999) as u8;

        //println!("{x}, {y} {r}, {g}, {b}");

        *pixel = image::Rgb([r, g, b]);
        //});
    }
    let duration = (start.elapsed().as_millis() as NumberType)/1000.0;

    let num_objects = LEN;
    println!("Rendered {num_objects}/2+1 objects in {duration} seconds");


    imgbuf.save("output.png").unwrap();

}



fn ray_color<'a, const LEN: usize>(
    ray: &Ray,
    world: &BVHHeap<LEN>,
    light: &'a HittableObject<'a>,
    depth: u32,
    acc: Vec3 //TODO iterative + background function
    ) -> Vec3 {
    if depth==0 {return Vec3::default();}

    let mut rec = HitRecord::default();
    if !world.hit(ray, Interval::EPSILON_UNIVERSE, &mut rec) {
        return Vec3::new(0.0,0.0,0.0) // sky
    }
    if true {
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

        //let mix_pdf = pdf_cosine;
        let scattered = Ray{ro: rec.p, rd: mix_pdf.generate()};
        let pdf = mix_pdf.value(scattered.rd);

        emitted 
            + albedo*rec.material.scattering_pdf(ray, &rec, &scattered)
            *ray_color(&scattered, world, light, depth-1, acc)/pdf}

    else {
        (rec.n+Vec3::one(1.0))*rec.t*0.1
    }
}

