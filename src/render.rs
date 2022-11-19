use std::time::Instant;

use crate::bvh::*;
use crate::common::*;
use crate::hittable::*;
use crate::interval::Interval;
use crate::material::*;
use crate::pdf::*;
use crate::random::*;
use crate::ray::Ray;
use crate::vector::Vec3;

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
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom - lookat).normalized();
        let u = up.cross(w).normalized();
        let v = w.cross(u);

        let mut cam: Camera = Camera::default();
        cam.origin = lookfrom;
        cam.horizontal = u * viewport_width * focus_dist;
        cam.vertical = v * viewport_height * focus_dist;
        cam.lower_left_corner =
            cam.origin - cam.horizontal / 2.0 - cam.vertical / 2.0 - w * focus_dist;
        //println!("{cam:?}");
        cam.lens_radius = aperture / 2.0;

        cam
    }
    #[inline(always)]
    fn get_ray(&self, u: NumberType, v: NumberType) -> Ray {
        //let rd = Vec3::random_in_unit_disk()*self.lens_radius;
        //let offset = rd*Vec3::new(u,v,0.0);
        ////let offset = u*rd.x + v*rd.y;
        ////let offset = Vec3::new(offset,0.0,0.0);
        ////

        //// rd is mostly a matrix multiplication, (u,v)*[horizontal,vertical]
        //Ray::new(
        //    self.origin + offset,
        //    self.lower_left_corner + self.horizontal*u + self.vertical*v - self.origin - offset,
        //)

        Ray::new(
            self.origin,
            self.lower_left_corner + self.horizontal * u + self.vertical * v - self.origin,
        )
    }
}

struct Scene {
    samples: u32,
    max_depth: u32,
    cam: Camera,
    imgx: usize,
    imgy: usize,
}

type ColorValueType = u8;
use rayon::prelude::*;
//use rayon::iter::IntoParallelIterator;
//use rayon::iter::ParallelIterator;
//use rayon::iter::IndexedParallelIterator;

use image::RgbImage;
use std::sync::Mutex;

pub fn render<'a, const LEN: usize>(
    lights: &'a HittableObject<'a>,
    bvh2: &BVHHeap<'a, LEN>,
    samples: u32,
) {
    println!("render tiem");
    //let scene;

    const ASPECT_RATIO: NumberType = 1.0; //16.0/9.0;
    const IMGX: usize = 512; //2160;
    const IMGY: usize = ((IMGX as NumberType) / ASPECT_RATIO) as usize;

    //let mut imgbuf = image::ImageBuffer::<image::Rgb<u8>>::new(IMGX as u32, IMGY as u32);
    //let imgbuf = image::ImageBuffer::new(30,30);
    let mut imgbuf = RgbImage::new(IMGX as u32, IMGY as u32);
    let cam = Camera::new(
        Vec3::new(278.0, 278.0, -800.0),
        Vec3::new(278.0, 278.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        40.0,
        ASPECT_RATIO,
        0.0,
        200.0,
    );

    let scene = Scene {
        samples,
        max_depth: 8, //8,//16,
        cam,
        imgx: IMGX,
        imgy: IMGY,
    };

    const IMGBUFX_SIZE: usize = IMGX * 3;

    static mut BUFFER: [[ColorValueType; IMGBUFX_SIZE]; IMGY] = [[0; IMGBUFX_SIZE]; IMGY];

    let rows_rendered = Mutex::new(0);
    let rows_rendered = &rows_rendered;

    // Multithreaded
    let start = Instant::now();
    unsafe {
        BUFFER.par_iter_mut()
    //BUFFER.iter_mut()
        //.with_min_len(64)
        .zip((0u32..u32::MAX).into_iter())
        .for_each(|(line,y)| {
            render_line(line, lights, bvh2, y, &scene);
            let mut num = rows_rendered.lock().unwrap();
            *num += 1;
            let f = (*num as NumberType)/(IMGY as NumberType);
            let p = f*100.0;
            let duration = (start.elapsed().as_millis() as NumberType)/1000.0;
            let eta_total_expect = duration/f;
            let eta = eta_total_expect-duration;
            println!("row {num}/{IMGY}: {p:.2}%, duration: {duration:.0} seconds, ETA: {eta:.0}/{eta_total_expect:.0} seconds");
        });
    }
    let duration = (start.elapsed().as_millis() as NumberType) / 1000.0;

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let xindex = x as usize * 3;
        unsafe {
            *pixel = image::Rgb([
                BUFFER[y as usize][xindex],
                BUFFER[y as usize][xindex + 1],
                BUFFER[y as usize][xindex + 2],
            ]);
        }
    }

    let num_objects = LEN / 2;
    let imgxy = IMGX * IMGY;

    let time_per_sample = duration / samples as NumberType;
    let sample_per_time = 1.0 / time_per_sample;
    let time_per_sample_per_resolution = time_per_sample * (600.0 * 600.0) / (imgxy as NumberType);

    println!();
    println!("Done!");
    println!();
    println!("objects: {num_objects}");
    println!("resolution: {IMGX}*{IMGY} = {imgxy}");
    println!("Samples: {samples}");
    println!("num_objects: {num_objects}");
    println!("duration: {duration} seconds");
    println!("sample/time: {sample_per_time} per second");
    println!("time/sample: {time_per_sample} seconds");
    println!("time/sample/resolution*(600*600): {time_per_sample_per_resolution} seconds");

    imgbuf.save("output.png").unwrap();
}

fn render_line<'a, const LEN: usize>(
    buffer: &mut [ColorValueType],
    lights: &'a HittableObject<'a>,
    bvh2: &BVHHeap<'a, LEN>,
    y: u32,
    scene: &Scene,
) {
    //let rng = rng();

    for x in 0..scene.imgx {
        let mut col = Vec3::default();
        for _ in 0..scene.samples {
            let u = (x as NumberType + random_val()) / (scene.imgx as NumberType - 1.0);
            let v = 1.0 - (y as NumberType + random_val()) / (scene.imgy as NumberType - 1.0);

            let ray = scene.cam.get_ray(u, v);

            let mut ray_col = ray_color(ray, bvh2, lights, scene.max_depth);

            if ray_col.x.is_nan() {
                ray_col.x = 0.0;
            }
            if ray_col.y.is_nan() {
                ray_col.y = 0.0;
            }
            if ray_col.z.is_nan() {
                ray_col.z = 0.0;
            }

            col += ray_col;
        }
        col = col / (scene.samples as NumberType);
        let r = (col.x.sqrt() * 255.999) as u8;
        let g = (col.y.sqrt() * 255.999) as u8;
        let b = (col.z.sqrt() * 255.999) as u8;

        let xindex = x * 3;
        buffer[xindex] = r;
        buffer[xindex + 1] = g;
        buffer[xindex + 2] = b;
    }
}

//#[inline(always)]
fn ray_color<'a, const LEN: usize>(
    ray: Ray,
    world: &BVHHeap<LEN>,
    light: &'a HittableObject<'a>,
    depth: u32,
) -> Vec3 {
    #[inline(always)]
    fn ray_color_rec<'a, const LEN: usize>(
        mut ray: Ray,
        world: &BVHHeap<LEN>,
        light: &'a HittableObject<'a>,
        mut depth: u32,
        mut tcol: Vec3, // total color, sum of all light
        mut fcol: Vec3, // product of all color
    ) -> Vec3 {
        if true {
            loop {
                if depth == 0 {
                    return tcol;
                }
                depth -= 1;

                let rec = match world.hit(&ray, Interval::EPSILON_UNIVERSE) {
                    None => return Vec3::new(0.0, 0.0, 0.0),
                    Some(rec) => rec,
                };

                //let mut scattered = Ray::default();
                //let mut pdf = 0.0;
                //let mut albedo = Vec3::one(0.0);

                let emitted = rec.material.emission();

                let mut srec: ScatterRecord = ScatterRecord::default();
                if !rec.material.scatter(&ray, &rec, &mut srec) {
                    // a material that terminates rays should not have any direct lighting calc
                    return fcol * emitted;
                }

                // surface color
                let scol = srec.attenuation;

                // direct lighting to this point (may include direct lighting rays)
                let dcol = emitted;

                fcol = fcol * scol;
                tcol += fcol * dcol;

                match srec.scatter {
                    ScatterEnum::RaySkip { ray: scattered } => {
                        ray = scattered;
                    }
                    ScatterEnum::Pdf { pdf } => {
                        let pdf_material = PDFEnum::PDFMaterialEnum(pdf);

                        //let p = CosinePDF::new(rec.n);
                        //scattered = Ray{ro: rec.p, rd: p.generate()};
                        //pdf = p.value(scattered.rd);

                        //let light_pdf = HittablePDF::new(light, rec.p);
                        //let scattered = Ray{ro: rec.p, rd:light_pdf.generate()};
                        //let pdf = light_pdf.value(scattered.rd);

                        let pdf_light = HittablePDF::create(light, rec.p);
                        //let pdf_material = PDFEnum::PDFMaterialEnum(CosinePDF::create(rec.n));
                        //let mix_pdf = pdf_light;
                        let mix_pdf = MixPDF::new(&pdf_light, &pdf_material, 0.5);
                        //let c1 = {
                        ////let mix_pdf = pdf_light;
                        //let scattered = Ray{ro: rec.p, rd: mix_pdf.generate()};
                        //let pdf = mix_pdf.value(scattered.rd);
                        //dcol
                        //    + albedo*rec.material.scattering_pdf(ray, &rec, &scattered)
                        //    *ray_color(&scattered, world, light, (depth-1).min(1), acc)/pdf};

                        //let mix_pdf = pdf_cosine;
                        let scattered = Ray::new(rec.p, mix_pdf.generate());

                        let pdf = mix_pdf.value(scattered.rd);

                        let pdf_factor = rec.material.scattering_pdf(&ray, &rec, &scattered) / pdf;

                        ray = scattered;
                        fcol = fcol * pdf_factor;
                    }
                }
                //ray_color_rec(ray, world, light, depth, tcol, fcol)
            }
        } else {
            let rec = match world.hit(&ray, Interval::EPSILON_UNIVERSE) {
                None => return Vec3::new(0.0, 0.0, 0.0),
                Some(rec) => rec,
            };
            (rec.n + Vec3::one(1.0)) * rec.t * 0.1
        }
    }

    ray_color_rec(ray, world, light, depth, Vec3::one(0.0), Vec3::one(1.0))
}
