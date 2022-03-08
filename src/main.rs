
extern crate image;
extern crate lazy_static;
extern crate rand;
extern crate smallvec;



//DivAssign,MulAssign
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign, Index};
use rand::{Rng,thread_rng};
use num_traits::real::Real;
use std::rc::Rc;
use std::mem;
use std::cmp::Ordering;
use ordered_float::OrderedFloat;
use std::cell::RefCell;

#[derive(Debug, Copy, Clone, Default)]
struct V3<T>
{
    x: T,
    y: T,
    z: T,
}

impl<T: Copy + Clone> V3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
    pub fn one(x: T) -> Self {
        Self { x, y:x, z:x }
    }
    fn set(&mut self,other: V3<T>)
    {
        self.x = other.x;
        self.y = other.y;
        self.z = other.z;
    }
}


impl<T> Neg for V3<T> 
where
T: Neg + Neg<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn neg(self) -> V3<T> {
        V3::new(-self.x, -self.y, -self.z)
    }
}

impl<T> Add for V3<T> 
where
T: Add + Add<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn add(self, other: V3<T>) -> V3<T> {
        V3::new(self.x+other.x, self.y+other.y, self.z+other.z)
    }
}

impl<T> AddAssign for V3<T> 
where
T: AddAssign + Copy + Clone {
    fn add_assign(&mut self, other: V3<T>) {
        self.x+=other.x; self.y+=other.y; self.z+=other.z;
    }
}

impl<T> Sub for V3<T> 
where
T: Sub + Sub<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn sub(self, other: V3<T>) -> V3<T> {
        V3::new(self.x-other.x, self.y-other.y, self.z-other.z)
    }
}

impl<T> SubAssign<V3<T>> for V3<T> 
where
T: SubAssign + Copy + Clone {
    fn sub_assign(&mut self, other: V3<T>) {
        self.x-=other.x; self.y-=other.y; self.z-=other.z;
    }
}

impl<T> Mul<T> for V3<T> 
where
T: Mul + Mul<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn mul(self, other: T) -> V3<T> {
        V3::new(self.x*other, self.y*other, self.z*other)
    }
}

impl<T> Mul for V3<T> 
where
T: Mul + Mul<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn mul(self, other: V3<T>) -> V3<T> {
        V3::new(self.x*other.x, self.y*other.y, self.z*other.z)
    }
}

impl<T> Div<T> for V3<T> 
where
T: Div + Div<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn div(self, other: T) -> V3<T> {
        V3::new(self.x/other, self.y/other, self.z/other)
    }
}

impl<T> Div for V3<T> 
where
T: Div + Div<Output = T> + Copy + Clone {
    type Output = V3<T>; 
    fn div(self, other: V3<T>) -> V3<T> {
        V3::new(self.x/other.x, self.y/other.y, self.z/other.z)
    }
}

impl<T> Index<u8> for V3<T>
{
    type Output = T;
    fn index(&self, index: u8) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => &self.z,
        }
    }
}

impl<T> V3<T> 
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy + Clone
{
    fn dot(self, other: V3<T>) -> T {
        let tmp = self*other;
        tmp.x+tmp.y+tmp.z
    }
    fn dot2(self) -> T {
        self.dot(self)
    }
    fn cross(self, other: V3<T>) -> Self
    {
        V3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
            )
    }
}

impl<T> V3<T> 
where
    T: Mul<T, Output = T> + Add<T, Output = T> + Copy + Clone + Real
{
    fn length(self) -> T {
        self.dot2().sqrt()
    }
    fn normalized(self) -> V3<T> {
        self/self.length()         
    }
    fn reflect(self, n: V3<T>) -> V3<T>
    {
        self-n*(self.dot(n)+self.dot(n))
    }
}

impl Vec3 {
    fn random() -> V3<NumberType>
    {
        V3::new(random_val(), random_val(), random_val()) 
    }
    fn random_range(a: NumberType, b:NumberType) -> V3<NumberType>
    {
        let mut q = rand::thread_rng();  
        V3::new(q.gen_range(a..b), q.gen_range(a..b), q.gen_range(a..b)) 
    }
    fn random_unit() -> V3<NumberType>
    {
        Self::random_range(-1.0,1.0).normalized()
    }
    fn refract(self, n: Vec3, etiot: NumberType) -> Vec3
    {
        let cos_theta = -self.dot(n).min(1.0);
        let r_out_prep = (self + n*cos_theta)*etiot;
        let r_out_parallel = n*(-(1.0 - r_out_prep.dot2()).abs().sqrt());
        r_out_prep+r_out_parallel
    }
}



type NumberType = f64;
type Vec3 = V3<NumberType>;

fn random_range(a: NumberType, b:NumberType) -> NumberType {
    let a: NumberType = rand::thread_rng().gen_range(a..b);a
}
fn random_val() -> NumberType {let a: NumberType = rand::thread_rng().gen();a}

#[derive(Debug, Default)]
struct Ray {
    ro: Vec3,
    rd: Vec3,
}

impl Ray {
    fn new (ro: Vec3, rd: Vec3) -> Self { Self {ro, rd} }
    fn at(&self, t: NumberType) -> Vec3 { self.ro + self.rd*t }
}

fn ray_color(ray: &Ray, world: &dyn Hittable, depth: u32, acc: Vec3) -> Vec3 {
    //let acc = acc+Vec3::one(0.1);
    if depth==0 {return Vec3::default();}
    let mut rec = HitRecord::default();
    if world.hit(ray, 0.001, NumberType::INFINITY, &mut rec) {
        let mut scattered = Ray::default();
        let mut attenuation = Vec3::default();

        let icol = if rec.material.scatter(ray, &rec, &mut attenuation, &mut scattered)
        {ray_color(&scattered, world, depth-1, acc)}
        else {Vec3::new(0.0,0.0,0.0)};

        return attenuation*icol + rec.material.emission();
    }
    let unit_dir = ray.rd.normalized();
    let t = (unit_dir.y + 1.0)*0.5;
    (Vec3::new(1.0,1.0,1.0)*(1.0-t) + Vec3::new(0.5,0.7,1.0)*t)*0.3
}

struct HitRecord2 {
    p: Vec3,
    n: Vec3,
    material: MaterialEnum,
    t: NumberType,
    front_face: bool,
}



struct HitRecord {
    p: Vec3,
    n: Vec3,
    material: Rc<dyn Material>,
    t: NumberType,
    front_face: bool,
}

impl Default for HitRecord {
    fn default() -> Self {
        HitRecord
        {
            p: Vec3::default(),
            n: Vec3::default(),
            material: Rc::new(Lambertian{albedo: Vec3::default()}),
            t: NumberType::default(),
            front_face: bool::default(),
        }
    }
}

#[derive(Default)]
struct HittableObjectList 
{
    l: Vec<Box<dyn Hittable>>
}


impl Hittable for HittableObjectList {
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool
    {
        let mut hit = false;
        let mut closest = t_max;
        for object in &self.l {
            if object.hit(ray, t_min, closest, rec) {
                hit = true;
                closest = rec.t;
            }
        }
        hit
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        if self.l.len()>0 {


            true
        }
        else {
            false
        }
    }
}

impl HitRecord {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3)
    {
        self.front_face = ray.rd.dot(outward_normal) < 0.0;
        self.n = if self.front_face {outward_normal} else {-outward_normal}
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool;
    fn bounding_box(&self, aabb: &mut AABB) -> bool {false}  
}

struct Sphere {
    center: Vec3,
    radius: NumberType,
    material: Rc<dyn Material>,
}

impl Default for Sphere{
    fn default() -> Self {
        Sphere
        {
            center: Vec3::default(),
            radius: NumberType::default(),
            material: Rc::new(Lambertian{albedo: Vec3::default()}),
        }
    }
}


impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool
    {
        let oc = ray.ro - self.center;
        let a = ray.rd.dot2();
        let b = oc.dot(ray.rd);
        let c = oc.dot2()-self.radius.powi(2);
        let discriminant = b*b - a*c;
        if discriminant < 0.0 {return false;}
        let sqrtd = discriminant.sqrt();
        let mut root = (-b-sqrtd)/a;
        if root<t_min  || t_max<root {
            root = (-b + sqrtd)/a;
            if root<t_min || t_max<root {return false;}
        }
        rec.t = root;
        rec.p = ray.at(root);
        let outward_normal = (rec.p - self.center)/self.radius;
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material.clone();
        true
    }
    fn bounding_box(&self,  aabb: &mut AABB) -> bool
    {
        aabb.maximum = self.center+Vec3::one(self.radius);
        aabb.maximum = self.center+Vec3::one(self.radius);
        true
    }
}

trait Material {
    fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _attenuation: &mut Vec3, _sray: &mut Ray) -> bool {false}
    fn emission(&self) -> Vec3 {Vec3::one(0.0)}
}

#[derive(Copy, Clone)]
enum MaterialEnum {
    Lambertian,
    Metal,
    Dielectric,
    Emissive,
}

#[derive(Default)]
struct Lambertian {
    albedo: Vec3,
}
impl Material for Lambertian {
    fn scatter(&self,_ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
    {
        sray.rd.set(rec.n + Vec3::random_unit()*0.1);
        sray.ro.set(rec.p);
        attenuation.set(self.albedo);
        true
    }
}

#[derive(Default)]
struct Emissive{
    light: Vec3,
}
impl Material for Emissive{
    fn emission(&self) -> Vec3 {self.light}
}

#[derive(Default)]
struct Metal {
    albedo: Vec3,
    blur: NumberType,
}
impl Material for Metal{
    fn scatter(&self,ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
    {
        sray.rd.set(ray.rd.normalized().reflect(rec.n)+Vec3::random_unit()*self.blur);
        sray.ro.set(rec.p);
        attenuation.set(self.albedo);
        sray.rd.dot(rec.n)>0.0
    }
}

#[derive(Default)]
struct Dielectric {
    ir: NumberType,
}
impl Dielectric {
    fn reflectance(cosine: NumberType, ref_idx: NumberType ) -> NumberType
    {
        let mut r0 = (1.0-ref_idx)/(1.0+ref_idx);
        r0 = r0*r0;
        return r0 + (1.0-r0)*(1.0-cosine).powi(5);
    }
}

impl Material for Dielectric{
    fn scatter(&self,ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
    {
        attenuation.set(Vec3::one(0.9));
        let refraction_ratio = if rec.front_face {1.0/self.ir} else {self.ir};
        
        let unit_dir = ray.rd.normalized();

        let cos_theta = (-unit_dir.dot(rec.n)).min(1.0);
        let sin_theta = 1.0-cos_theta.powi(2);
        
        let direction: Vec3;
        if refraction_ratio * sin_theta > 1.0
        || Self::reflectance(cos_theta, refraction_ratio) > random_val(){
            direction = unit_dir.reflect(rec.n);
        }
        else
        {
            direction = unit_dir.refract(rec.n, refraction_ratio);
        }
        //let refracted = unit_dir.refract(rec.n, refraction_ratio);
        sray.rd.set(direction);
        sray.ro.set(rec.p);
        true
    }
}

#[derive(Clone, Copy, Default, Debug)]
struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

const PI: NumberType = 3.1415926535897932385;

fn deg2rad(deg: NumberType) -> NumberType {
    deg*PI/180.0
}

impl Camera {
    fn new(
        lookfrom: Vec3,
        lookat: Vec3,
        up: Vec3,
        fov: NumberType,
        aspect_ratio: NumberType,
        ) -> Self
    {
        //let theta = deg2rad(fov);
        //let h = (theta/2.0).tan();
        //let viewport_height = 2.0*h;
        //let viewport_width = aspect_ratio * viewport_height;

        //let w = (lookfrom-lookat).normalized();
        //let u = up.cross(w).normalized();
        //let v = w.cross(u);

        let theta = deg2rad(fov);
        let h = (theta/2.0).tan();
        let viewport_height = 2.0*h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (lookfrom-lookat).normalized();
        let u = up.cross(w).normalized();
        let v = w.cross(u);

        let mut cam: Camera = Camera::default();
        cam.origin = lookfrom;
        cam.horizontal = u*viewport_width;
        cam.vertical = v*viewport_height;
        cam.lower_left_corner = cam.origin - cam.horizontal/2.0 - cam.vertical/2.0 - w;
        println!("{cam:?}");
        cam
    }
    fn get_ray(&self, u: NumberType, v: NumberType) -> Ray
    {
        Ray{
            ro: self.origin, 
            rd: self.lower_left_corner + self.horizontal*u + self.vertical*v - self.origin,
        }
    }
}

struct Test{
    x: i32,
    y: Option<Box<Test>>,
}

enum List<'a> {
    Node {
        data: i32,
        next: &'a List<'a>,
    },
    Tail,
}

#[derive(Copy, Clone, Default)]
struct AABB {
    maximum: Vec3,
    minimum: Vec3,
}

impl AABB {
    fn hit(&self, ray: &Ray, mut t_min: NumberType, mut t_max: NumberType) -> bool
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
    fn surrounding_box(&self, other: AABB) -> AABB
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
}

struct BVHnode {
    aabb: AABB,
    left: Rc<RefCell<dyn Hittable>>,
    right: Rc<RefCell<dyn Hittable>>,
}

impl Default for BVHnode{
    fn default() -> Self {
        BVHnode
        {
            aabb: AABB::default(),
            left: Rc::new(RefCell::new(Sphere::default())),
            right: Rc::new(RefCell::new(Sphere::default())),
        }
    }
}
impl Hittable for BVHnode {
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool {
        if !self.aabb.hit(ray, t_min, t_max) {return false;}
        let hit_left = self.left.hit(ray, t_min, t_max, rec);
        let hit_right = self.right.hit(ray, t_min, if hit_left {rec.t} else {t_max}, rec);
        return hit_left || hit_right;
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {false}  
}



impl BVHnode {
    fn box_val<T: Hittable + ?Sized>(a: &Rc<RefCell<T>>, axis: u8) -> NumberType
    {
        let mut a_box = AABB::default();
        if !a.borrow_mut().bounding_box(&mut a_box) {panic!("missing implemenation for AABB");}
        else {a_box.minimum[axis]}
    }
    //fn box_cmp<T: Hittable + ?Sized, U: Hittable + ?Sized>(a: Rc<T>, b: Rc<U>) -> Ordering{
    //    Self::box_val(a,0).partial_cmp(&Self::box_val(b,0)).unwrap()
    //}

    fn construct(&mut self, objects: Vec<Rc<RefCell<dyn Hittable>>>)
    {
        let mut copy = objects.clone();
        let axis: u8 = thread_rng().gen_range(0..3);
       

        let object_span = copy.len();
        if object_span == 1 {
            self.left  = copy[copy.len()-1].clone();
            self.right = copy[copy.len()-1].clone();
        }
        else if object_span == 2 {
            self.left  = copy[copy.len()-1].clone();
            self.right = copy[copy.len()-2].clone();
        }
        else {
            let x = move |a:&Rc<RefCell<dyn Hittable>>| OrderedFloat(Self::box_val(a,axis));
            copy.sort_by_key(x);
            let mid = object_span/2;
            let left_node = Rc::new(BVHnode::default());
            left_node.construct(copy);
            self.left = left_node;

        }
    }
}

fn main() {
    let aspect_ratio      = 3.0/2.0;//16.0/9.0;
    let imgx              = 400;
    let imgy              = ((imgx as NumberType)/aspect_ratio) as u32;

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);

    let mut world: HittableObjectList = HittableObjectList::default();
    
    let material = Metal{albedo:Vec3::new(0.5,0.5,0.5),blur:0.3};
    world.l.push(Box::new(Sphere {material: Rc::new(material), radius: 1000.0,center: Vec3::new(0.0,-1000.0,0.0)}));
    let material = Lambertian{albedo:Vec3::new(0.4,0.2,0.1)};
    world.l.push(Box::new(Sphere {material: Rc::new(material), radius: 1.0,center: Vec3::new(0.0,1.0,0.0)}));
    let material = Dielectric{ir:1.5};
    world.l.push(Box::new(Sphere {material: Rc::new(material), radius: 1.0,center: Vec3::new(-4.0,1.0,0.0)}));
    let material = Metal{albedo:Vec3::new(0.7,0.6,0.5), blur:0.0};
    world.l.push(Box::new(Sphere {material: Rc::new(material), radius: 1.0,center: Vec3::new(4.0,1.0,0.0)}));
    
    let mut rng = thread_rng();

    if true{ 
        let wid = 8;
        let sc = 1.3;
        for a in -wid..wid {
            for b in -wid..wid {
                let rad = random_range(0.1,0.4);
                let center = Vec3::new((a as NumberType + 0.9*random_val())*sc, rad, (b as NumberType + 0.9*random_val())*sc);
                if (center-Vec3::new(4.0,rad,0.0)).length() > 0.9
                {
                    let choose_mat = random_val();

                    if choose_mat < 0.25 {
                        let albedo = Vec3::random();
                        let material = Rc::new(Lambertian {albedo});
                        world.l.push(Box::new(Sphere{material: material.clone(), radius: rad, center}));
                    }
                    else if choose_mat < 0.5 {
                        let light = Vec3::random()*4.0;
                        let material = Rc::new(Emissive{light});
                        world.l.push(Box::new(Sphere{material: material.clone(), radius: rad, center}));
                    }
                    else if choose_mat < 0.75{
                        let albedo = Vec3::random();
                        let blur = random_range(0.0,0.5);
                        let material = Rc::new(Metal{albedo,blur});
                        world.l.push(Box::new(Sphere{material: material.clone(), radius: rad, center}));
                    }
                    else {
                        let material = Rc::new(Dielectric{ir:1.5});
                        world.l.push(Box::new(Sphere{material: material.clone(), radius: rad, center}));
                    }

                    //let material_right2  = Metal {albedo: Vec3::new(rng.gen::<NumberType>(),rng.gen::<NumberType>(),rng.gen::<NumberType>()), blur: rng.gen::<NumberType>()};
                    //let mat = Rc::new(material_right2);
                    //world.l.push(Box::new(Sphere {material: mat.clone(), radius: 0.2, center}));
                }
            }
        }
    }
    
    //let cam = Camera::new(Vec3::new(-2.0,2.0,1.0), Vec3::new(0.0,0.0,-1.0), Vec3::new(0.0,1.0,0.0), 90.0, aspect_ratio);
    let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    //let cam = Camera::new(Vec3::one(0.0), Vec3::new(0.0,0.0,-1.0), Vec3::new(0.0,1.0,0.0), 90.0, aspect_ratio);
    
    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        if x==0 {println!("row {y}/{imgy}");}

        let samples   = 100;
        let max_depth = 50;

        let mut col = Vec3::default();

        for _ in 0..samples
        {
            let u =     (x as NumberType+rng.gen::<NumberType>()) / (imgx as NumberType - 1.0);
            let v = 1.0-(y as NumberType+rng.gen::<NumberType>()) / (imgy as NumberType - 1.0);
        
            //cam.origin = origin;
            //cam.lower_left_corner = lower_left_corner;
            //cam.horizontal = horizontal;
            //cam.vertical = vertical;

            let ray = cam.get_ray(u,v);//Ray::new(origin, (lower_left_corner + horizontal*u + vertical*v - origin).normalized());
            col += ray_color(&ray, &world, max_depth, Vec3::one(0.0));
        }
        col=col/(samples as NumberType);

        let r = (col.x.sqrt()*255.999) as u8;
        let g = (col.y.sqrt()*255.999) as u8;
        let b = (col.z.sqrt()*255.999) as u8;

        *pixel = image::Rgb([r, g, b]);
    }

    imgbuf.save("output.png").unwrap();
}
