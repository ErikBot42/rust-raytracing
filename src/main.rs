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
use ordered_float::OrderedFloat;
use std::cell::RefCell;
use std::time::Instant;


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
    fn at(&self, t: NumberType) -> Vec3 { self.ro + self.rd*t }
}

fn ray_color(ray: &Ray, world: &dyn Hittable, depth: u32, acc: Vec3) -> Vec3 {
    if depth==0 {return Vec3::default();}

    let mut rec = HitRecord::default();
    if !world.hit(ray, 0.001, NumberType::INFINITY, &mut rec) {
        return Vec3::one(0.01) // sky
    }

    let mut scattered = Ray::default();
    let mut attenuation = Vec3::one(0.0);
    let emitted = rec.material.emission();

    if !rec.material.scatter(ray, &rec, &mut attenuation, &mut scattered) {return emitted;}

    return emitted + attenuation*ray_color(&scattered, world, depth-1, acc);
}


#[derive(Default)]
struct HitRecord {
    p: Vec3,
    n: Vec3,
    material: MaterialEnum,
    t: NumberType,
    front_face: bool,
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
    fn bounding_box(&self, _aabb: &mut AABB) -> bool {false}  
}


//impl Default for BVHnode2<'_> {fn default() -> Self {BVHnode2::Tail}}
//impl BVHnode2<'a> {
//    fn add(&'a self, aabb: AABB) -> Self {
//        BVHnode2::Node{aabb, left: self, right: self}
//    } 
//}

enum HittableObject<'a> {
    Sphere(Sphere),
    BVHnode(BVHnode),
    XYRect(XYRect),
    XZRect(XZRect),
    YZRect(YZRect),
    ConstantMedium(ConstantMedium<'a>),
}

impl<'a> Hittable for HittableObject<'a>
{
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool {
        match self {
            HittableObject::Sphere(s) => s.hit(ray, t_min, t_max, rec),
            HittableObject::BVHnode(b) => b.hit(ray, t_min, t_max, rec),
            HittableObject::XYRect(xy) => xy.hit(ray, t_min, t_max, rec),
            HittableObject::XZRect(xz) => xz.hit(ray, t_min, t_max, rec),
            HittableObject::YZRect(yz) => yz.hit(ray, t_min, t_max, rec),
            HittableObject::ConstantMedium(c) => c.hit(ray, t_min, t_max, rec),
        }
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        match self {
            HittableObject::Sphere(s) => s.bounding_box(aabb),
            HittableObject::BVHnode(b) => b.bounding_box(aabb),
            HittableObject::XYRect(xy) => xy.bounding_box(aabb),
            HittableObject::XZRect(xz) => xz.bounding_box(aabb),
            HittableObject::YZRect(yz) => yz.bounding_box(aabb),
            HittableObject::ConstantMedium(c) => c.bounding_box(aabb),
        }
    }
}

struct XYRect {
    x0: NumberType,
    x1: NumberType,
    y0: NumberType,
    y1: NumberType,
    k: NumberType,
    material: MaterialEnum,
}

impl Hittable for XYRect {
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        aabb.minimum = Vec3::new(
            self.x0,
            self.y0,
            self.k-0.0001);
        aabb.maximum= Vec3::new(
            self.x0,
            self.y0,
            self.k-0.0001);
        true
    }
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool {
        let t = (self.k - ray.ro.z)/ray.rd.z;
        if t<t_min || t>t_max {return false;}

        let x = ray.ro.x + t*ray.rd.x;
        let y = ray.ro.y + t*ray.rd.y;

        if x<self.x0 || x>self.x1 || y<self.y0 || y>self.y1 {return false;}
        
        //TODO uv

        rec.t = t;
        let outward_normal = Vec3::new(0.0,0.0,1.0);
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material;
        rec.p = ray.at(t);
        true

    }
}


struct XZRect {
    x0: NumberType,
    x1: NumberType,
    z0: NumberType,
    z1: NumberType,
    k: NumberType,
    material: MaterialEnum,
}

impl Hittable for XZRect {
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        aabb.minimum = Vec3::new(
            self.x0,
            self.z0,
            self.k-0.0001);
        aabb.maximum= Vec3::new(
            self.x0,
            self.z0,
            self.k-0.0001);
        true
    }
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool {
        let t = (self.k - ray.ro.y)/ray.rd.y;
        if t<t_min || t>t_max {return false;}

        let x = ray.ro.x + t*ray.rd.x;
        let z = ray.ro.z + t*ray.rd.z;

        if x<self.x0 || x>self.x1 || z<self.z0 || z>self.z1 {return false;}
        
        //TODO uv

        rec.t = t;
        let outward_normal = Vec3::new(0.0,1.0,0.0);
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material;
        rec.p = ray.at(t);
        true

    }
}


struct YZRect {
    y0: NumberType,
    y1: NumberType,
    z0: NumberType,
    z1: NumberType,
    k: NumberType,
    material: MaterialEnum,
}

impl Hittable for YZRect {
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        aabb.minimum = Vec3::new(
            self.y0,
            self.z0,
            self.k-0.0001);
        aabb.maximum= Vec3::new(
            self.y0,
            self.z0,
            self.k-0.0001);
        true
    }
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool {
        let t = (self.k - ray.ro.x)/ray.rd.x;
        if t<t_min || t>t_max {return false;}

        let y = ray.ro.y + t*ray.rd.y;
        let z = ray.ro.z + t*ray.rd.z;

        if y<self.y0 || y>self.y1 || z<self.z0 || z>self.z1 {return false;}
        
        //TODO uv

        rec.t = t;
        let outward_normal = Vec3::new(1.0,0.0,0.0);
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material;
        rec.p = ray.at(t);
        true

    }
}

struct Sphere {
    center: Vec3,
    radius: NumberType,
    material: MaterialEnum,//Rc<dyn Material>,
}

impl Default for Sphere{
    fn default() -> Self {
        Sphere
        {
            center: Vec3::default(),
            radius: NumberType::default(),
            material: MaterialEnum::Lambertian(Lambertian{albedo: Vec3::default()}),
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
        println!("sphere bounding box");
        aabb.maximum = self.center+Vec3::one(self.radius);
        aabb.maximum = self.center+Vec3::one(self.radius);
        true
    }
}

//struct Cuboid {
//    center: Vec3,
//    dim: Vec3,
//    material: MaterialEnum
//}
//
//impl Hittable for Cuboid {
//    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool
//    {
//        let m = Vec3::new(1.0/ray.rd.x, 1.0/ray.rd.y, 1.0/ray.rd.z);
//        
//
//        true
//    }
//}

struct ConstantMedium<'a> {
    neg_inv_denisty: NumberType,
    boundary: &'a HittableObject<'a>,
    material: MaterialEnum,
}

impl<'a> Hittable for ConstantMedium<'a> {
    fn hit(&self, ray: &Ray, t_min: NumberType, t_max: NumberType, rec: &mut HitRecord) -> bool
    {
        let enable_debug = false;
        let debugging = enable_debug && random_val()<0.00001;
        
        let mut rec1 = HitRecord::default();
        let mut rec2 = HitRecord::default();

        if !self.boundary.hit(ray, -NumberType::INFINITY, NumberType::INFINITY, &mut rec1) {return false;}
        if !self.boundary.hit(ray, rec1.t+0.0001, NumberType::INFINITY, &mut rec2) {return false;}
        
        if debugging {println!("t_min: {t_min}, t_max: {t_max}");}
        
        if rec1.t<t_min {rec1.t = t_min}
        if rec2.t>t_max {rec2.t = t_max}
        
        if rec1.t >= rec2.t {return false;}
        
        if rec1.t<0.0 {rec1.t=0.0;}


        let raylen = ray.rd.length();
        let dist_in_boundary = (rec2.t-rec1.t)*raylen;
        let hit_dist = self.neg_inv_denisty*random_val().ln();
        
        if hit_dist>dist_in_boundary {return false;}

        rec.t = rec1.t + hit_dist/raylen;
        rec.p = ray.at(rec.t);

        rec.n = Vec3::new(1.0,0.0,0.0);
        rec.material = self.material;
        true
    }
    fn bounding_box(&self,  aabb: &mut AABB) -> bool {
        self.boundary.bounding_box(aabb)
    }

}

#[derive(Copy,Clone)]
struct Isotropic {
    albedo: Vec3,
}
impl Material for Isotropic {
    fn scatter(&self,_ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool {
        *sray = Ray {ro: rec.p, rd:Vec3::random_unit()};
        *attenuation = self.albedo;
        true 
    }
}

trait Material {
    fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _attenuation: &mut Vec3, _sray: &mut Ray) -> bool {false}
//    fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _attenuation: &mut Vec3, _sray: &mut Ray) -> bool {false}
//    fn scattering_pdf(&self, ray, 
    fn emission(&self) -> Vec3 {Vec3::one(0.0)}
}

#[derive(Copy,Clone)]
enum MaterialEnum {
    Lambertian(Lambertian),
    Metal(Metal),
    Dielectric(Dielectric),
    Emissive(Emissive),
    Isotropic(Isotropic),
}

impl Default for MaterialEnum {
    fn default() -> Self {
        MaterialEnum::Lambertian(Lambertian{albedo: Vec3::default()})
    }
}

impl Material for MaterialEnum
{
    fn scatter(&self,ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
    {
        match self {
            MaterialEnum::Metal(m) => m.scatter(ray, rec, attenuation, sray),
            MaterialEnum::Lambertian(l) => l.scatter(ray, rec, attenuation, sray),
            MaterialEnum::Dielectric(d) => d.scatter(ray, rec, attenuation, sray),
            MaterialEnum::Emissive(e) => e.scatter(ray, rec, attenuation, sray),
            MaterialEnum::Isotropic(i) => i.scatter(ray, rec, attenuation, sray),
        }
    }
    fn emission(&self) -> Vec3 {
        match self {
            MaterialEnum::Metal(m) => m.emission(),
            MaterialEnum::Lambertian(l) => l.emission(),
            MaterialEnum::Dielectric(d) => d.emission(),
            MaterialEnum::Emissive(e) => e.emission(),
            MaterialEnum::Isotropic(i) => i.emission(),
        }
    }
}
#[derive(Copy,Clone,Default)]
struct Lambertian {
    albedo: Vec3,
}
impl Material for Lambertian {
    fn scatter(&self,_ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
    {
        sray.rd.set(rec.n + Vec3::random_unit());
        sray.ro.set(rec.p);
        attenuation.set(self.albedo);
        true
    }
}

#[derive(Copy,Clone,Default)]
struct Emissive{
    light: Vec3,
}
impl Material for Emissive{
    fn emission(&self) -> Vec3 {self.light}
}

#[derive(Copy,Clone,Default)]
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



#[derive(Copy,Clone,Default)]
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
        cam.horizontal = u*viewport_width;
        cam.vertical = v*viewport_height;
        cam.lower_left_corner = cam.origin - cam.horizontal/2.0 - cam.vertical/2.0 - w;
        println!("{cam:?}");
        cam
    }
    fn get_ray(&self, u: NumberType, v: NumberType) -> Ray {
        Ray {
            ro: self.origin, 
            rd: self.lower_left_corner + self.horizontal*u + self.vertical*v - self.origin,
        }
    }
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
        let hit_left = self.left.borrow_mut().hit(ray, t_min, t_max, rec);
        let hit_right = self.right.borrow_mut().hit(ray, t_min, if hit_left {rec.t} else {t_max}, rec);
        return hit_left || hit_right;
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        aabb.minimum = self.aabb.minimum;
        aabb.maximum = self.aabb.maximum;
        println!("BVHnode bounding box");
        true
    }  
}

impl BVHnode {
    fn box_val<T: Hittable + ?Sized>(a: &Rc<RefCell<T>>, axis: u8) -> NumberType
    {
        let mut a_box = AABB::default();
        if !a.borrow_mut().bounding_box(&mut a_box) {panic!("missing implemenation for AABB");}
        else {a_box.minimum[axis]}
    }

    fn construct(&mut self, objects: Vec<Rc<RefCell<dyn Hittable>>>)
    {
        let mut copy = objects.clone();
        let axis: u8 = thread_rng().gen_range(0..3);

        println!("Axis: {axis}");
       
        let x = move |a:&Rc<RefCell<dyn Hittable>>| OrderedFloat(Self::box_val(a,axis));

        let object_span = copy.len();
        println!("object_span = {object_span}");

        if object_span == 0 {
            panic!("no elements when running construct");
        }
        if object_span == 1 {
            self.left  = copy[copy.len()-1].clone();
            self.right = copy[copy.len()-1].clone();
        }
        else if object_span == 2 {
            self.left  = copy[copy.len()-1].clone();
            self.right = copy[copy.len()-2].clone();
            //if x(&self.left) > x(&self.right)
            //{
            //    mem::swap(&mut self.left, &mut self.right);
            //}
        }
        else {
            copy.sort_by_key(x);
            let mid = object_span/2;
            let left_node = Rc::new(RefCell::new(BVHnode::default()));
            left_node.borrow_mut().construct(copy[mid..].to_vec());
            self.left = left_node;
            let right_node = Rc::new(RefCell::new(BVHnode::default()));
            right_node.borrow_mut().construct(copy[..mid].to_vec());
            self.right = right_node;
        }

        let mut box_left = AABB::default(); 
        let mut box_right = AABB::default(); 
        
        let has_left = self.left.borrow().bounding_box(&mut box_left);
        let has_right = self.right.borrow().bounding_box(&mut box_right);
        if !has_left || !has_right
        {
            panic!("AABB missing");
        }

        self.aabb = box_left.surrounding_box(box_right);
    }
}

fn main() {
    let aspect_ratio = 1.0;//16.0/9.0;
    let imgx         = 600;
    let imgy         = ((imgx as NumberType)/aspect_ratio) as u32;

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
    
    
    let mut object_list: Vec<Rc<RefCell<dyn Hittable>>> = Vec::new();

    let red   = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.65,0.05,0.05)});
    let white = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.73,0.73,0.73)});
    let green = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.12,0.45,0.15)});
    let light = MaterialEnum::Emissive(Emissive{light:Vec3::new(15.0,15.0,15.0)});
    //let blue  = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.12,0.12,0.45)});
    //let grey  = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.73,0.73,0.73)*0.4});
    //let light_red = MaterialEnum::Emissive(Emissive{light:Vec3::new(7.0,0.0,0.0)});
    //let light_blue = MaterialEnum::Emissive(Emissive{light:Vec3::new(0.0,0.0,7.0)});

    object_list.push(Rc::new(RefCell::new( YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, })));
    object_list.push(Rc::new(RefCell::new( YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, })));
    object_list.push(Rc::new(RefCell::new( XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, })));
    object_list.push(Rc::new(RefCell::new( XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, })));
    object_list.push(Rc::new(RefCell::new( XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, })));
    object_list.push(Rc::new(RefCell::new( XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, })));


    object_list.push(Rc::new(RefCell::new(
                XZRect{material: light, 
                    x0: 123.0,
                    x1: 423.0,
                    z0: 147.0, 
                    z1: 412.0,
                    k:  554.0, 
                })));
    
//    let material = Isotropic{albedo: Vec3::one(0.5)};
//    let fog_sphere = Rc::new(RefCell::new(
//            HittableObject::Sphere(
//                Sphere{
//                    material: white,
//                    radius: 5000.0,
//                    center: Vec3::one(0.0),
//                } )));
//
//    let fog_sphere = Rc::new(RefCell::new(
//                ConstantMedium{
//                    material: MaterialEnum::Isotropic(material),
//                    boundary: fog_sphere,
//                    neg_inv_denisty: -1.0/0.0001,
//                } ));
//    object_list.push(fog_sphere);
    
    let material = Lambertian{albedo: Vec3::new(0.5,0.7,0.2)};
    object_list.push(Rc::new(RefCell::new(
                Sphere{
                    material: MaterialEnum::Lambertian(material),
                    radius: 100.0,
                    center: Vec3::new(555.0/2.0+100.0,555.0/2.0-100.0,555.0/2.0),
                } )));

    
    
    let mut rng = thread_rng();

    if false { 
        for i in 0..10 {
            let rad = 50.0;
            let center = Vec3::new(random_range(0.0,350.0),random_range(0.0,350.0),random_range(0.0,350.0));
            let choose_mat = random_val();

            if choose_mat < 0.5 {
                let albedo = Vec3::random();
                let material = MaterialEnum::Lambertian(Lambertian {albedo});
                object_list.push(Rc::new(RefCell::new(Sphere{material, radius: rad, center})));
            }
            //else if choose_mat < 0.5 {
            //    let light = Vec3::random()*4.0;
            //    let material = MaterialEnum::Emissive(Emissive{light});
            //    object_list.push(Rc::new(RefCell::new(Sphere{material: material.clone(), radius: rad, center})));
            //}
            else if choose_mat < 0.75{
                let albedo = Vec3::random();
                let blur = random_range(0.0,0.5);
                let material = MaterialEnum::Metal(Metal{albedo,blur});
                object_list.push(Rc::new(RefCell::new(Sphere{material: material.clone(), radius: rad, center})));
            }
            else {
                let material = MaterialEnum::Dielectric(Dielectric{ir:1.5});
                object_list.push(Rc::new(RefCell::new(Sphere{material: material.clone(), radius: rad, center})));
            }

        }
    }
    let mut bvh = BVHnode::default();

    let num_objects = object_list.len();
    bvh.construct(object_list);
    
    //let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 45.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(26.0,3.0,6.0), Vec3::new(0.0,2.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    let cam = Camera::new(Vec3::new(278.0,278.0,-800.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(478.0,278.0,-600.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio);
    
    let start = Instant::now();

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        if x==0 {let f = y as NumberType/imgy as NumberType * 100.0;println!("row {y}/{imgy}: {f}%");}

        let samples   = 15;
        let max_depth = 32;

        let mut col = Vec3::default();

        for _ in 0..samples
        {
            let u =     (x as NumberType+rng.gen::<NumberType>()) / (imgx as NumberType - 1.0);
            let v = 1.0-(y as NumberType+rng.gen::<NumberType>()) / (imgy as NumberType - 1.0);
        
            let ray = cam.get_ray(u,v);
            col += ray_color(&ray, &bvh, max_depth, Vec3::one(0.0));
        }
        col=col/(samples as NumberType);

        let r = (col.x.sqrt()*255.999) as u8;
        let g = (col.y.sqrt()*255.999) as u8;
        let b = (col.z.sqrt()*255.999) as u8;

        *pixel = image::Rgb([r, g, b]);
    }
    let duration = (start.elapsed().as_millis() as NumberType)/1000.0;
    println!("Rendered {num_objects} objects in {duration} seconds");


    imgbuf.save("output.png").unwrap();
}
