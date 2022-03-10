extern crate image;
extern crate lazy_static;
extern crate rand;
extern crate smallvec;

//DivAssign,MulAssign
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign, Index, IndexMut};
use rand::{Rng,thread_rng};
use rand_distr::StandardNormal;
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
    //fn set(&mut self,other: V3<T>) {
    //    self.x = other.x;
    //    self.y = other.y;
    //    self.z = other.z;
    //}
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

impl<T> Index<u8> for V3<T> {
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
impl<T> IndexMut<u8> for V3<T> {
    fn index_mut(&mut self, index: u8) -> &mut T{
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => &mut self.z,
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
    //fn reflect(self, n: V3<T>) -> V3<T>
    //{
    //    self-n*(self.dot(n)+self.dot(n))
    //}
}

impl Vec3 {
    //fn random() -> V3<NumberType> {
    //    V3::new(random_val(), random_val(), random_val()) 
    //}
    //fn random_range(a: NumberType, b:NumberType) -> V3<NumberType> {
    //    let mut q = rand::thread_rng();  
    //    V3::new(q.gen_range(a..b), q.gen_range(a..b), q.gen_range(a..b)) 
    //}
    //fn random_unit() -> V3<NumberType> {
    //    Self::random_range(-1.0,1.0).normalized()
    //}
    //fn random_in_unit_hemisphere(n:Vec3) -> V3<NumberType> {
    //    let r = Self::random_unit();
    //    if r.dot(n)>0.0 {r} else {-r}
    //}
    //
    fn random_dir() -> Vec3 {
        let mut rng = thread_rng();
        Vec3::new(rng.sample(StandardNormal),
        rng.sample(StandardNormal),
        rng.sample(StandardNormal)
        )
    } 
    fn random_unit() -> Vec3 {
        Vec3::random_dir().normalized()
    }
    fn random_cosine_direction() -> Vec3 {
        let mut v = Vec3::random_unit();
        v.z = v.z.abs();
        v
    }

    //fn random_cosine_direction() -> Vec3 {
    //    let mut rng = thread_rng();
    //    let mut v = Vec3::new(rng.sample(StandardNormal),
    //    rng.sample(StandardNormal),
    //    rng.sample(StandardNormal)
    //    );
    //    v.z = v.z.abs();

    //    v.normalized()
    //}
    //fn random_cosine_direction() -> Vec3 {
    //    let r1 = random_val();
    //    let r2 = random_val();
    //    let z = (1.0-r2).sqrt();

    //    let phi = 2.0*PI*r1;
    //    let x = phi.cos()*r2.sqrt();
    //    let y = phi.sin()*r2.sqrt();

    //    Vec3::new(x,y,z)

    //    //rand::distributions::Normal
    //}
    fn refract(self, n: Vec3, etiot: NumberType) -> Vec3 {
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

fn ray_color<'a>(
    ray: &Ray,
    world: &HittableObject,
    light: &'a HittableObject<'a>,
    depth: u32,
    acc: Vec3 //TODO iterative + background function
    ) -> Vec3 {
    if depth==0 {return Vec3::default();}

    let mut rec = HitRecord::default();
    if !world.hit(ray, Interval::new(0.001,NumberType::INFINITY), &mut rec) {
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


#[derive(Copy, Clone, Default)]
struct HitRecord<'a> {
    p: Vec3,
    n: Vec3,
    u: NumberType,
    v: NumberType,
    material: MaterialEnum<'a>,
    t: NumberType,
    front_face: bool,
}

impl<'a> HitRecord<'a> {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3)
    {
        self.front_face = ray.rd.dot(outward_normal) < 0.0;
        self.n = if self.front_face {outward_normal} else {-outward_normal}
    }
}

trait Hittable<'a> {

    // TODO: return option instead
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool;
    fn bounding_box(&self, _aabb: &mut AABB) -> bool {false}  

    fn pdf_value (&self, _o: Vec3,_v: Vec3) -> NumberType {panic!("pdf_value for Hittable is not implemented")}
    fn random (&self, _o: Vec3) -> Vec3 {panic!("random for Hittable is not implemented")}
}


//impl Default for BVHnode2<'_> {fn default() -> Self {BVHnode2::Tail}}
//impl BVHnode2<'a> {
//    fn add(&'a self, aabb: AABB) -> Self {
//        BVHnode2::Node{aabb, left: self, right: self}
//    } 
//}

#[derive(Clone)]
enum HittableObject<'a> {
    Sphere(Sphere<'a>),
    BVHnode(BVHnode<'a>),
    XYRect(XYRect<'a>),
    XZRect(XZRect<'a>),
    YZRect(YZRect<'a>),
    //ConstantMedium(ConstantMedium<'a>),
    Cuboid(Cuboid<'a>),
    Translate(Translate<'a>),
    RotateY(RotateY<'a>),
}

impl<'a> Hittable<'a> for HittableObject<'a>
{
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        match self {
            HittableObject::Sphere(s) => s.hit(ray, ray_t, rec),
            HittableObject::BVHnode(b) => b.hit(ray, ray_t, rec),
            HittableObject::XYRect(xy) => xy.hit(ray, ray_t, rec),
            HittableObject::XZRect(xz) => xz.hit(ray, ray_t, rec),
            HittableObject::YZRect(yz) => yz.hit(ray, ray_t, rec),
            //HittableObject::ConstantMedium(c) => c.hit(ray, ray_t, rec),
            HittableObject::Cuboid(u) => u.hit(ray, ray_t, rec),
            HittableObject::Translate(t) => t.hit(ray, ray_t, rec),
            HittableObject::RotateY(ry) => ry.hit(ray, ray_t, rec),
        }
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        match self {
            HittableObject::Sphere(s) => s.bounding_box(aabb),
            HittableObject::BVHnode(b) => b.bounding_box(aabb),
            HittableObject::XYRect(xy) => xy.bounding_box(aabb),
            HittableObject::XZRect(xz) => xz.bounding_box(aabb),
            HittableObject::YZRect(yz) => yz.bounding_box(aabb),
            //HittableObject::ConstantMedium(c) => c.bounding_box(aabb),
            HittableObject::Cuboid(u) => u.bounding_box(aabb),
            HittableObject::Translate(t) => t.bounding_box(aabb),
            HittableObject::RotateY(ry) => ry.bounding_box(aabb),
        }
    }
    fn pdf_value (&self, o: Vec3,v: Vec3) -> NumberType {
        match self {
            HittableObject::Sphere(s) => s.pdf_value(o,v),
            HittableObject::BVHnode(b) => b.pdf_value(o,v),
            HittableObject::XYRect(xy) => xy.pdf_value(o,v),
            HittableObject::XZRect(xz) => xz.pdf_value(o,v),
            HittableObject::YZRect(yz) => yz.pdf_value(o,v),
            //HittableObject::ConstantMedium(c) => c.pdf_value(o,v),
            HittableObject::Cuboid(u) => u.pdf_value(o,v),
            HittableObject::Translate(t) => t.pdf_value(o,v),
            HittableObject::RotateY(ry) => ry.pdf_value(o,v),
        }
    }
    fn random (&self, o: Vec3) -> Vec3 {
        match self {
            HittableObject::Sphere(s) => s.random(o),
            HittableObject::BVHnode(b) => b.random(o),
            HittableObject::XYRect(xy) => xy.random(o),
            HittableObject::XZRect(xz) => xz.random(o),
            HittableObject::YZRect(yz) => yz.random(o),
            //HittableObject::ConstantMedium(c) => c.random(o),
            HittableObject::Cuboid(u) => u.random(o),
            HittableObject::Translate(t) => t.random(o),
            HittableObject::RotateY(ry) => ry.random(o),
        }

    }
}

impl<'a> Default for HittableObject<'a> {
    fn default() -> Self {
        HittableObject::Sphere(Sphere::default())
    }
}

#[derive(Default, Clone, Copy)]
struct XYRect<'a> {
    x0: NumberType,
    x1: NumberType,
    y0: NumberType,
    y1: NumberType,
    k: NumberType,
    material: MaterialEnum<'a>,
}

impl<'a> Hittable<'a> for XYRect<'a> {
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
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        let t = (self.k - ray.ro.z)/ray.rd.z;
        if t<ray_t.min|| t>ray_t.max {return false;}

        let x = ray.ro.x + t*ray.rd.x;
        let y = ray.ro.y + t*ray.rd.y;

        if x<self.x0 || x>self.x1 || y<self.y0 || y>self.y1 {return false;}
        
        rec.u = (x-self.x0)/(self.x1-self.x0);
        rec.v = (y-self.y0)/(self.y1-self.y0);

        rec.t = t;
        let outward_normal = Vec3::new(0.0,0.0,1.0);
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material;
        rec.p = ray.at(t);
        true

    }
}


//struct Quad {
//    v: Vec3,
//    u: Vec3,
//    o: Vec3,
//}



#[derive(Default, Clone, Copy)]
struct XZRect<'a> {
    x0: NumberType,
    x1: NumberType,
    z0: NumberType,
    z1: NumberType,
    k: NumberType,
    material: MaterialEnum<'a>,
}

impl<'a> Hittable<'a> for XZRect<'a> {
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
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        let t = (self.k - ray.ro.y)/ray.rd.y;
        if t<ray_t.min || t>ray_t.max {return false;}

        let x = ray.ro.x + t*ray.rd.x;
        let z = ray.ro.z + t*ray.rd.z;

        if x<self.x0 || x>self.x1 || z<self.z0 || z>self.z1 {return false;}
        
        rec.u = (x-self.x0)/(self.x1-self.x0);
        rec.v = (z-self.z0)/(self.z1-self.z0);

        rec.t = t;
        let outward_normal = Vec3::new(0.0,1.0,0.0);
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material;
        rec.p = ray.at(t);
        true

    }
    fn pdf_value (&self, o: Vec3,v: Vec3) -> NumberType {
        let mut rec = HitRecord::default();
        if !self.hit(&Ray{ro: o, rd: v}, Interval::new(0.001, NumberType::INFINITY), &mut rec) {return 0.0}

        let area = (self.x1-self.x0)*(self.z1-self.z0);
        let dist2 = rec.t*rec.t*v.dot2();
        let cosine = v.dot(rec.n).abs()/v.length();

        dist2/(cosine*area)
    }
    fn random (&self, o: Vec3) -> Vec3 {
        let random_point = Vec3::new(random_range(self.x0,self.x1), self.k, random_range(self.z0, self.z1)); 
        random_point - o
    }
}


#[derive(Default, Clone, Copy)]
struct YZRect<'a> {
    y0: NumberType,
    y1: NumberType,
    z0: NumberType,
    z1: NumberType,
    k: NumberType,
    material: MaterialEnum<'a>,
}

impl<'a> Hittable<'a> for YZRect<'a> {
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
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        let t = (self.k - ray.ro.x)/ray.rd.x;
        if t<ray_t.min|| t>ray_t.max {return false;}

        let y = ray.ro.y + t*ray.rd.y;
        let z = ray.ro.z + t*ray.rd.z;

        if y<self.y0 || y>self.y1 || z<self.z0 || z>self.z1 {return false;}
        
        rec.u = (y-self.y0)/(self.y1-self.y0);
        rec.v = (z-self.z0)/(self.z1-self.z0);

        rec.t = t;
        let outward_normal = Vec3::new(1.0,0.0,0.0);
        rec.set_face_normal(ray, outward_normal);
        rec.material = self.material;
        rec.p = ray.at(t);
        true

    }
}

#[derive(Default, Clone, Copy)]
struct Sphere<'a> {
    center: Vec3,
    radius: NumberType,
    material: MaterialEnum<'a>,//Rc<dyn Material>,
}

impl<'a> Hittable<'a> for Sphere<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool
    {
        let oc = ray.ro - self.center;
        let a = ray.rd.dot2();
        let half_b = oc.dot(ray.rd);
        let c = oc.dot2()-self.radius.powi(2);
        let discriminant = half_b*half_b - a*c;
        if discriminant < 0.0 {return false;}
        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b-sqrtd)/a;
        if root<ray_t.min  || ray_t.max<root {
            root = (-half_b + sqrtd)/a;
            if root<ray_t.min || ray_t.max<root {return false;}
        }
        rec.t = root;
        rec.p = ray.at(root);
        let outward_normal = (rec.p - self.center)/self.radius;
        rec.set_face_normal(ray, outward_normal);
        self.get_uv(outward_normal, &mut rec.u, &mut rec.v);
        rec.material = self.material;
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

impl<'a> Sphere<'a> {
    fn get_uv(&self, p: Vec3, u: &mut NumberType, v: &mut NumberType)
    {
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + PI;

        *u = phi / (2.0*PI);
        *v = theta / PI;
    }
}

#[derive(Default, Clone)]
struct HittableList<'a> {
    l: Vec<HittableObject<'a>>,
}

impl<'a> Hittable<'a> for HittableList<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        let mut hit_anything = false;
        let mut closest = ray_t.max;
        for object in &self.l {
            if object.hit(ray, Interval::new(ray_t.min, closest), rec) {
                hit_anything = true;
                closest = rec.t;
            }
        }
        hit_anything
    }
}

#[derive(Default, Clone)]
struct Cuboid<'a> {
    min: Vec3,
    max: Vec3,
    sides: HittableList<'a>,
}
impl<'a> Cuboid<'a> {
    fn new(min: Vec3, max: Vec3, material: MaterialEnum<'a>) -> Self {
        let mut sides = HittableList::default();
        sides.l.push(HittableObject::XYRect(XYRect {x0: min.x, x1: max.x, y0: min.y, y1: max.y, k:max.z,material}));
        sides.l.push(HittableObject::XYRect(XYRect {x0: min.x, x1: max.x, y0: min.y, y1: max.y, k:min.z,material}));
        
        sides.l.push(HittableObject::XZRect(XZRect {x0: min.x, x1: max.x, z0: min.z, z1: max.z, k:max.y,material}));
        sides.l.push(HittableObject::XZRect(XZRect {x0: min.x, x1: max.x, z0: min.z, z1: max.z, k:min.y,material}));

        sides.l.push(HittableObject::YZRect(YZRect {y0: min.y, y1: max.y, z0: min.z, z1: max.x, k:max.x,material}));
        sides.l.push(HittableObject::YZRect(YZRect {y0: min.y, y1: max.y, z0: min.z, z1: max.x, k:min.x,material}));

        Cuboid {
            min,
            max,
            sides,
        }
    }
}

impl<'a> Hittable<'a> for Cuboid<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool
    {
        self.sides.hit(ray, ray_t, rec)
    }
    fn bounding_box(&self,  aabb: &mut AABB) -> bool {
        *aabb = AABB{minimum:self.min, maximum:self.max};
        true
    }
}

#[derive(Clone, Copy)]
struct ConstantMedium<'a> {
    neg_inv_denisty: NumberType,
    boundary: &'a HittableObject<'a>,
    material: MaterialEnum<'a>,
}

impl<'a> Hittable<'a> for ConstantMedium<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool
    {
        //let enable_debug = false;
        //let debugging = enable_debug && random_val()<0.00001;
        
        let mut rec1 = HitRecord::default();
        let mut rec2 = HitRecord::default();

        if !self.boundary.hit(ray, Interval::UNIVERSE, &mut rec1) {return false;}
        if !self.boundary.hit(ray, Interval::new(rec1.t+0.0001, NumberType::INFINITY), &mut rec2) {return false;}
        
        //if debugging {println!("t_min: {t_min}, t_max: {t_max}");}
        
        if rec1.t<ray_t.min {rec1.t = ray_t.min}
        if rec2.t>ray_t.max {rec2.t = ray_t.max}
        
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

//#[derive(Copy,Clone)]
//struct Isotropic {
//    albedo: Vec3,
//}
//impl Material for Isotropic {
//    fn scatter(&self,_ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray, pdf: &mut NumberType) -> bool {
//        *sray = Ray {ro: rec.p, rd:Vec3::random_unit()};
//        *attenuation = self.albedo;
//        true 
//    }
//}




#[derive(Copy,Clone,Default)]
struct Lambertian<'a> {
    texture: TextureEnum<'a>,
}
impl<'a> Material for Lambertian<'a> {
    fn scatter(&self,_ray: &Ray, rec: &HitRecord, albedo: &mut Vec3, sray: &mut Ray, pdf: &mut NumberType) -> bool
    {

        //sray.rd.set(rec.n + Vec3::random_unit());//Vec3::random_in_unit_hemisphere(rec.n));
        //sray.rd.set(Vec3::random_in_unit_hemisphere(rec.n));
        //sray.ro.set(rec.p);
        //*attenuation = self.texture.value(rec.u,rec.v,rec.p);
        
        
        //sray.rd = (rec.n + Vec3::random_unit()).normalized();
        //sray.ro = rec.p;
        //*albedo = self.texture.value(rec.u, rec.v, rec.p);
        //*pdf = rec.n.dot(sray.rd)/PI;
        
        //sray.rd = Vec3::random_in_unit_hemisphere(rec.n);
        //sray.ro = rec.p;
        //*albedo = self.texture.value(rec.u, rec.v, rec.p);
        //*pdf = 0.5/PI;
        
        let onb = ONB::build_from_w(rec.n);

        sray.rd = onb.local(Vec3::random_cosine_direction()).normalized();
        sray.ro = rec.p;
        *albedo = self.texture.value(rec.u, rec.v, rec.p);
        *pdf = onb.w.dot(sray.rd)/PI;
        true
    }
    fn scattering_pdf(&self, _ray: &Ray, rec: &HitRecord, sray: &Ray) -> NumberType {
        let cosine = rec.n.dot(sray.rd.normalized());
        if cosine < 0.0 {0.0} else {cosine/PI}
        //(cosine/PI).abs()
    }
}
impl<'a> Lambertian<'a> {
    fn new(texture: TextureEnum) -> MaterialEnum {
        MaterialEnum::Lambertian(Lambertian {texture,})
    } 
    fn col(color: Vec3) -> MaterialEnum<'a> {
        MaterialEnum::Lambertian(Lambertian {
            texture: SolidColor::new(color),
        }) 
    }
}

#[derive(Copy,Clone,Default)]
struct Emissive{
    light: Vec3,
}
impl Material for Emissive{
    fn emission(&self) -> Vec3 {self.light}
}

trait Material {
    fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _albedo: &mut Vec3, _scattered: &mut Ray, _pdf: &mut NumberType) -> bool {false}

    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _sray: &Ray) -> NumberType {0.0}//TODO

    fn emission(&self) -> Vec3 {Vec3::one(0.0)}
}

#[derive(Copy,Clone)]
enum MaterialEnum<'a> {
    Lambertian(Lambertian<'a>),
    Emissive(Emissive),
//    Metal(Metal),
//    Dielectric(Dielectric),
//    Isotropic(Isotropic),
}

impl<'a> Default for MaterialEnum<'a> {
    fn default() -> Self {
        MaterialEnum::Lambertian(Lambertian::default())
    }
}

impl<'a> Material for MaterialEnum<'a>
{
    fn scatter(&self,ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray, pdf: &mut NumberType) -> bool
    {
        match self {
            MaterialEnum::Lambertian(l) => l.scatter(ray, rec, attenuation, sray, pdf),
            MaterialEnum::Emissive(e) => e.scatter(ray, rec, attenuation, sray, pdf),
            //MaterialEnum::Metal(m) => m.scatter(ray, rec, attenuation, sray),
            //MaterialEnum::Dielectric(d) => d.scatter(ray, rec, attenuation, sray),
            //MaterialEnum::Isotropic(i) => i.scatter(ray, rec, attenuation, sray),
        }
    }
    fn emission(&self) -> Vec3 {
        match self {
            MaterialEnum::Lambertian(l) => l.emission(),
            MaterialEnum::Emissive(e) => e.emission(),
            //MaterialEnum::Metal(m) => m.emission(),
            //MaterialEnum::Dielectric(d) => d.emission(),
            //MaterialEnum::Isotropic(i) => i.emission(),
        }
    }
    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _sray: &Ray) -> NumberType {
        match self {
            MaterialEnum::Lambertian(l) => l.scattering_pdf(_ray, _rec, _sray),
            MaterialEnum::Emissive(e) => e.scattering_pdf(_ray, _rec, _sray),
        }
    }
}

trait Texture {
    fn value(&self, u: NumberType, v: NumberType, p: Vec3) -> Vec3; 
}
#[derive(Clone, Copy)]
enum TextureEnum<'a> {
    SolidColor(SolidColor),
    CheckerTexture(CheckerTexture<'a>),
}
impl<'a> Default for TextureEnum<'a> {
    fn default() -> Self {
        TextureEnum::SolidColor(SolidColor::default())
    }
}

impl<'a> Texture for TextureEnum<'a> {
    fn value(&self, u: NumberType, v: NumberType, p: Vec3) -> Vec3 {
        match self {
            TextureEnum::SolidColor(s) => s.value(u,v,p),
            TextureEnum::CheckerTexture(c) => c.value(u,v,p),
        }
    }
}
#[derive(Clone, Copy, Default)]
struct SolidColor {color_value: Vec3,}
impl Texture for SolidColor {fn value(&self, _u: NumberType, _v: NumberType, _p: Vec3) -> Vec3 {self.color_value}}
impl<'a> SolidColor {
    fn new(color_value: Vec3) -> TextureEnum<'a> {TextureEnum::SolidColor(SolidColor {color_value})}
}

// no default 
#[derive(Clone, Copy)]
struct CheckerTexture<'a> {
    odd: &'a TextureEnum<'a>,
    even: &'a TextureEnum<'a>,
}
impl<'a> Texture for CheckerTexture<'a> {
    fn value(&self, u: NumberType, v: NumberType, p: Vec3) -> Vec3 {
        let fac = 8.0;//0.2;
        //let sines = (fac*p.x).sin()*(fac*p.y).sin()*(fac*p.z).sin();
        //let sines = (fac*u).sin()*(fac*v).sin();
        let sines = ((u*fac).fract()*2.0-1.0)*((v*fac).fract()*2.0-1.0);
        if sines < 0.0 {self.even.value(u,v,p)} else {self.odd.value(u,v,p)}
    }
}
impl<'a> CheckerTexture<'a> {
    fn new(odd: &'a TextureEnum<'a>, even: &'a TextureEnum<'a>) -> TextureEnum<'a> {
        TextureEnum::CheckerTexture(CheckerTexture {odd,even,})
    }
    // inputting 2 arbitrary colors is not possible without allocation
}

//#[derive(Copy,Clone,Default)]
//struct Metal {
//    albedo: Vec3,
//    blur: NumberType,
//}
//impl Material for Metal{
//    fn scatter(&self,ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
//    {
//        sray.rd.set(ray.rd.normalized().reflect(rec.n)+Vec3::random_unit()*self.blur);
//        sray.ro.set(rec.p);
//        attenuation.set(self.albedo);
//        sray.rd.dot(rec.n)>0.0
//    }
//}

//#[derive(Copy,Clone,Default)]
//struct Dielectric {
//    ir: NumberType,
//}
//impl Dielectric {
//    fn reflectance(cosine: NumberType, ref_idx: NumberType ) -> NumberType
//    {
//        let mut r0 = (1.0-ref_idx)/(1.0+ref_idx);
//        r0 = r0*r0;
//        return r0 + (1.0-r0)*(1.0-cosine).powi(5);
//    }
//}
//impl Material for Dielectric{
//    fn scatter(&self,ray: &Ray, rec: &HitRecord, attenuation: &mut Vec3, sray: &mut Ray) -> bool
//    {
//        attenuation.set(Vec3::one(0.9));
//        let refraction_ratio = if rec.front_face {1.0/self.ir} else {self.ir};
//        
//        let unit_dir = ray.rd.normalized();
//
//        let cos_theta = (-unit_dir.dot(rec.n)).min(1.0);
//        let sin_theta = 1.0-cos_theta.powi(2);
//        
//        let direction: Vec3;
//        if refraction_ratio * sin_theta > 1.0
//        || Self::reflectance(cos_theta, refraction_ratio) > random_val(){
//            direction = unit_dir.reflect(rec.n);
//        }
//        else
//        {
//            direction = unit_dir.refract(rec.n, refraction_ratio);
//        }
//        //let refracted = unit_dir.refract(rec.n, refraction_ratio);
//        sray.rd.set(direction);
//        sray.ro.set(rec.p);
//        true
//    }
//}

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


// Translate incoming ray for object 
 
#[derive(Clone, Copy)]
struct Translate<'a> {
    offset: Vec3,
    object: &'a HittableObject<'a>,
}

impl<'a> Hittable<'a> for Translate<'a> {

    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        let moved = Ray{ro: ray.ro - self.offset, rd: ray.rd};
        if !self.object.hit(&moved, ray_t, rec) {return false;}
        rec.p+=self.offset;
        rec.set_face_normal(&moved, rec.n);
        true
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        if !self.object.bounding_box(aabb) {return false;}
        aabb.minimum += self.offset;
        aabb.maximum += self.offset;
        true
    }
}

// TODO: replace with general transform?
#[derive(Clone, Copy)]
struct RotateY<'a> {
    sin_theta: NumberType,
    cos_theta: NumberType,
    aabb: Option<AABB>,
    object: &'a HittableObject<'a>,
}
impl<'a> RotateY<'a> {
    fn new(object: &'a HittableObject<'a>, angle: NumberType) -> Self {
        let radians = deg2rad(angle);
        let sin_theta = radians.sin();
        let cos_theta = radians.cos();
        let mut aabb = AABB::default();
        let has_box = object.bounding_box(&mut aabb);
        let mut min = Vec3::one(NumberType::INFINITY);
        let mut max = Vec3::one(-NumberType::INFINITY);
        
        let aabb_o: Option<AABB>;
        if has_box {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let i = i as NumberType;
                        let j = j as NumberType;
                        let k = k as NumberType;
                        let x = i*aabb.minimum.x + (1.0-i)*aabb.minimum.x;
                        let y = j*aabb.minimum.x + (1.0-j)*aabb.minimum.z;
                        let z = k*aabb.minimum.x + (1.0-k)*aabb.minimum.z;

                        let newx = cos_theta*x + sin_theta*z;
                        let newz = -sin_theta*x + cos_theta*z;

                        let tester = Vec3::new(newx,y,newz);

                        for c in 0..3 {
                            min[c] = min[c].min(tester[c]);
                            max[c] = max[c].max(tester[c]);
                        }
                    }
                }
            }
            aabb_o = Some(aabb);
        } else {aabb_o = None;}

        RotateY {
            sin_theta,
            cos_theta,
            aabb: aabb_o,
            object,
        }
    }
}
impl<'a> Hittable<'a> for RotateY<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        let mut ro = ray.ro;
        let mut rd = ray.rd;

        ro[0] = self.cos_theta*ray.ro[0] - self.sin_theta*ray.ro[2];
        ro[2] = self.sin_theta*ray.ro[0] + self.cos_theta*ray.ro[2];

        rd[0] = self.cos_theta*ray.rd[0] - self.sin_theta*ray.rd[2];
        rd[2] = self.sin_theta*ray.rd[0] + self.cos_theta*ray.rd[2];

        let rotated = Ray{ro,rd};

        if !self.object.hit(&rotated, ray_t, rec) {return false;}

        let mut p = rec.p;
        let mut n = rec.n;

        p[0] = self.cos_theta*rec.p[0] + self.sin_theta*rec.p[2];
        p[2] = -self.sin_theta*rec.p[0] + self.cos_theta*rec.p[2];

        n[0] = self.cos_theta*rec.n[0] + self.sin_theta*rec.n[2];
        n[2] = -self.sin_theta*rec.n[0] + self.cos_theta*rec.n[2];
    
        rec.p = p;
        rec.set_face_normal(&rotated, n);

        true
    }

    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        //TODO: AABB default
        *aabb = self.aabb.unwrap_or(AABB::default());
        self.aabb.is_some()
    }
}

#[derive(Clone)]
struct BVHnode<'a> {
    aabb: AABB,
    left: Rc<RefCell<HittableObject<'a>>>,
    right: Rc<RefCell<HittableObject<'a>>>,
}

impl<'a> Default for BVHnode<'a>{
    fn default() -> Self {
        BVHnode
        {
            aabb: AABB::default(),
            left: Rc::new(RefCell::new(HittableObject::default())),
            right: Rc::new(RefCell::new(HittableObject::default())),
        }
    }
}
impl<'a> Hittable<'a> for BVHnode<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        if !self.aabb.hit(ray, ray_t.min, ray_t.max) {return false;}
        let hit_left = self.left.borrow_mut().hit(ray, ray_t, rec);
        let hit_right = self.right.borrow_mut().hit(ray, Interval::new(ray_t.min, if hit_left {rec.t} else {ray_t.max}), rec);
        return hit_left || hit_right;
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        aabb.minimum = self.aabb.minimum;
        aabb.maximum = self.aabb.maximum;
        println!("BVHnode bounding box");
        true
    }  
}

impl<'a> BVHnode<'a> {
    fn box_val<T: Hittable<'a> + ?Sized>(a: &Rc<RefCell<T>>, axis: u8) -> NumberType
    {
        let mut a_box = AABB::default();
        if !a.borrow_mut().bounding_box(&mut a_box) {panic!("missing implemenation for AABB");}
        else {a_box.minimum[axis]}
    }

    fn construct(objects: Vec<Rc<RefCell<HittableObject<'a>>>>) -> Rc<RefCell<HittableObject>>
    {
        let mut node = BVHnode::default();
        let mut copy = objects.clone();
        let axis: u8 = thread_rng().gen_range(0..3);

        println!("Axis: {axis}");
       
        let x = move |a:&Rc<RefCell<HittableObject<'a>>>| OrderedFloat(Self::box_val(a,axis));

        let object_span = copy.len();
        println!("object_span = {object_span}");

        if object_span == 0 {
            panic!("no elements when running construct");
        }
        if object_span == 1 {
            return copy[copy.len()-1].clone();
            //node.left  = copy[copy.len()-1].clone();
            //node.right = copy[copy.len()-1].clone();

            //println!("SINGLE OBJECT");
        }
        else if object_span == 2 {
            node.left  = copy[copy.len()-1].clone();
            node.right = copy[copy.len()-2].clone();
            //if x(&self.left) > x(&self.right)
            //{
            //    mem::swap(&mut self.left, &mut self.right);
            //}
        }
        else {
            copy.sort_by_key(x);
            let mid = object_span/2;
            
            
            let left_node = Self::construct(copy[mid..].to_vec());
            node.left = left_node;

            let right_node = Self::construct(copy[..mid].to_vec());
            node.right = right_node;
        }

        let mut box_left = AABB::default(); 
        let mut box_right = AABB::default(); 
        
        let has_left = node.left.borrow().bounding_box(&mut box_left);
        let has_right = node.right.borrow().bounding_box(&mut box_right);
        if !has_left || !has_right
        {
            panic!("AABB missing");
        }
        node.aabb = box_left.surrounding_box(box_right);
        Rc::new(RefCell::new(HittableObject::BVHnode(node)))
    }
}

#[derive(Clone, Copy)]
struct ONB {
    u: Vec3,
    v: Vec3,
    w: Vec3,
}
impl ONB {
    fn local(&self,p: Vec3)->Vec3 {
        self.u*p.x+self.v*p.y+self.w*p.z
    }
    fn build_from_w(n: Vec3) -> Self {
        let w = n.normalized();
        let a = if w.x.abs() > 0.9 {Vec3::new(0.0,1.0,0.0)} else {Vec3::new(1.0,0.0,0.0)};
        let v = w.cross(a).normalized();
        let u = w.cross(v);
        ONB {u,v,w}
    }
}

trait PDF {
    fn value (&self, direction: Vec3) -> NumberType;
    fn generate(&self) -> Vec3;
}

#[derive(Clone, Copy)]
enum PDFEnum<'a> {
    CosinePDF(CosinePDF),
    HittablePDF(HittablePDF<'a>),
}
impl<'a> PDF for PDFEnum<'a> {
    fn value (&self, direction: Vec3) -> NumberType {
        match self {
            PDFEnum::CosinePDF(c) => c.value(direction),
            PDFEnum::HittablePDF(h) => h.value(direction),
        }
    }
    fn generate(&self) -> Vec3 {
        match self {
            PDFEnum::CosinePDF(c) => c.generate(),
            PDFEnum::HittablePDF(h) => h.generate(),
        }
    }
}

#[derive(Clone, Copy)]
struct CosinePDF {
    onb: ONB
}
impl PDF for CosinePDF {
    fn value (&self, direction: Vec3) -> NumberType {
        let cosine = direction.normalized().dot(self.onb.w);
        if cosine < 0.0 {0.0} else {cosine/PI}
    }
    fn generate(&self) -> Vec3 {
        self.onb.local(Vec3::random_cosine_direction())
    }
}
impl<'a> CosinePDF {
    fn new(w: Vec3) -> PDFEnum<'a> {
        PDFEnum::CosinePDF(CosinePDF { onb: ONB::build_from_w(w)})
    }
}

#[derive(Clone, Copy)]
struct HittablePDF<'a> {
    o: Vec3,
    object:&'a HittableObject<'a>,
}
impl<'a> PDF for HittablePDF<'a> {
    fn value (&self, direction: Vec3) -> NumberType {
        self.object.pdf_value(self.o, direction) 
    }
    fn generate(&self) -> Vec3 {
        self.object.random(self.o)
    }
}
impl<'a> HittablePDF<'a> {
    fn new(object:&'a HittableObject<'a>, o: Vec3) -> PDFEnum<'a> {
        PDFEnum::HittablePDF(HittablePDF {object, o,})
    }
}

struct MixPDF<'a,'b,'c,'d> {
    p1: &'a PDFEnum<'b>,
    p2: &'c PDFEnum<'d>,
    f: NumberType,
}
impl<'a,'b,'c,'d> PDF for MixPDF<'a,'b,'c,'d> {
    fn value (&self, direction: Vec3) -> NumberType {
        (1.0-self.f)*self.p2.value(direction)
             +self.f*self.p1.value(direction)
    }
    fn generate(&self) -> Vec3 {
        if random_val() < self.f {
            self.p1.generate()
        }
        else {
            self.p2.generate()
        }
    }
}
impl<'a,'b,'c,'d> MixPDF<'a,'b,'c,'d> {
    fn new(p1: &'a PDFEnum<'b>, p2: &'c PDFEnum<'d>, f: NumberType) -> Self {
        MixPDF { p1,p2,f } 
    }
}

#[derive(Clone,Copy)]
struct Interval {
    min: NumberType,
    max: NumberType,
}
impl Interval {
    fn new(min: NumberType, max: NumberType) -> Self {Interval {min, max}}
    fn contains(&self, x: NumberType) -> bool {self.min <= x && x <= self.max}
    
    const EMPTY: Self = Interval {min: NumberType::INFINITY, max:NumberType::NEG_INFINITY};
    const UNIVERSE: Self = Interval {min: NumberType::NEG_INFINITY, max:NumberType::INFINITY};
}
impl Default for Interval {
    fn default() -> Self {Self::EMPTY}
}

fn main() {
    let aspect_ratio = 1.0;//16.0/9.0;
    let imgx         = 1080;//600;
    let imgy         = ((imgx as NumberType)/aspect_ratio) as u32;

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
    
    
    let mut object_list: Vec<Rc<RefCell<HittableObject>>> = Vec::new();
    let big_light = false;

    //let red   = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.65,0.05,0.05)});
    //let white = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.73,0.73,0.73)});
    //let green = MaterialEnum::Lambertian(Lambertian{albedo: Vec3::new(0.12,0.45,0.15)});
    

    let c1 = SolidColor::new(Vec3::one(0.8));
    let c2 = SolidColor::new(Vec3::one(0.2));
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
  
    //object_list.push(Rc::new(RefCell::new( HittableObject::Sphere(Sphere{material: glass, center: Vec3::one(250.0), radius: 100.0}))));
    object_list.push(Rc::new(RefCell::new( HittableObject::Sphere(Sphere{material: white, center: Vec3::one(400.0), radius: 100.0}))));

    let lights =  if big_light {XZRect{material: light, x0: 113.0, x1: 443.0, z0: 127.0, z1: 432.0, k: 554.0, }}
                  else {XZRect{material: light, x0: 213.0, x1: 343.0, z0: 227.0, z1: 332.0, k: 554.0, }};

    object_list.push(Rc::new(RefCell::new(HittableObject::XZRect(lights.clone()))));
    let lights = HittableObject::XZRect(lights.clone());

    object_list.push(Rc::new(RefCell::new( HittableObject::YZRect(YZRect{material: green, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }))));
    object_list.push(Rc::new(RefCell::new( HittableObject::YZRect(YZRect{material: red, y0: 0.0, y1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }))));
    object_list.push(Rc::new(RefCell::new( HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 555.0, }))));
    object_list.push(Rc::new(RefCell::new( HittableObject::XZRect(XZRect{material: white, x0: 0.0, x1: 555.0, z0: 0.0, z1: 555.0, k: 0.0, }))));
    object_list.push(Rc::new(RefCell::new( HittableObject::XYRect(XYRect{material: white, x0: 0.0, x1: 555.0, y0: 0.0, y1: 555.0, k: 555.0, }))));

    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::new(165.0,330.0,165.0), white));
    let cube = HittableObject::RotateY(RotateY::new(&cube, 15.0));
    let cube = HittableObject::Translate(Translate{object: &cube, offset: Vec3::new(265.0,0.0,295.0)});
    object_list.push(Rc::new(RefCell::new(cube)));

    let cube = HittableObject::Cuboid(Cuboid::new(Vec3::one(0.0), Vec3::one(165.0), white));
    let cube = HittableObject::RotateY(RotateY::new(&cube, -18.0));
    let cube = HittableObject::Translate(Translate{object: &cube, offset: Vec3::new(130.0,0.0,65.0)});
    object_list.push(Rc::new(RefCell::new(cube)));


    //object_list.push(Rc::new(RefCell::new( HittableObject::Cuboid(
    //                Cuboid::new(Vec3::new(265.0,0.0,295.0), Vec3::new(430.0,330.0,460.0), white)))));

    //object_list.push(Rc::new(RefCell::new(
    //            XZRect{material: light, 
    //                x0: 123.0,
    //                x1: 423.0,
    //                z0: 147.0, 
    //                z1: 412.0,
    //                k:  554.0, 
    //            })));
    
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
    
    //let material = Lambertian{albedo: Vec3::new(0.5,0.7,0.2)};
    //object_list.push(Rc::new(RefCell::new(
    //            Sphere{
    //                material: MaterialEnum::Lambertian(material),
    //                radius: 100.0,
    //                center: Vec3::new(555.0/2.0+100.0,555.0/2.0-100.0,555.0/2.0),
    //            } )));
    
    
    let mut rng = thread_rng();

    //if false { 
    //    for i in 0..10 {
    //        let rad = 50.0;
    //        let center = Vec3::new(random_range(0.0,350.0),random_range(0.0,350.0),random_range(0.0,350.0));
    //        let choose_mat = random_val();

    //        if choose_mat < 0.5 {
    //            let albedo = Vec3::random();
    //            let material = MaterialEnum::Lambertian(Lambertian {albedo});
    //            object_list.push(Rc::new(RefCell::new(Sphere{material, radius: rad, center})));
    //        }
    //        //else if choose_mat < 0.5 {
    //        //    let light = Vec3::random()*4.0;
    //        //    let material = MaterialEnum::Emissive(Emissive{light});
    //        //    object_list.push(Rc::new(RefCell::new(Sphere{material: material.clone(), radius: rad, center})));
    //        //}
    //        else if choose_mat < 0.75{
    //            let albedo = Vec3::random();
    //            let blur = random_range(0.0,0.5);
    //            let material = MaterialEnum::Metal(Metal{albedo,blur});
    //            object_list.push(Rc::new(RefCell::new(Sphere{material: material.clone(), radius: rad, center})));
    //        }
    //        else {
    //            let material = MaterialEnum::Dielectric(Dielectric{ir:1.5});
    //            object_list.push(Rc::new(RefCell::new(Sphere{material: material.clone(), radius: rad, center})));
    //        }

    //    }
    //}

    let num_objects = object_list.len();
    let bvh = BVHnode::construct(object_list);
    let bvh = bvh.borrow_mut();
    
    //let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(13.0,2.0,3.0), Vec3::new(0.0,0.0,0.0), Vec3::new(0.0,1.0,0.0), 45.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(26.0,3.0,6.0), Vec3::new(0.0,2.0,0.0), Vec3::new(0.0,1.0,0.0), 20.0, aspect_ratio);
    let cam = Camera::new(Vec3::new(278.0,278.0,-800.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio);
    //let cam = Camera::new(Vec3::new(478.0,278.0,-600.0), Vec3::new(278.0,278.0,0.0), Vec3::new(0.0,1.0,0.0), 40.0, aspect_ratio);
    
    let start = Instant::now();

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        if x==0 {let f = y as NumberType/imgy as NumberType * 100.0;println!("row {y}/{imgy}: {f}%");}

        let samples   = 4;//16;//32;//256;
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
    }
    let duration = (start.elapsed().as_millis() as NumberType)/1000.0;
    println!("Rendered {num_objects} objects in {duration} seconds");


    imgbuf.save("output.png").unwrap();
}
