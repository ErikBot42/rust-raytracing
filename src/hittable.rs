
use crate::vector::Vec3;
use crate::common::*;
//use crate::onb::ONB;
//use crate::common::{NumberType,PI};
use crate::random::{random_range,random_val};
//use crate::random::{rng_seed,rng,random_range,random_val};
use crate::material::*;
use crate::ray::*;
use crate::interval::*;
use crate::aabb::*;
use crate::bvh::*;

#[derive(Copy, Clone, Default)]
#[derive(Debug)]
pub struct HitRecord<'a> {
    pub p: Vec3,
    pub n: Vec3,
    pub u: NumberType,
    pub v: NumberType,
    pub material: MaterialEnum<'a>,
    pub t: NumberType,
    pub front_face: bool,
}

impl<'a> HitRecord<'a> {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3)
    {
        self.front_face = ray.rd.dot(outward_normal) < 0.0;
        self.n = if self.front_face {outward_normal} else {-outward_normal}
    }
}

pub trait Hittable<'a> {
    // TODO: return option instead
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool;
    fn bounding_box(&self, _aabb: &mut AABB) -> bool {false}  

    fn pdf_value (&self, _o: Vec3,_v: Vec3) -> NumberType {panic!("pdf_value for Hittable is not implemented")}
    fn random (&self, _o: Vec3) -> Vec3 {panic!("random for Hittable is not implemented")}
}


#[derive(Clone)]
#[derive(Debug)]
pub enum HittableObject<'a> {
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

#[derive(Clone)]
#[derive(Debug)]
pub enum HittableObjectSimple<'a> {
    Sphere(Sphere<'a>),
    XYRect(XYRect<'a>),
    XZRect(XZRect<'a>),
    YZRect(YZRect<'a>),
    Cuboid(Cuboid<'a>),
    Translate(Translate<'a>),
    RotateY(RotateY<'a>),
}
impl<'a> Hittable<'a> for HittableObjectSimple<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool {
        match self {
            HittableObjectSimple::Sphere(s) => s.hit(ray, ray_t, rec),
            //HittableObjectSimple::BVHnode(b) => b.hit(ray, ray_t, rec),
            HittableObjectSimple::XYRect(xy) => xy.hit(ray, ray_t, rec),
            HittableObjectSimple::XZRect(xz) => xz.hit(ray, ray_t, rec),
            HittableObjectSimple::YZRect(yz) => yz.hit(ray, ray_t, rec),
            //HittableObjectSimple::ConstantMedium(c) => c.hit(ray, ray_t, rec),
            HittableObjectSimple::Cuboid(u) => u.hit(ray, ray_t, rec),
            HittableObjectSimple::Translate(t) => t.hit(ray, ray_t, rec),
            HittableObjectSimple::RotateY(ry) => ry.hit(ray, ray_t, rec),
        }
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        match self {
            HittableObjectSimple::Sphere(s) => s.bounding_box(aabb),
            //HittableObjectSimple::BVHnode(b) => b.bounding_box(aabb),
            HittableObjectSimple::XYRect(xy) => xy.bounding_box(aabb),
            HittableObjectSimple::XZRect(xz) => xz.bounding_box(aabb),
            HittableObjectSimple::YZRect(yz) => yz.bounding_box(aabb),
            //HittableObjectSimple::ConstantMedium(c) => c.bounding_box(aabb),
            HittableObjectSimple::Cuboid(u) => u.bounding_box(aabb),
            HittableObjectSimple::Translate(t) => t.bounding_box(aabb),
            HittableObjectSimple::RotateY(ry) => ry.bounding_box(aabb),
        }
    }
    fn pdf_value (&self, o: Vec3,v: Vec3) -> NumberType {
        match self {
            HittableObjectSimple::Sphere(s) => s.pdf_value(o,v),
            //HittableObjectSimple::BVHnode(b) => b.pdf_value(o,v),
            HittableObjectSimple::XYRect(xy) => xy.pdf_value(o,v),
            HittableObjectSimple::XZRect(xz) => xz.pdf_value(o,v),
            HittableObjectSimple::YZRect(yz) => yz.pdf_value(o,v),
            //HittableObjectSimple::ConstantMedium(c) => c.pdf_value(o,v),
            HittableObjectSimple::Cuboid(u) => u.pdf_value(o,v),
            HittableObjectSimple::Translate(t) => t.pdf_value(o,v),
            HittableObjectSimple::RotateY(ry) => ry.pdf_value(o,v),
        }
    }
    fn random (&self, o: Vec3) -> Vec3 {
        match self {
            HittableObjectSimple::Sphere(s) => s.random(o),
            //HittableObjectSimple::BVHnode(b) => b.random(o),
            HittableObjectSimple::XYRect(xy) => xy.random(o),
            HittableObjectSimple::XZRect(xz) => xz.random(o),
            HittableObjectSimple::YZRect(yz) => yz.random(o),
            //HittableObjectSimple::ConstantMedium(c) => c.random(o),
            HittableObjectSimple::Cuboid(u) => u.random(o),
            HittableObjectSimple::Translate(t) => t.random(o),
            HittableObjectSimple::RotateY(ry) => ry.random(o),
        }

    }


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
#[derive(Debug)]
pub struct XYRect<'a> {
    pub x0: NumberType,
    pub x1: NumberType,
    pub y0: NumberType,
    pub y1: NumberType,
    pub k: NumberType,
    pub material: MaterialEnum<'a>,
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

        //if x<self.x0 || x>self.x1 || y<self.y0 || y>self.y1 {return false;}
        if !Interval::new(self.x0,self.x1).contains(x) || !Interval::new(self.y0,self.y1).contains(y) {return false;}
        
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
#[derive(Debug)]
pub struct XZRect<'a> {
    pub x0: NumberType,
    pub x1: NumberType,
    pub z0: NumberType,
    pub z1: NumberType,
    pub k: NumberType,
    pub material: MaterialEnum<'a>,
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
        if !self.hit(&Ray{ro: o, rd: v}, Interval::EPSILON_UNIVERSE, &mut rec) {return 0.0}

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
#[derive(Debug)]
pub struct YZRect<'a> {
    pub y0: NumberType,
    pub y1: NumberType,
    pub z0: NumberType,
    pub z1: NumberType,
    pub k: NumberType,
    pub material: MaterialEnum<'a>,
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
#[derive(Debug)]
pub struct Sphere<'a> {
    pub center: Vec3,
    pub radius: NumberType,
    pub material: MaterialEnum<'a>,//Rc<dyn Material>,
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
#[derive(Debug)]
pub struct HittableList<'a> {
    pub l: Vec<HittableObject<'a>>,
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
#[derive(Debug)]
pub struct Cuboid<'a> {
    min: Vec3,
    max: Vec3,
    sides: HittableList<'a>,
}
impl<'a> Cuboid<'a> {
    pub fn new(min: Vec3, max: Vec3, material: MaterialEnum<'a>) -> Self {
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
#[derive(Debug)]
pub struct ConstantMedium<'a> {
    pub neg_inv_denisty: NumberType,
    pub boundary: &'a HittableObject<'a>,
    pub material: MaterialEnum<'a>,
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



// Translate incoming ray for object 

#[derive(Clone, Copy)]
#[derive(Debug)]
pub struct Translate<'a> {
    pub offset: Vec3,
    pub object: &'a HittableObject<'a>,
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
#[derive(Debug)]
pub struct RotateY<'a> {
    pub sin_theta: NumberType,
    pub cos_theta: NumberType,
    pub aabb: Option<AABB>,
    pub object: &'a HittableObject<'a>,
}
impl<'a> RotateY<'a> {
    pub fn new(object: &'a HittableObject<'a>, angle: NumberType) -> Self {
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
