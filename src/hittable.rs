
use crate::vector::Vec3;
use crate::common::*;
//use crate::onb::ONB;
//use crate::common::{NumberType,PI};
use crate::random::*;
//use crate::random::{rng_seed,rng,random_range,random_val};
use crate::material::*;
use crate::ray::*;
use crate::interval::*;
use crate::aabb::*;
use crate::onb::*;

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
// TODO: material should be reference


//impl<'a> Default for HitRecord<'a> {
//    fn default() -> Self {
//        HitRecord {
//            p: Vec3::default(),
//            n: Vec3::default(),
//            u: NumberType::default(),
//            v: NumberType::default(),
//            material: MaterialEnum::default(),
//            t: NumberType::default(),
//            front_face: bool::default(),
//        }
//    }
//}
impl<'a> HitRecord<'a> {
    fn new<U,V>(material: &MaterialEnum<'a>, ray: &Ray, t: NumberType, n: Vec3, u: U, v: V) -> Self 
    where
    U: Into<Option<NumberType>>,
    V: Into<Option<NumberType>>
    {
        let mut rec = HitRecord {
            p: ray.at(t),
            n,
            u: u.into().unwrap_or(0.0), 
            v: v.into().unwrap_or(0.0),
            material: *material,
            t,
            front_face: bool::default(),
        };
        rec.update_normal(ray);
        rec
    }
    
    fn update_normal(&mut self, ray: &Ray) {
        self.front_face = ray.rd.dot(self.n) < 0.0;
        self.n = if self.front_face {self.n} else {-self.n}
    }
}

pub trait Hittable<'a> {
    // TODO: return option instead
    //fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord<'a>) -> bool;
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>;
    fn bounding_box(&self, _aabb: &mut AABB) -> bool {false}  

    fn pdf_value (&self, _o: Vec3,_v: Vec3) -> NumberType {unimplemented!()}
    fn random (&self, _o: Vec3) -> Vec3 {unimplemented!()}
}

#[derive(Clone)]
#[derive(Debug)]
pub enum HittableObject<'a> {
    Sphere(Sphere<'a>),
    XYRect(XYRect<'a>),
    XZRect(XZRect<'a>),
    YZRect(YZRect<'a>),
    Cuboid(Cuboid<'a>),
    Translate(Translate<'a>),
    RotateY(RotateY<'a>),
    Quad(Quad<'a>),
}


impl<'a> Hittable<'a> for HittableObject<'a>
{
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        match self {
            HittableObject::Sphere(s) => s.hit(ray, ray_t),
            //HittableObject::BVHnode(b) => b.hit(ray, ray_t),
            HittableObject::XYRect(xy) => xy.hit(ray, ray_t),
            HittableObject::XZRect(xz) => xz.hit(ray, ray_t),
            HittableObject::YZRect(yz) => yz.hit(ray, ray_t),
            //HittableObject::ConstantMedium(c) => c.hit(ray, ray_t),
            HittableObject::Cuboid(u) => u.hit(ray, ray_t),
            HittableObject::Translate(t) => t.hit(ray, ray_t),
            HittableObject::RotateY(ry) => ry.hit(ray, ray_t),
            HittableObject::Quad(q) => q.hit(ray, ray_t),
        }
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        match self {
            HittableObject::Sphere(s) => s.bounding_box(aabb),
            //HittableObject::BVHnode(b) => b.bounding_box(aabb),
            HittableObject::XYRect(xy) => xy.bounding_box(aabb),
            HittableObject::XZRect(xz) => xz.bounding_box(aabb),
            HittableObject::YZRect(yz) => yz.bounding_box(aabb),
            //HittableObject::ConstantMedium(c) => c.bounding_box(aabb),
            HittableObject::Cuboid(u) => u.bounding_box(aabb),
            HittableObject::Translate(t) => t.bounding_box(aabb),
            HittableObject::RotateY(ry) => ry.bounding_box(aabb),
            HittableObject::Quad(q) => q.bounding_box(aabb),
        }
    }
    fn pdf_value (&self, o: Vec3,v: Vec3) -> NumberType {
        match self {
            HittableObject::Sphere(s) => s.pdf_value(o,v),
            //HittableObject::BVHnode(b) => b.pdf_value(o,v),
            HittableObject::XYRect(xy) => xy.pdf_value(o,v),
            HittableObject::XZRect(xz) => xz.pdf_value(o,v),
            HittableObject::YZRect(yz) => yz.pdf_value(o,v),
            //HittableObject::ConstantMedium(c) => c.pdf_value(o,v),
            HittableObject::Cuboid(u) => u.pdf_value(o,v),
            HittableObject::Translate(t) => t.pdf_value(o,v),
            HittableObject::RotateY(ry) => ry.pdf_value(o,v),
            HittableObject::Quad(q) => q.pdf_value(o,v),

        }
    }
    fn random (&self, o: Vec3) -> Vec3 {
        match self {
            HittableObject::Sphere(s) => s.random(o),
            //HittableObject::BVHnode(b) => b.random(o),
            HittableObject::XYRect(xy) => xy.random(o),
            HittableObject::XZRect(xz) => xz.random(o),
            HittableObject::YZRect(yz) => yz.random(o),
            //HittableObject::ConstantMedium(c) => c.random(o),
            HittableObject::Cuboid(u) => u.random(o),
            HittableObject::Translate(t) => t.random(o),
            HittableObject::RotateY(ry) => ry.random(o),
            HittableObject::Quad(q) => q.random(o),
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
pub struct Quad<'a> {
    q: Vec3,
    u: Vec3,
    v: Vec3,
    material: MaterialEnum<'a>,
    n: Vec3, 
    d: NumberType,
    w: Vec3,
}

impl<'a> Quad<'a> {
    pub fn new(q: Vec3, u: Vec3, v: Vec3, material: MaterialEnum<'a>) -> Self {
        let n = u.cross(v);
        let normal = n.normalized();
        let d = normal.dot(q);
        let w = n / n.dot(n);

        let n = normal;
        Quad {q, u, v, material, n, d, w, }
    }
}

impl<'a> Hittable<'a> for Quad<'a> {


    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        *aabb = AABB::new(self.q, self.q + self.v + self.u);
        aabb.pad();
        true
    }

    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {

        let inv_denom = self.n.dot(ray.rd_inv);

        // I am ignoring case ray is paralell to surface
        
        let t = (self.d - self.n.dot(ray.ro))*inv_denom;
        if !ray_t.contains(t) {return None;}

        let p = ray.at(t);
        let planar = p-self.q;

        let a = self.w.dot(planar.cross(self.v));
        let b = self.w.dot(self.u.cross(planar));

        if (a < 0.0) || (1.0 < a) || (b < 0.0) || (1.0 < b) {return None;}
        
        Some(HitRecord::new(&self.material, ray, t, self.n, a, b))
    }



    //fn pdf_value (&self, o: Vec3,v: Vec3) -> NumberType {
    //    match self.hit(&Ray::new(o, v), Interval::EPSILON_UNIVERSE) {
    //        None => 0.0,
    //        Some(rec) => {
    //            let area = (self.x1-self.x0)*(self.z1-self.z0);
    //            let dist2 = rec.t*rec.t*v.dot2();
    //            let cosine = v.dot(rec.n).abs()/v.length();
    //            dist2/(cosine*area)
    //        }
    //    }
    //}
    //fn random (&self, o: Vec3) -> Vec3 {
    //    let random_point = Vec3::new(random_range(self.x0,self.x1), self.k, random_range(self.z0, self.z1)); 
    //    random_point - o
    //}


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
        aabb.min = Vec3::new(
            self.x0,
            self.y0,
            self.k-0.0001);
        aabb.max= Vec3::new(
            self.x0,
            self.y0,
            self.k-0.0001);
        true
    }
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let t = (self.k - ray.ro.z)*ray.rd_inv.z;
        if t<ray_t.min|| t>ray_t.max {return None;}

        let x = ray.ro.x + t*ray.rd.x;
        let y = ray.ro.y + t*ray.rd.y;

        //if x<self.x0 || x>self.x1 || y<self.y0 || y>self.y1 {return false;}
        if !Interval::new(self.x0,self.x1).contains(x) || !Interval::new(self.y0,self.y1).contains(y) {return None;}
        

        let u = (x-self.x0)/(self.x1-self.x0);
        let v = (y-self.y0)/(self.y1-self.y0);

        let outward_normal = Vec3::new(0.0,0.0,1.0);
        Some(HitRecord::new(&self.material, ray, t, outward_normal, u, v))

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
        aabb.min = Vec3::new(
            self.x0,
            self.z0,
            self.k-0.0001);
        aabb.max= Vec3::new(
            self.x0,
            self.z0,
            self.k-0.0001);
        true
    }
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>{
        let t = (self.k - ray.ro.y)*ray.rd_inv.y;
        if t<ray_t.min || t>ray_t.max {return None;}

        let x = ray.ro.x + t*ray.rd.x;
        let z = ray.ro.z + t*ray.rd.z;

        if x<self.x0 || x>self.x1 || z<self.z0 || z>self.z1 {return None;}
        
        let mut rec = HitRecord::default();

        let u = (x-self.x0)/(self.x1-self.x0);
        let v = (z-self.z0)/(self.z1-self.z0);

        rec.t = t;
        let outward_normal = Vec3::new(0.0,1.0,0.0);
        Some(HitRecord::new(&self.material, ray, t, outward_normal, u, v))
        

    }
    fn pdf_value (&self, o: Vec3,v: Vec3) -> NumberType {
        match self.hit(&Ray::new(o, v), Interval::EPSILON_UNIVERSE) {
            None => 0.0,
            Some(rec) => {
                let area = (self.x1-self.x0)*(self.z1-self.z0);
                let dist2 = rec.t*rec.t*v.dot2();
                let cosine = v.dot(rec.n).abs()/v.length();
                dist2/(cosine*area)
            }
        }
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
        aabb.min = Vec3::new(
            self.y0,
            self.z0,
            self.k-0.0001);
        aabb.max= Vec3::new(
            self.y0,
            self.z0,
            self.k-0.0001);
        true
    }
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>{
        let t = (self.k - ray.ro.x)*ray.rd_inv.x;
        if t<ray_t.min|| t>ray_t.max {return None;}

        let y = ray.ro.y + t*ray.rd.y;
        let z = ray.ro.z + t*ray.rd.z;

        if y<self.y0 || y>self.y1 || z<self.z0 || z>self.z1 {return None;}
        let u = (y-self.y0)/(self.y1-self.y0);
        let v = (z-self.z0)/(self.z1-self.z0);

        let outward_normal = Vec3::new(1.0,0.0,0.0);
        Some(HitRecord::new(&self.material, ray, t, outward_normal, u, v))
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
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>
    {
        let oc = ray.ro - self.center;
        let a = ray.rd.dot2(); // = 1 if rd is normalized
        let half_b = oc.dot(ray.rd);
        let c = oc.dot2()-self.radius.powi(2);
        let discriminant = half_b*half_b - a*c;
        if discriminant < 0.0 {return None;}
        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b-sqrtd)/a;
        if root<ray_t.min  || ray_t.max<root {
            root = (-half_b + sqrtd)/a;
            if root<ray_t.min || ray_t.max<root {return None;}
        }
        let t = root;
        let p = ray.at(root);
        let outward_normal = (p - self.center)/self.radius;
        let (u,v) = self.get_uv(outward_normal);
        Some(HitRecord::new(&self.material, ray, t, outward_normal, u, v))
    }
    fn bounding_box(&self,  aabb: &mut AABB) -> bool
    {
        println!("sphere bounding box");
        aabb.max = self.center+Vec3::one(self.radius);
        aabb.max = self.center+Vec3::one(self.radius);
        true
    }

    fn pdf_value (&self, o: Vec3, v: Vec3) -> NumberType {
        if let Some(_rec) = self.hit(&Ray::new(o,v), Interval::EPSILON_UNIVERSE) {
            let cos_theta_max = ((1.0 - self.radius*self.radius)/(self.center - o).dot2()).sqrt();
            let solid_angle = 2.0*PI*(1.0-cos_theta_max);
            1.0/solid_angle+_rec.u
        }
        else {
            0.0
        }
    }
    fn random (&self, o: Vec3) -> Vec3 {
        let rd = self.center - o;
        let dist2 = rd.dot2();
        let uvw = ONB::build_from_w(rd);

        
        uvw.local(Self::random_to_sphere(self.radius, dist2))
    }



}
impl<'a> Sphere<'a> {
    fn random_to_sphere(radius: NumberType, distance_squared: NumberType) -> Vec3 {
        let r1 = random_val();
        let r2 = random_val();

        let z = 1.0 + r2*( (1.0-radius*radius/distance_squared).sqrt() - 1.0);
        
        let phi = 2.0*PI*r1;
        let x = phi.cos()*(1.0-z*z).sqrt();
        let y = phi.sin()*(1.0-z*z).sqrt();

        return Vec3::new(x,y,z);
    }


    pub fn new(material: MaterialEnum<'a>, center: Vec3, radius: NumberType) -> Self {
        Sphere {
            center,
            radius,
            material
        }
    }

    fn get_uv(&self, p: Vec3) -> (NumberType, NumberType)
    {
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + PI;

        let u = phi / (2.0*PI);
        let v = theta / PI;
        (u,v)
    }
}

#[derive(Default, Clone)]
#[derive(Debug)]
pub struct HittableList<'a> {
    pub l: Vec<HittableObject<'a>>,
}

impl<'a> Hittable<'a> for HittableList<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut closest = ray_t.max;
        let mut rec: Option<HitRecord> = None;
        for object in &self.l {
            if let Some(x) = object.hit(ray, Interval::new(ray_t.min, closest)) {
                rec = Some(x);
                closest = x.t;
            }
        }
        rec
    }
}

#[derive(Default, Clone)]
#[derive(Debug)]
pub struct Cuboid<'a> {
    min: Vec3,
    max: Vec3,
//    center: Vec3,
//    size: Vec3,
    sides: HittableList<'a>,
//    material: MaterialEnum<'a>,
}
impl<'a> Cuboid<'a> {
    pub fn new(min: Vec3, max: Vec3, material: MaterialEnum<'a>) -> Self {
        /*let mut sides = HittableList::default();
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
        }*/
        let mut sides = HittableList::default();
        sides.l.push(HittableObject::XYRect(XYRect {x0: min.x, x1: max.x, y0: min.y, y1: max.y, k:max.z,material}));
        sides.l.push(HittableObject::XYRect(XYRect {x0: min.x, x1: max.x, y0: min.y, y1: max.y, k:min.z,material}));
        
        sides.l.push(HittableObject::XZRect(XZRect {x0: min.x, x1: max.x, z0: min.z, z1: max.z, k:max.y,material}));
        sides.l.push(HittableObject::XZRect(XZRect {x0: min.x, x1: max.x, z0: min.z, z1: max.z, k:min.y,material}));

        sides.l.push(HittableObject::YZRect(YZRect {y0: min.y, y1: max.y, z0: min.z, z1: max.x, k:max.x,material}));
        sides.l.push(HittableObject::YZRect(YZRect {y0: min.y, y1: max.y, z0: min.z, z1: max.x, k:min.x,material}));
        
        //let center = (min + max)/2.0;

        //let size = max - min;

        Cuboid {
            min,
            max,
//            center,
//            size,
            sides,
//            material,
        }
    }
}

impl<'a> Hittable<'a> for Cuboid<'a> {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>
    {

        //let radius = self.size.x; 
        //let center= self.center; 
        //
        //let oc = ray.ro - center;
        //let a = ray.rd.dot2();
        //let half_b = oc.dot(ray.rd);
        //let c = oc.dot2()-radius.powi(2);
        //let discriminant = half_b*half_b - a*c;
        //if discriminant < 0.0 {return false;}
        //let sqrtd = discriminant.sqrt();
        //let mut root = (-half_b-sqrtd)/a;
        //
        //let m = Vec3::new(1.0/ray.rd.x,1.0/ray.rd.y,1.0/ray.rd.z);
        //let n = m*ray.ro;
        //let k = Vec3::new(m.x.abs(), m.y.abs(), m.z.abs());

        //let t1 = -n -k;
        //let t2 = -n +k;
        //
        //let t_n = t1.x.max(t1.y).max(t1.z);
        //let t_f = t1.x.min(t1.y).min(t1.z);

        ////if t_n > 


        //


        //if root<ray_t.min  || ray_t.max<root {
        //    root = (-half_b + sqrtd)/a;
        //    if root<ray_t.min || ray_t.max<root {return false;}
        //}
        //rec.t = root;
        //rec.p = ray.at(root);
        //let outward_normal = (rec.p - center)/radius;
        //rec.set_face_normal(ray, outward_normal);
        ////get_uv(outward_normal, &mut rec.u, &mut rec.v);
        //rec.material = self.material;
        //true


        /*        let mut tmin = 0.0;
                  let mut tmax = 0.0;


                  let tx1 = (-1.0 - ray.ro.x)/ray.rd.x;
                  let tx2 = (1.0 - ray.ro.x)/ray.rd.x;

                  let mut nmin = Vec3::default();
                  let mut nmax = Vec3::default();

                  if tx1 < tx2 {
                  tmin = tx1;
                  tmax = tx2;
                  nmin = Vec3::new(-1.0, 0.0, 0.0);
                  nmax = Vec3::new(1.0, 0.0, 0.0);
                  } else {
                  tmin = tx2;
                  tmax = tx1;
                  nmin = Vec3::new(1.0, 0.0, 0.0);
                  nmax = Vec3::new(-1.0, 0.0, 0.0);
                  }

                  if tmin > tmax {return false;}

                  let ty1 = (-1.0 - ray.ro.y)/ray.rd.y;
                  let ty2 = (1.0 - ray.ro.y)/ray.rd.y;

                  if ty1 < ty2 {
                  if ty1 > tmin {
                  tmin = ty1;
                  nmin = Vec3::new(0.0, -1.0, 0.0);
                  }
                  if ty2 < tmax {
                  tmax = ty2;
                  nmax = Vec3::new(0.0, 1.0, 0.0);
                  }
                  } else {
                  if ty2 > tmin {
                  tmin = ty2;
                  nmin = Vec3::new(0.0, 1.0, 0.0);
                  }
                  if ty1 < tmax {
                  tmax = ty1;
                  nmax = Vec3::new(0.0, -1.0, 0.0);
                  }
                  }

                  if tmin > tmax {return false;}

                  let tz1 = (-1.0 - ray.ro.z)/ray.rd.z;
                  let tz2 = (1.0 - ray.ro.z)/ray.rd.z;

                  if tz1 < tz2 {
                  if tz1 > tmin {
                  tmin = tz1;
                  nmin = Vec3::new(0.0, 0.0, -1.0);
                  }
                  if tz2 < tmax {
                  tmax = tz2;
                  nmax = Vec3::new(0.0, 0.0, 1.0);
                  }
                  } else {
                  if tz2 > tmin {
                  tmin = tz2;
                  nmin = Vec3::new(0.0, 0.0, 1.0);
                  }
                  if tz1 < tmax {
                  tmax = tz1;
                  nmax = Vec3::new(0.0, 0.0, -1.0);
                  }
                  }

        if tmin > tmax {return false;}

        if tmin < 0.0 {
            tmin = tmax;
            nmin = nmax;
        }

        if tmin < 0.0 {
            return false;
        }

        rec.t = tmin;
        rec.p = ray.at(rec.t);
        rec.n = nmin;

        rec.material = self.material;
        true*/

            //rec.t = root;
            //rec.p = ray.at(root);
            //let outward_normal = (rec.p - center)/radius;
            //rec.set_face_normal(ray, outward_normal);
            ////get_uv(outward_normal, &mut rec.u, &mut rec.v);
            //rec.material = self.material;

            //let n = invdir*ray.ro;

            //let k = Vec3::new(m.x.abs(), m.y.abs(), m.z.abs());

            //let t1 = -n +k;
            //let t2 = -n -k;



            //if false {
            //    
            //    
            //    fn step(a: NumberType, b: NumberType) -> NumberType {
            //        if a>b {0.0} else {1.0}
            //    }
            //    fn step3(a: Vec3, b: Vec3) -> Vec3 {
            //        Vec3::new(step(a.x, b.x),
            //        step(a.y, b.y),
            //        step(a.z, b.z))
            //    }
            //    fn sign3(a: Vec3) -> Vec3 {
            //        Vec3::new(
            //            a.x.signum(),
            //            a.y.signum(),
            //            a.z.signum()
            //            )
            //    }

            //    let m = Vec3::new(1.0/ray.rd.x,1.0/ray.rd.y,1.0/ray.rd.z);
            //    let n = m*ray.ro;
            //    let k = Vec3::new(m.x.abs(), m.y.abs(), m.z.abs())*self.size*2.0;


            //    let t1 = -n -k;
            //    let t2 = -n +k;

            //    let t_n = t1.x.max(t1.y).max(t1.z);
            //    let t_f = t2.x.min(t2.y).min(t2.z);

            //    if t_n>t_f || t_f<0.0 {
            //        return false;
            //    }

            //    if !self.sides.hit(ray, ray_t, rec) {
            //        return false;
            //    }
            //    

            //    rec.n = Vec3::one(1.0);/*-sign3(ray.rd)*
            //        step3(Vec3::new(t1.y,t1.z,t1.x),Vec3::new(t1.x,t1.y,t1.z));
            //        step3(Vec3::new(t1.z,t1.x,t1.y),Vec3::new(t1.x,t1.y,t1.z));*/

            //    rec.set_face_normal(ray, rec.n);
            //    rec.t = 200.0;//ray_t.max.min(ray_t.min);
            //    rec.p = ray.at(rec.t);
            //    rec.material = self.material;



            //    true
            //} 
            //else {
            self.sides.hit(ray, ray_t)
            //}
    }
    fn bounding_box(&self,  aabb: &mut AABB) -> bool {
        *aabb = AABB::new(self.min, self.max);
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
    fn hit(&self, _ray: &Ray, _ray_t: Interval) -> Option<HitRecord>
    {
        /*
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
        */
        unimplemented!()
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

    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let moved = Ray::newi(ray.ro - self.offset, ray.rd, ray.rd_inv);
        
        match self.object.hit(&moved, ray_t) {
            None => None,
            Some(mut rec) => {
                rec.p+=self.offset;
                rec.update_normal(&moved);
                Some(rec)

            }
        }
    }
    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        if !self.object.bounding_box(aabb) {return false;}
        aabb.min += self.offset;
        aabb.max += self.offset;
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
                        let x = i*aabb.min.x + (1.0-i)*aabb.min.x;
                        let y = j*aabb.min.x + (1.0-j)*aabb.min.z;
                        let z = k*aabb.min.x + (1.0-k)*aabb.min.z;

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
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut ro = ray.ro;
        let mut rd = ray.rd;

        ro[0] = self.cos_theta*ray.ro[0] - self.sin_theta*ray.ro[2];
        ro[2] = self.sin_theta*ray.ro[0] + self.cos_theta*ray.ro[2];

        rd[0] = self.cos_theta*ray.rd[0] - self.sin_theta*ray.rd[2];
        rd[2] = self.sin_theta*ray.rd[0] + self.cos_theta*ray.rd[2];

        let rotated = Ray::new(ro,rd);

        match self.object.hit(&rotated, ray_t) {
            None => None,
            Some(mut rec) => {
                let mut p = rec.p;
                let mut n = rec.n;

                p[0] = self.cos_theta*rec.p[0] + self.sin_theta*rec.p[2];
                p[2] = -self.sin_theta*rec.p[0] + self.cos_theta*rec.p[2];
                n[0] = self.cos_theta*rec.n[0] + self.sin_theta*rec.n[2];
                n[2] = -self.sin_theta*rec.n[0] + self.cos_theta*rec.n[2];

                rec.p = p;
                rec.n = n;

                rec.update_normal(&rotated);
                Some(rec)
            }

        }

    }

    fn bounding_box(&self, aabb: &mut AABB) -> bool {
        //TODO: AABB default
        *aabb = self.aabb.unwrap_or_default();
        self.aabb.is_some()
    }
}
