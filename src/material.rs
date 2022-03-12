
use crate::texture::*;
use crate::common::*;
use crate::hittable::*;
use crate::ray::*;
use crate::vector::*;
use crate::onb::*;

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
#[derive(Debug)]
pub struct Lambertian<'a> {
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
    pub fn new(texture: TextureEnum) -> MaterialEnum {
        MaterialEnum::Lambertian(Lambertian {texture,})
    } 
    pub fn col(color: Vec3) -> MaterialEnum<'a> {
        MaterialEnum::Lambertian(Lambertian {
            texture: SolidColor::new(color),
        }) 
    }
}

#[derive(Copy,Clone,Default)]
#[derive(Debug)]
pub struct Emissive{
    pub light: Vec3,
}
impl Material for Emissive{
    fn emission(&self) -> Vec3 {self.light}
}

pub trait Material {
    fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _albedo: &mut Vec3, _scattered: &mut Ray, _pdf: &mut NumberType) -> bool {false}

    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _sray: &Ray) -> NumberType {0.0}//TODO

    fn emission(&self) -> Vec3 {Vec3::one(0.0)}
}

#[derive(Copy,Clone)]
#[derive(Debug)]
pub enum MaterialEnum<'a> {
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
