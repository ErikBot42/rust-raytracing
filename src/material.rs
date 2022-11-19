
use crate::pdf::*;
use crate::texture::*;
use crate::common::*;
use crate::hittable::*;
use crate::ray::*;
use crate::vector::*;
use crate::random::*;

#[derive(Default)]
pub struct ScatterRecord {
    pub attenuation: Vec3,
    pub scatter: ScatterEnum,
}

pub enum ScatterEnum {
    RaySkip {ray: Ray},
    Pdf {pdf: PDFMaterialEnum}
}
impl Default for ScatterEnum {
    fn default() -> Self {
        ScatterEnum::RaySkip {ray: Ray::default()}
    }
}

pub trait Material {
    //TODO: return option
    //fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _albedo: &mut Vec3, _scattered: &mut Ray, _pdf: &mut NumberType) -> bool {false}
    fn scatter(&self,_ray: &Ray, _rec: &HitRecord, _srec: &mut ScatterRecord) -> bool {false}

    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _sray: &Ray) -> NumberType {0.0}//TODO

    fn emission(&self) -> Vec3 {Vec3::one(0.0)}
}

#[derive(Copy,Clone)]
#[derive(Debug)]
pub struct Isotropic<'a> {
    albedo: TextureEnum<'a>,
}
impl<'a> Material for Isotropic<'a> {
    fn scatter(&self, _ray: &Ray, rec: &HitRecord, srec: &mut ScatterRecord) -> bool {
        srec.attenuation = self.albedo.value(rec.u, rec.v, rec.p);
        srec.scatter = ScatterEnum::Pdf {pdf: SpherePDF::create()};
        true 
    }
    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _sray: &Ray) -> NumberType {
        1.0 / (4.0 * PI)
    }
}

#[derive(Copy,Clone,Default)]
#[derive(Debug)]
pub struct Lambertian<'a> {
    texture: TextureEnum<'a>,
}
impl<'a> Material for Lambertian<'a> {

    fn scatter(&self, _ray: &Ray, rec: &HitRecord, srec: &mut ScatterRecord) -> bool {
        srec.attenuation = self.texture.value(rec.u, rec.v, rec.p);
        srec.scatter = ScatterEnum::Pdf{pdf: CosinePDF::create(rec.n)};
        true
    }

    //fn scatter(&self,_ray: &Ray, rec: &HitRecord, albedo: &mut Vec3, sray: &mut Ray, pdf: &mut NumberType) -> bool
    //{

    //    //sray.rd.set(rec.n + Vec3::random_unit());//Vec3::random_in_unit_hemisphere(rec.n));
    //    //sray.rd.set(Vec3::random_in_unit_hemisphere(rec.n));
    //    //sray.ro.set(rec.p);
    //    //*attenuation = self.texture.value(rec.u,rec.v,rec.p);
    //    
    //    
    //    //sray.rd = (rec.n + Vec3::random_unit()).normalized();
    //    //sray.ro = rec.p;
    //    //*albedo = self.texture.value(rec.u, rec.v, rec.p);
    //    //*pdf = rec.n.dot(sray.rd)/PI;
    //    
    //    //sray.rd = Vec3::random_in_unit_hemisphere(rec.n);
    //    //sray.ro = rec.p;
    //    //*albedo = self.texture.value(rec.u, rec.v, rec.p);
    //    //*pdf = 0.5/PI;
    //    
    //    let onb = ONB::build_from_w(rec.n);

    //    *sray = Ray::new(
    //        rec.p,
    //        onb.local(Vec3::random_cosine_direction()).normalized()
    //        );
    //    *albedo = self.texture.value(rec.u, rec.v, rec.p);
    //    *pdf = onb.w.dot(sray.rd)/PI;
    //    true
    //}
    fn scattering_pdf(&self, _ray: &Ray, rec: &HitRecord, sray: &Ray) -> NumberType {
        let cosine = rec.n.dot(sray.rd.normalized());
        (cosine/PI).max(0.0)
    }
}
impl<'a> Lambertian<'a> {
    pub fn create(texture: TextureEnum) -> MaterialEnum {
        MaterialEnum::Lambertian(Lambertian {texture,})
    } 
    pub fn col(color: Vec3) -> MaterialEnum<'a> {
        MaterialEnum::Lambertian(Lambertian {
            texture: SolidColor::create(color),
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


#[derive(Copy,Clone)]
#[derive(Debug)]
pub enum MaterialEnum<'a> {
    Lambertian(Lambertian<'a>),
    Emissive(Emissive),
    Metal(Metal<'a>),
    Dielectric(Dielectric),
    Isotropic(Isotropic<'a>),
}

impl<'a> Default for MaterialEnum<'a> {
    fn default() -> Self {
        MaterialEnum::Lambertian(Lambertian::default())
    }
}


impl<'a> Material for MaterialEnum<'a>
{
    fn scatter(&self,ray: &Ray, rec: &HitRecord, srec: &mut ScatterRecord) -> bool 
    {
        match self {
            MaterialEnum::Lambertian(l) => l.scatter(ray, rec, srec),
            MaterialEnum::Emissive(e) => e.scatter(ray, rec, srec),
            MaterialEnum::Metal(m) => m.scatter(ray, rec, srec),
            MaterialEnum::Dielectric(d) => d.scatter(ray, rec, srec),
            MaterialEnum::Isotropic(i) => i.scatter(ray, rec, srec),
        }
    }
    fn emission(&self) -> Vec3 {
        match self {
            MaterialEnum::Lambertian(l) => l.emission(),
            MaterialEnum::Emissive(e) => e.emission(),
            MaterialEnum::Metal(m) => m.emission(),
            MaterialEnum::Dielectric(d) => d.emission(),
            MaterialEnum::Isotropic(i) => i.emission(),
        }
    }
    fn scattering_pdf(&self, _ray: &Ray, _rec: &HitRecord, _sray: &Ray) -> NumberType {
        match self {
            MaterialEnum::Lambertian(l) => l.scattering_pdf(_ray, _rec, _sray),
            MaterialEnum::Emissive(e) => e.scattering_pdf(_ray, _rec, _sray),
            MaterialEnum::Metal(m) => m.scattering_pdf(_ray, _rec, _sray),
            MaterialEnum::Dielectric(d) => d.scattering_pdf(_ray, _rec, _sray),
            MaterialEnum::Isotropic(i) => i.scattering_pdf(_ray, _rec, _sray),
        }
    }
}


#[derive(Copy,Clone,Default)]
#[derive(Debug)]
pub struct Metal<'a> {
    albedo: TextureEnum<'a>,
    blur: NumberType,
}
impl<'a> Material for Metal<'a>{
    fn scatter(&self, ray: &Ray, rec: &HitRecord, srec: &mut ScatterRecord) -> bool {
        srec.attenuation = self.albedo.value(rec.u, rec.v, rec.p);

        //TODO: remove normalized()

        let reflected = ray.rd.normalized().reflect(rec.n); 
        let reflected = reflected + Vec3::random_unit()*self.blur;
        let reflected = Ray::new(rec.p, reflected);

        srec.scatter = ScatterEnum::RaySkip {ray:reflected};


        true

        //sray.rd.set(ray.rd.normalized().reflect(rec.n)+Vec3::random_unit()*self.blur);
        //sray.ro.set(rec.p);
        //attenuation.set(self.albedo);
        //sray.rd.dot(rec.n)>0.0
    }
}
impl<'a> Metal<'a> {
    pub fn create(albedo: TextureEnum<'a>, blur: NumberType) -> MaterialEnum<'a> {
        MaterialEnum::Metal(Metal {albedo, blur})
    }

    pub fn col(color: Vec3, blur: NumberType) -> MaterialEnum<'a> {
        MaterialEnum::Metal(Metal{
            albedo: SolidColor::create(color),
            blur,
        }) 
    }

}


#[derive(Copy,Clone,Default)]
#[derive(Debug)]
pub struct Dielectric {
    ir: NumberType,
}
impl<'a> Dielectric {
    pub fn create(ir: NumberType) -> MaterialEnum<'a> {
        MaterialEnum::Dielectric(Dielectric {ir})
    }

    fn reflectance(cosine: NumberType, ref_idx: NumberType ) -> NumberType {
        let mut r0 = (1.0-ref_idx)/(1.0+ref_idx);
        r0 = r0*r0;
        return r0 + (1.0-r0)*(1.0-cosine).powi(5);
    }
}
impl Material for Dielectric{
    fn scatter(&self, ray: &Ray, rec: &HitRecord, srec: &mut ScatterRecord) -> bool {
        srec.attenuation = Vec3::one(1.0);

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
        //sray.rd.set(direction);
        //sray.ro.set(rec.p);
        //
        srec.scatter = ScatterEnum::RaySkip {ray: Ray::new(rec.p, direction)};


        true
    }
}

