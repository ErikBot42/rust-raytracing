
use crate::hittable::*;
use crate::random::*;
use crate::vector::*;
use crate::common::{NumberType,PI};
use crate::onb::ONB;

pub trait PDF {
    fn value (&self, direction: Vec3) -> NumberType;
    fn generate(&self) -> Vec3;
}

// Must be returnable by value
// and must therefore be simple.
#[derive(Clone, Copy)]
pub enum PDFMaterialEnum {
    CosinePDF(CosinePDF),
    SpherePDF(SpherePDF),
}
impl<'a> PDF for PDFMaterialEnum {
    fn value (&self, direction: Vec3) -> NumberType {
        match self {
            PDFMaterialEnum::CosinePDF(c) => c.value(direction),
            PDFMaterialEnum::SpherePDF(s) => s.value(direction),
        }
    }
    fn generate(&self) -> Vec3 {
        match self {
            PDFMaterialEnum::CosinePDF(c) => c.generate(),
            PDFMaterialEnum::SpherePDF(s) => s.generate(),
        }
    }
}

// PDF that is supported by MixPDF
#[derive(Clone, Copy)]
pub enum PDFEnum<'a> {
    PDFMaterialEnum(PDFMaterialEnum),
    HittablePDF(HittablePDF<'a>),
}
impl<'a> PDF for PDFEnum<'a> {
    fn value (&self, direction: Vec3) -> NumberType {
        match self {
            PDFEnum::PDFMaterialEnum(c) => c.value(direction),
            PDFEnum::HittablePDF(h) => h.value(direction),
        }
    }
    fn generate(&self) -> Vec3 {
        match self {
            PDFEnum::PDFMaterialEnum(c) => c.generate(),
            PDFEnum::HittablePDF(h) => h.generate(),
        }
    }
}

#[derive(Clone, Copy)]
#[derive(Default)]
pub struct SpherePDF {}
impl SpherePDF {pub fn new() -> Self {Self {}}}
impl PDF for SpherePDF {
    fn value (&self, _direction: Vec3) -> NumberType {
        1.0/(4.0*PI)
    }
    fn generate(&self) -> Vec3 {
        Vec3::random_unit()
    }
}
impl SpherePDF {
    pub fn create() -> PDFMaterialEnum {
        PDFMaterialEnum::SpherePDF(SpherePDF{})
    }
}

#[derive(Clone, Copy)]
pub struct CosinePDF {
    onb: ONB
}
impl PDF for CosinePDF {
    fn value (&self, direction: Vec3) -> NumberType {
        let cosine = direction.normalized().dot(self.onb.w);
        (cosine/PI).max(0.0)
    }
    fn generate(&self) -> Vec3 {
        self.onb.local(Vec3::random_cosine_direction())
    }
}
impl CosinePDF {
    pub fn create(w: Vec3) -> PDFMaterialEnum {
        PDFMaterialEnum::CosinePDF(CosinePDF { onb: ONB::build_from_w(w)})
    }
}

#[derive(Clone, Copy)]
pub struct HittablePDF<'a> {
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
    pub fn create(object:&'a HittableObject<'a>, o: Vec3) -> PDFEnum<'a> {
        PDFEnum::HittablePDF(HittablePDF {object, o,})
    }
}

pub struct MixPDF<'a,'b,'c,'d> {
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
    pub fn new(p1: &'a PDFEnum<'b>, p2: &'c PDFEnum<'d>, f: NumberType) -> Self {
        MixPDF { p1,p2,f } 
    }
}
