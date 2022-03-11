

use crate::vector::Vec3;
use crate::common::{NumberType,PI};
use crate::onb::ONB;

pub trait PDF {
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
pub struct CosinePDF {
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
    pub fn new(w: Vec3) -> PDFEnum<'a> {
        PDFEnum::CosinePDF(CosinePDF { onb: ONB::build_from_w(w)})
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
