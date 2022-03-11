
use crate::vector::Vec3;
use crate::common::NumberType;


#[derive(Debug, Default)]
pub struct Ray {
    pub ro: Vec3,
    pub rd: Vec3,
}

impl Ray {
    pub fn at(&self, t: NumberType) -> Vec3 { self.ro + self.rd*t }
}