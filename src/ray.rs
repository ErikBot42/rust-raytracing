
use crate::vector::Vec3;
use crate::common::NumberType;


#[derive(Debug, Default, Copy, Clone)]
pub struct Ray {
    pub ro: Vec3,
    pub rd: Vec3,
    pub rd_inv: Vec3,
}

impl Ray {
    pub fn at(&self, t: NumberType) -> Vec3 { self.ro + self.rd*t }
    pub fn new(ro: Vec3, rd: Vec3) -> Self { Ray {ro, rd, rd_inv: rd.inv()} }
    pub fn newi(ro: Vec3, rd: Vec3, rd_inv: Vec3) -> Self { Ray {ro, rd, rd_inv} }
}
