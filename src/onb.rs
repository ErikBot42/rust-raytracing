
use crate::vector::Vec3;
#[derive(Clone, Copy)]
pub struct ONB {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}
impl ONB {
    pub fn local(&self,p: Vec3)->Vec3 {
        self.u*p.x+self.v*p.y+self.w*p.z
    }
    pub fn build_from_w(n: Vec3) -> Self {
        let w = n.normalized();
        let a = if w.x.abs() > 0.9 {Vec3::new(0.0,1.0,0.0)} else {Vec3::new(1.0,0.0,0.0)};
        let v = w.cross(a).normalized();
        let u = w.cross(v);
        ONB {u,v,w}
    }
}
