
use rand::Rng;
use rand_distr::StandardNormal;
use num_traits::real::Real;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign, Index, IndexMut};

use crate::random::*;
use crate::common::*;

pub type Vec3 = V3<NumberType>;

#[derive(Debug, Copy, Clone, Default)]
pub struct V3<T>
{
    pub x: T,
    pub y: T,
    pub z: T,
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
    pub fn dot(self, other: V3<T>) -> T {
        let tmp = self*other;
        tmp.x+tmp.y+tmp.z
    }
    pub fn dot2(self) -> T {
        self.dot(self)
    }
    pub fn cross(self, other: V3<T>) -> Self
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
    pub fn length(self) -> T {
        self.dot2().sqrt()
    }
    pub fn normalized(self) -> V3<T> {
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
    //pub fn random_unit() -> V3<NumberType> {
    //    Self::random_range(-1.0,1.0).normalized()
    //}
    //pub fn random_in_unit_hemisphere(n:Vec3) -> V3<NumberType> {
    //    let r = Self::random_unit();
    //    if r.dot(n)>0.0 {r} else {-r}
    //}
    //
    pub fn random_dir() -> Vec3 {
        Vec3::new(
            rng().sample(StandardNormal),
            rng().sample(StandardNormal),
            rng().sample(StandardNormal)
            )
    } 
    pub fn random_unit() -> Vec3 {
        Vec3::random_dir().normalized()
    }


    //pub fn random_cosine_direction() -> Vec3 {
    //    let r1 = random_val();
    //    let r2 = random_val();
    //    let z = (1.0-r2).sqrt();

    //    let phi = 2.0*PI*r1;
    //    let x = phi.cos()*r2.sqrt();
    //    let y = phi.sin()*r2.sqrt();

    //    Vec3::new(x,y,z)
    //}

    pub fn random_cosine_direction() -> Vec3 {
        let mut v = Vec3::random_unit();
        v.z = v.z.abs();
        v
    }
    pub fn random_in_unit_disk() -> Vec3 {
        Vec3::new(
            rng().sample(StandardNormal),
            rng().sample(StandardNormal),
            0.0
            ).normalized()
    }

    //pub fn random_cosine_direction() -> Vec3 {
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
    //fn refract(self, n: Vec3, etiot: NumberType) -> Vec3 {
    //    let cos_theta = -self.dot(n).min(1.0);
    //    let r_out_prep = (self + n*cos_theta)*etiot;
    //    let r_out_parallel = n*(-(1.0 - r_out_prep.dot2()).abs().sqrt());
    //    r_out_prep+r_out_parallel
    //}
}



//pub type NumberType = f32;
