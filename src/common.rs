
pub type NumberType = f32;

#[allow(clippy::excessive_precision)]
pub const PI: NumberType = 3.1415926535897932385;


//const PI: NumberType = 3.1415926535897932385;

pub fn deg2rad(deg: NumberType) -> NumberType {
    deg*PI/180.0
}
