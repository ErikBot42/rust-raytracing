
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
static mut RNG: Option<SmallRng> = None;
use crate::common::NumberType;

// will probably produce weird results when multithreading
pub fn rng_seed() {
    unsafe {
        RNG = Some(SmallRng::from_entropy());
    }
}
pub fn rng() -> &'static mut SmallRng{
    unsafe {
        RNG.as_mut().unwrap()
    }
}
pub fn random_range(a: NumberType, b:NumberType) -> NumberType {
    let a: NumberType = rng().gen_range(a..b);a
}


pub fn random_val() -> NumberType {let a: NumberType = rng().gen();a}
