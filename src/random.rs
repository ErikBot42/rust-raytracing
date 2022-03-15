
use rand_distr::StandardNormal;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
thread_local! {
//static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::from_entropy());
static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::seed_from_u64(48943982));
}
use crate::common::NumberType;


// will probably produce weird results when multithreading
pub fn rng_seed() {
//    unsafe {
//        RNG = Some(SmallRng::from_entropy());
//    }
}

#[inline(always)]
pub fn random_range(a: NumberType, b:NumberType) -> NumberType {
    //let a: NumberType = (a+b)/2;
    let a: NumberType = RNG.with(|rng| rng.borrow_mut().gen_range(a..b));
    a
}
#[inline(always)]
pub fn random_val() -> NumberType {
    let a: NumberType = RNG.with(|rng| rng.borrow_mut().gen());a
}
#[inline(always)]
pub fn random_standard_normal() -> NumberType {
    let a: NumberType = RNG.with(|rng| rng.borrow_mut().sample(StandardNormal));a

}

