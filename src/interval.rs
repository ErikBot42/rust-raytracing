use crate::common::NumberType;

#[derive(Clone, Copy)]
pub struct Interval {
    pub min: NumberType,
    pub max: NumberType,
}

impl Interval {
    pub fn new(min: NumberType, max: NumberType) -> Self {Interval {min, max}}
    pub fn contains(&self, x: NumberType) -> bool {self.min <= x && x <= self.max}
    pub fn clamp(&self, x: NumberType) -> NumberType {x.clamp(self.min, self.max)}
    pub fn is_empty(&self) -> bool {self.max<self.min}
    pub fn size(&self) -> NumberType {self.max-self.min}

    pub const EMPTY: Self = Interval {min: NumberType::INFINITY, max:NumberType::NEG_INFINITY};
    pub const UNIVERSE: Self = Interval {min: NumberType::NEG_INFINITY, max:NumberType::INFINITY};
    pub const EPSILON_UNIVERSE: Self = Interval {min: 0.0001, max:NumberType::INFINITY};
}
impl Default for Interval {
    fn default() -> Self {Self::EMPTY}
}
#[test]
fn test_interval() {
    assert!(Interval::UNIVERSE.contains(4.5));
    assert!(!Interval::EMPTY.contains(4.5));
    assert!(Interval::new(4.0,5.6).contains(4.5));
    assert!(!Interval::new(4.0,5.6).contains(6.5));
    assert!(!Interval::UNIVERSE.is_empty());
    assert!(!Interval::EPSILON_UNIVERSE.is_empty());
    assert!(Interval::EMPTY.is_empty());
}
