

use crate::vector::Vec3;
use crate::common::NumberType;

pub trait Texture {
    fn value(&self, u: NumberType, v: NumberType, p: Vec3) -> Vec3; 
}
#[derive(Clone, Copy)]
#[derive(Debug)]
pub enum TextureEnum<'a> {
    SolidColor(SolidColor),
    CheckerTexture(CheckerTexture<'a>),
}
impl<'a> Default for TextureEnum<'a> {
    fn default() -> Self {
        TextureEnum::SolidColor(SolidColor::default())
    }
}

impl<'a> Texture for TextureEnum<'a> {
    fn value(&self, u: NumberType, v: NumberType, p: Vec3) -> Vec3 {
        match self {
            TextureEnum::SolidColor(s) => s.value(u,v,p),
            TextureEnum::CheckerTexture(c) => c.value(u,v,p),
        }
    }
}
#[derive(Clone, Copy, Default)]
#[derive(Debug)]
pub struct SolidColor {color_value: Vec3,}
impl Texture for SolidColor {fn value(&self, _u: NumberType, _v: NumberType, _p: Vec3) -> Vec3 {self.color_value}}
impl<'a> SolidColor {
    pub fn new(color_value: Vec3) -> TextureEnum<'a> {TextureEnum::SolidColor(SolidColor {color_value})}
}

// no default 
#[derive(Clone, Copy)]
#[derive(Debug)]
pub struct CheckerTexture<'a> {
    pub odd: &'a TextureEnum<'a>,
    pub even: &'a TextureEnum<'a>,
}
impl<'a> Texture for CheckerTexture<'a> {
    fn value(&self, u: NumberType, v: NumberType, p: Vec3) -> Vec3 {
        let fac = 4.0;//0.2;
        let sines = ((u*fac).fract()*2.0-1.0)*((v*fac).fract()*2.0-1.0);
        if sines < 0.0 {self.even.value(u,v,p)} else {self.odd.value(u,v,p)}
    }
}
impl<'a> CheckerTexture<'a> {
    pub fn new(odd: &'a TextureEnum<'a>, even: &'a TextureEnum<'a>) -> TextureEnum<'a> {
        TextureEnum::CheckerTexture(CheckerTexture {odd,even,})
    }
    // inputting 2 arbitrary colors is not possible without allocation
}
