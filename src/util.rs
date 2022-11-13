use super::*;
use std::ops::{Add,Sub,Mul,Div,Neg,Rem};
pub trait FromPosSize<T>{
	fn from_pos_size(pos:T,size:T)->Self;
}
fn from_pos_size<T:Copy,X:FromPosSize<T>>(pos:T,size:T)->X{
	X::from_pos_size(pos,size)
}


//use sdl2::image::LoadTexture;
pub type V2<T> = (T,T);
pub type V2i = V2<i32>;
pub type V2f = V2<f32>;
pub type V2u = V2<usize>;
pub type Vec2 = V2i;
pub fn v2make<T>(a:T,b:T)->V2<T> {(a,b)}
pub trait VecElem : Copy+PartialOrd+std::fmt::Debug+Mul<Output=Self> + Add<Output=Self>+ Sub<Output=Self>+ Div<Output=Self>+Rem<Output=Self>+Div<Output=Self>+Neg<Output=Self> {}
impl VecElem for i32{}
impl VecElem for f32{}
impl VecElem for isize{}
pub fn v2add<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0+b.0, a.1+b.1)}
pub fn v2acc<T:VecElem>(acc:&mut V2<T>,b:V2<T>){*acc=v2add(*acc,b);}
pub fn v2mul_acc<T:VecElem>(acc:&mut V2<T>,b:V2<T>,c:V2<T>){*acc=v2add(*acc,v2mul(b,c));}
pub fn v2sub<T:VecElem >(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0-b.0, a.1-b.1)}
pub fn v2div<T:VecElem>(a:V2<T>,d:T)->V2<T>{v2make(a.0/d, a.1/d)}
pub fn v2mod<T:VecElem>(a:V2<T>,d:T)->V2<T>{v2make(a.0%d, a.1%d)}
pub fn v2dot<T:VecElem>(a:V2<T>,b:V2<T>)->T{v2hsum(v2mul(a,b))}
pub fn v2hsum<T:VecElem>(a:V2<T>)->T{a.0+a.1}
pub fn v2lerp<T:VecElem>(a:V2<T>,b:V2<T>,f:T)->V2<T>{v2madd(a,v2sub(b,a),f)}
pub fn v2madd<T:VecElem>(a:V2<T>,b:V2<T>,f:T)->V2<T>{v2make(a.0 + b.0*f, a.1+ b.1*f)}
pub fn v2hmul<T:Mul<Output=T>>(a:V2<T>)->T{a.0*a.1}
pub fn v2maxcomp<T:VecElem>(a:V2<T>)->(usize,T){if a.0 > a.1 {(0,a.0)}else{(1,a.1)}}
pub fn v2mul<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0*b.0, a.1*b.1)}
pub fn minp<T:PartialOrd>(a:T,b:T)->T{if a<b{a} else {b}}
pub fn maxp<T:PartialOrd>(a:T,b:T)->T{if a>b{a} else {b}}
pub fn v2max<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(maxp(a.0,b.0), maxp(a.1,b.1))}
pub fn v2min<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(minp(a.0,b.0), minp(a.1,b.1))}
pub fn v2scale<T:VecElem>(a:V2<T>,f:T)->V2<T>{v2make(a.0*f, a.1*f)}
pub fn v2abs<T:VecElem+num_traits::Signed>(a:V2<T>)->V2<T>{v2make(a.0 .abs(), a.1 .abs())}
pub fn v2ymx<T:VecElem>(a:V2<T>)->V2<T>{v2make(a.1,-a.0)}
pub fn v2_x0<T:VecElem>(a:V2<T>)->V2<T>{v2make(a.0,a.1-a.1)}
pub fn v2_0y<T:VecElem>(a:V2<T>)->V2<T>{v2make(a.0-a.0,a.1)}
pub fn v2myx<T:VecElem>(a:V2<T>)->V2<T>{v2make(-a.1,a.0)}
pub fn v2i32_to_u32(a:V2<i32>)->V2<u32>{v2make(a.0 as _,a.1 as _)}
pub fn v2f32_to_i32(a:V2<f32>)->V2<i32>{v2make(a.0 as _,a.1 as _)}
pub fn v2i32_to_f32(a:V2<i32>)->V2<f32>{v2make(a.0 as _,a.1 as _)}
pub fn v2u32_to_i32(a:V2<i32>)->V2<i32>{v2make(a.0 as _,a.1 as _)}
pub fn v2i32_to_usize(a:V2<i32>)->V2<usize>{v2make(a.0 as _,a.1 as _)}
pub fn v2mxmy<T:VecElem>(a:V2<T>)->V2<T>{v2make(-a.0,-a.1)}
pub fn v2manhattan_dist<T:VecElem+num_traits::Signed>(a:V2<T>,b:V2<T>)->T {
    v2hsum(v2abs(v2sub(b,a)))
}
pub fn v2sqr<T:VecElem>(a:V2<T>)->T{v2dot(a,a)}
pub fn v2distSqr<T:VecElem>(a:V2<T>,b:V2<T>)->T{v2sqr(v2sub(b,a))}
pub fn v2muldiv<T:VecElem>(a:V2<T>,m:T,d:T)->V2<T>{ v2make( (a.0*m)/d, (a.0*m)/d ) }
