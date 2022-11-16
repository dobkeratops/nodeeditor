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
pub type V3<T> = (T,T,T);
pub type V4<T> = (T,T,T,T);
pub type V2i = V2<i32>;
pub type V2f = V2<f32>;
pub type V2u = V2<usize>;
pub type Vec2 = V2i;
pub fn v2splat<T:Copy>(s:T)->V2<T> {(s,s)}
pub fn v2make<T>(a:T,b:T)->V2<T> {(a,b)}
pub fn v3make<T>(a:T,b:T,c:T)->V3<T> {(a,b,c)}
pub fn v3splat<T:Copy>(s:T)->V3<T> {(s,s,s)}
pub fn v4make<T>(a:T,b:T,c:T,d:T)->V4<T> {(a,b,c,d)}
pub fn v4splat<T:Copy>(s:T)->V4<T> {(s,s,s,s)}
pub trait VecElem : Fromi32+Copy+PartialOrd+std::fmt::Debug+Mul<Output=Self> + Add<Output=Self>+ Sub<Output=Self>+ Div<Output=Self>+Rem<Output=Self>+Div<Output=Self>+Neg<Output=Self> {}
impl VecElem for i32{}
impl VecElem for f32{}
impl VecElem for isize{}
trait Fromi32{fn fromi32(x:i32)->Self;}
impl Fromi32 for usize{fn fromi32(x:i32)->Self{x as usize}}
impl Fromi32 for isize{fn fromi32(x:i32)->Self{x as isize}}
impl Fromi32 for u32{fn fromi32(x:i32)->Self{x as u32}}
impl Fromi32 for i32{fn fromi32(x:i32)->Self{x as i32}}
impl Fromi32 for f32{fn fromi32(x:i32)->Self{x as f32}}
macro_rules! vecbinops{
	($(($fname2:ident,$fname3:ident,$fname4:ident; $methodname:ident)),*)=>{$(
		pub fn $fname2<T:VecElem>((x0,y0):V2<T>,(x1,y1):V2<T>)->V2<T>{v2make(x0.$methodname(x1),y0.$methodname(y1))}
		pub fn $fname3<T:VecElem>((x0,y0,z0):V3<T>,(x1,y1,z1):V3<T>)->V3<T>{v3make(x0.$methodname(x1),y0.$methodname(y1),z0.$methodname(z1))}
		pub fn $fname4<T:VecElem>((x0,y0,z0,w0):V4<T>,(x1,y1,z1,w1):V4<T>)->V4<T>{v4make(x0.$methodname(x1),y0.$methodname(y1),z0.$methodname(z1),w0.$methodname(w1))}
	)*}
}

pub fn v2contains<T:PartialOrd>((vmin,vmax):(V2<T>,V2<T>), (x,y):V2<T>)->bool{
	x >= vmin.0 &&  x <vmax.0 && y>=vmin.0 && y<vmax.1
}
vecbinops!{(v2add,v3add,v4add;add),(v2sub,v3sub,v4sub;sub),(v2mul,v3mul,v4mul;mul),(v2div,v3div,v4div;div)}
//pub fn v2add<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0+b.0, a.1+b.1)}
//pub fn v2sub<T:VecElem >(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0-b.0, a.1-b.1)}
//pub fn v2mul<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0*b.0, a.1*b.1)}
//pub fn v2div<T:VecElem>(a:V2<T>,d:T)->V2<T>{v2make(a.0/d, a.1/d)}

pub fn v2avr<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make((a.0+b.0)/T::fromi32(2), (a.1+b.1)/T::fromi32(2))}
pub fn v2acc<T:VecElem>(acc:&mut V2<T>,b:V2<T>){*acc=v2add(*acc,b);}
pub fn v2mul_acc<T:VecElem>(acc:&mut V2<T>,b:V2<T>,c:V2<T>){*acc=v2add(*acc,v2mul(b,c));}
pub fn v2expand<T:VecElem>((vmin,vmax):(V2<T>,V2<T>), ofs:V2<T>)->(V2<T>,V2<T>){(v2sub(vmin,ofs),v2add(vmax,ofs))}
pub fn v2mod<T:VecElem>(a:V2<T>,d:T)->V2<T>{v2make(a.0%d, a.1%d)}
pub fn v2dot<T:VecElem>(a:V2<T>,b:V2<T>)->T{v2hsum(v2mul(a,b))}
pub fn v2hsum<T:VecElem>(a:V2<T>)->T{a.0+a.1}
pub fn v2lerp<T:VecElem>(a:V2<T>,b:V2<T>,f:T)->V2<T>{v2madd(a,v2sub(b,a),f)}
pub fn v2madd<T:VecElem>(a:V2<T>,b:V2<T>,f:T)->V2<T>{v2make(a.0 + b.0*f, a.1+ b.1*f)}
pub fn v2hmul<T:Mul<Output=T>>(a:V2<T>)->T{a.0*a.1}
pub fn v2maxcomp<T:VecElem>(a:V2<T>)->(usize,T){if a.0 > a.1 {(0,a.0)}else{(1,a.1)}}
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
