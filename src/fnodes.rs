#![allow(non_camel_case_types)]
#![allow(unused_imports)]
use serde::{Serialize, Deserialize};
use std::ops::Index;
use super::*;

macro_rules! define_fnodes{
    (	
        $(
            pub fn $fnname:ident( 
                $($argname:ident : &$argtype:ident),*
            )-> $returntype:ident
            $fnbody:block
        )*
	)=>
	{
		// implement the actual functions, they were input to this macro
		$(pub fn $fnname($($argname:&$argtype),*)->$returntype {$fnbody})*

        #[derive(Clone, Debug, Copy, PartialEq, Serialize, Deserialize)]
        pub enum FNodeType{
            $($fnname),*
        }
		#[derive(Debug)]
		// reflected descriptions for UI.
		pub struct SlotDesc{pub name:&'static str, pub typeid:SlotTypeId}
		#[derive(Debug)]
		pub struct FNodeTypeDesc{pub name:&'static str, pub inputs:&'static [SlotDesc],pub outputs:&'static [SlotDesc]}

        impl FNodeType {
			pub fn node_types()->&'static[FNodeTypeDesc] {
				&[$(
					FNodeTypeDesc{
						name:stringify!($fnname),
						inputs:&[$(
							SlotDesc{name:stringify!($argname),typeid:SlotTypeId::$argtype}
						),*],
						outputs:&[SlotDesc{name:"output",typeid:SlotTypeId::$returntype}]
					}
				),*]
			}
            pub fn name(&self)->&'static str {
                match self {$(
                    FNodeType::$fnname => stringify!($fnname)
                ),*}
            }
            pub fn output_type(&self)->&'static str{
                match self {$(
                    FNodeType::$fnname=>stringify!($returntype)
                ),*}
            }
            pub fn get_param_list(&self)->&'static [SlotDesc]{
                match self {$(
                    FNodeType::$fnname=>&[$(
						SlotDesc{name:stringify!($argname),typeid:SlotTypeId::$argtype}
					),*]
                ),*}
			}

			pub fn num_inputs(&self)->usize {
				// static - we hope the optimizer can figure this out..
				self.get_param_list().len()
			}
			// initially only support one output.
			pub fn num_outputs(&self)->usize {1}
			pub fn num_slots(&self)->usize{self.num_inputs()+self.num_outputs()}

			pub fn eval(&self, args:&[SlotTypeRef])->SlotTypeVal {
				assert!(args.len() == self.num_inputs());
				let mut arg_iter=args.iter();
				
				match self{
					$(FNodeType::$fnname=>{
						println!("eval {:?}",stringify!($fnname));
						let result = self::$fnname(
							$({
								let optarg:Option<&SlotTypeRef> = arg_iter.next();
								let enum_arg:&SlotTypeRef=optarg.unwrap();
								//let actual_arg:&$argtype = (<&$argtype>::from(*enum_arg));
								let actual_arg:&$argtype  = (*enum_arg).into();
								//<$argtype>::from(arg)
								actual_arg
							}),*
						);
						SlotTypeVal::from(result)
					}),*
				}
			}
        }

        fn foreach_node_type(f:& mut dyn FnMut(FNodeType)){
            $( f(FNodeType::$fnname); )*
        }

    }
}
impl Default for FNodeType{fn default()->Self{Self::img_add}}

#[test]
fn dump_node_types(){
    println!("\n***NODE DEFINITION LIST***");
    
    foreach_node_type(&mut|x|{
        println!("node type: {:?}",x.name());
        println!("\targlist:{:?}",x.get_param_list());
        println!("\toutputType:{:?}",x.output_type());
    });
    println!("\n");
}



#[derive(Copy,Clone,Debug,Serialize,Deserialize)]
pub struct Float32(pub f32);

impl From<f32> for Float32{fn from(x:f32)->Self{Self(x)}}
impl From<&Float32> for f32{fn from(x:&Float32)->Self{x.0}}

impl Default for Float32{fn default()->Self{Self(0.0)}}


// macro rolls reference and type version of a multi-type variant
// and conversions to & from plain values
macro_rules! slot_types{

	(enum SlotTypeVal{
		$($ename:ident($slot_type:ty)),*
	})=>{
		#[derive(Copy,Clone,Debug)]
		pub enum SlotTypeRef<'a> {
			$(
				$ename(&'a $slot_type)
			),*
		}
		
		#[derive(Debug)]
		pub enum SlotTypeId {$($ename),*}

		//the trait bound `&Image2D<f32, 4>: From<&SlotTypeRef<'_>>` is not satisfie
		$(impl<'a> From<SlotTypeRef<'a> > for &'a $slot_type{
			fn from(src:SlotTypeRef<'a> )-> Self{
				match src{
					SlotTypeRef::$ename(r)=>r,
					_=>panic!("need to stop user from connecting incompatible nodes, or auto-insert conversion node"),
				}
			}
		})*

		#[derive(Debug,Clone)]
		pub enum SlotTypeVal {
			$(
				$ename($slot_type)
			),*
		}

		impl SlotTypeVal{
			pub fn type_id(&self)->SlotTypeId {
				match self {
					$(Self::$ename(_)=> SlotTypeId::$ename),*
				}
			}
		}
		impl<'a> SlotTypeRef<'a> {
			pub fn type_id(&self)->SlotTypeId {
				match self {
					$(Self::$ename(_)=> SlotTypeId::$ename),*
				}
			}
		}
		impl SlotTypeId {
			pub fn name(&self)->&'static str {
				match self{
					$(Self::$ename=>&stringify!($ename)),*
				}
			}
		}

		impl<'a> From<&'a SlotTypeVal> for SlotTypeRef<'a>{
			fn from(src:&'a SlotTypeVal)->Self{
				match src{
					$(SlotTypeVal::$ename(x)=>SlotTypeRef::$ename(x)),*
				}
			}
		}

		// macro expect all pairs of types to have conversion operator defined.
		// and implements blanket conversion for the any slot-type enum.
		// TODO - is there a way to constrain compat check?
		//
		//he trait bound `Float32: From<SlotTypeRef<'_>>` is not satisfied
		/*
		$(
			
			impl<'a> From<SlotTypeRef<'a>> for &'a $slot_type {
				fn from(src:SlotTypeRef<'a>)->&'a $slot_type {
					src.into()
				}
			}
		)*
		*/
			
		$(
			impl From<$slot_type> for SlotTypeVal {
				fn from(src:$slot_type)->SlotTypeVal{
					SlotTypeVal::$ename(src)
				}
			}

		)*
		//he trait bound `Float32: From<SlotTypeRef<'_>>` is not satisfied
/* 			impl<'a> From<&'a $slot_type> for SlotTypeRef<'a> {
				fn from(&'a self)->SlotTypesRef<'a> {
					SlotTypeRef::$ename(self)
				}
			}
*/
		// nested fn acheives list X list
		fn from_slot_type_ref<'a,T>(src:SlotTypeRef<'a>) -> T
			where T: $(From<&'a $slot_type>+)* Default
		{
			match src {
				$(SlotTypeRef::$ename(x)=>{
					T::from(x)
				}),*
			}
		}
		
		// nested fn acheives list X list
		//fn into_slot_type_ref<'a,T>(src:&'a T) -> SlotTypeRef<'a>{
		$(impl<'a> Into<SlotTypeRef<'a>> for &'a $slot_type{
			fn into(self)->SlotTypeRef<'a> {
				SlotTypeRef::$ename(self)
			}
		})*

	}
}

slot_types! {
	enum SlotTypeVal {
		Filename(Filename ),
		Float32(Float32),
		Image2dRGBA(Image2dRGBA),
		Image2dLuma(Image2dLuma)
	}
}
#[derive(Debug,Clone,PartialEq,Default)]
pub struct Filename(pub String);
impl<'a> From<&'a Filename> for &'a str {
	fn from(src:&'a Filename)->Self {&src.0}
}
//impl Default for SlotTypeVal {
	//fn default()->Self{Self::Empty(())}
//}

pub struct Rnd(pub u32);
impl Rnd{
	pub fn float(&mut self)->f32 {
		self.0 = self.0 & 0x090412 ^ (self.0>>15) ^ 0x9512512^(self.0<<4) ^ self.0>>19 ^0x12412;
		let irand =self.0 &0xffff;
		(irand as f32) * (1.0/(0x10000 as f32))
	}
	pub fn float4(&mut self)->[f32;4]{[self.float(),self.float(),self.float(),self.float()]}
}

pub trait ImgComp : 'static + Default+std::fmt::Debug+Clone+Copy+Serialize+DeserializeOwned+PartialEq+PartialOrd{}
impl ImgComp for f32{}
impl ImgComp for u8{}
impl ImgComp for i8{}
impl ImgComp for i32{}
impl ImgComp for i16{}


#[derive(Default, Clone, Debug)]
pub struct Image2D<T:ImgComp=f32,const CH:usize=4>{
	pub data:Vec<[T;CH]>,
	pub size:V2u
}


pub type Image2dRGBA = Image2D<f32,4>;
pub type Image2dRGB = Image2D<f32,3>;
pub type Image2dLuma = Image2D<f32,1>;
pub type Image2dLumaAlpha = Image2D<f32,2>;

impl<T:ImgComp+DeserializeOwned,const C:usize> Image2D<T,C> {
	pub fn map_pixels<X:ImgComp,F:FnMut(&[T;C])->[X;M], const M:usize>(&self,f:F)->Image2D<X,M> {
		Image2D{
			size:self.size,
			data:self.data.iter().map(f).collect::<Vec<_>>()
		}
	}
	pub fn map_channels<X:ImgComp,F:Fn(T)->X>(&self,f:F)->Image2D<X,C> {
		Image2D{
			size:self.size,
			data:self.data.iter().map(|p|{
				let mut out=[X::default();C];
				for i in 0..C{out[i]=f(p[i]);}
				out
			}).collect::<Vec<_>>()
		}
	}

	pub fn bin_op<X:ImgComp,Y:ImgComp,F:FnMut(&[T;C],&[X;C])->[Y;C]>(&self,other:&Image2D<X,C>,mut f:F)->Image2D<Y,C> {
		assert!(self.size==other.size);
		Image2D::<Y,C>{
			size:self.size,
			data:self.data.iter().zip(other.data.iter()).map(|(a,b)|f(a,b)).collect()
		}
	}

	pub fn linear_index(&self, pos:V2u)->usize{pos.0 + pos.1 * self.size.0}
	pub fn at(&self,pos:V2u)->&[T;C]{ let i=self.linear_index(pos); &self.data[i] }
	pub fn at_mut(&mut self,pos:V2u)->&mut [T;C]{ let i=self.linear_index(pos);&mut self.data[i] }
    pub fn per_pixel_op1<'a,F>(src1:&Image2D<T,C>,func:F)->Image2D<T,C>
		where F:Fn(T)->T 
	{
		Image2D::from_fn(src1.size, 
			|xy|{
				let pa=*src1.at(xy);
				let mut tmp=[T::default();C];
				for i in 0..C{tmp[i]=func(*pa.index(i))}
				tmp
			}
		)
	}

    pub fn per_pixel_op2<F>(src1:&Image2D<T,C>,src2:&Image2D<T,C>,func:F)->Image2D<T,C>
		where F:Fn(T,T)->T 
	{	
		Image2D::from_fn(src1.size, 
			|xy|{
				let (pa,pb)=(src1.at(xy),src2.at(xy));
				let mut tmp=[T::default();C];
				for i in 0..C{tmp[i]=func(*pa.index(i),*pb.index(i))}
				tmp
			}
		)
	}
	pub fn per_pixel_op3<F>(src1:&Image2D<T,C>,src2:&Image2D<T,C>,src3:&Image2D<T,C>,func:F)->Image2D<T,C> 
		where F:Fn(T,T,T)->T 
	{	Image2D::from_fn(src1.size,
		|pos|{
			let (v1,v2,v3)=(src1.at(pos),src2.at(pos),src3.at(pos));
			let mut tmp=[T::default();C];
			for i in 0..C{tmp[i]=func(*v1.index(i),*v2.index(i),*v3.index(i))}
			tmp
		})
	}

	pub fn per_pixel_trinary_blend<F>(src1:&Image2D<T,C>,src2:&Image2D<T,C>,src3:&Image2D<T,1>,func:F)->Image2D<T,C> 
		where F:Fn(T,T,T)->T 
	{
		Image2D::from_fn(src1.size,
			|pos|{
				let mut tmp=[T::default();C];
				let (v1,v2,v3)=(src1.at(pos),src2.at(pos),src3.at(pos));
				// 'blend' = single channel for last parameter; it is broadcast
				for i in 0..C{tmp[i]=func(*v1.index(i),*v2.index(i),*v3.index(0));}
				tmp
			}
		)
	}
	pub fn from_fn<F>(size:V2u, mut func:F)->Image2D<T,C>
		where F:FnMut(V2u)->[T;C]
	{
		let val = [T::default();C];
		let num:usize = v2hmul(size);
		let mut img=Image2D{data:vec![val;num], size:size};
		let mut ii=0;
		for j in 0..size.1{
			for i in 0..size.0 {
				img.data[ii]=func(v2make(i,j));
				ii+=1
			}
		}
		img

	}
}

pub fn clamp<T:PartialOrd+Copy>(x:T,lo:T,hi:T)->T{
	if x<lo {lo} else if x<hi {x} else {hi}
}

// "node definitions" is just UI over FUNCTIONS
// macro will generate the NodeTypes etc etc etc
// caveat - parameter types must be listed in some kind of DataTypes enum?
// guts of supporting implementation in impl Image{} etc
// fn Image::per_pixel_binary_op(src1:&Image,src2:&Image, &Fn((usize,usize),a:f32,b:f32)->f32) ->Image
define_fnodes!{
    pub fn img_add(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2D::per_pixel_op2(src1,src2,|a,b|a+b)}
    pub fn img_sub(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2D::per_pixel_op2(src1,src2,|a,b|a-b)}
    pub fn img_mul(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2D::per_pixel_op2(src1,src2,|a,b|a*b)}
    pub fn img_add_const(src1:&Image2dRGBA,val:&Float32)->Image2dRGBA{Image2dRGBA::per_pixel_op1(src1,|a| a+val.0)}
    pub fn img_mul_const(src1:&Image2dRGBA,val:&Float32)->Image2dRGBA{Image2dRGBA::per_pixel_op1(src1,|a| a*val.0)}
	pub fn img_add_mul_const(src1:&Image2dRGBA,val1:&Float32,val2:&Float32)->Image2dRGBA{
		let tmp=img_mul_const(src1,&Float32(val1.0));
		img_add_const(&tmp,val2)
	}
    pub fn img_pow(src1:&Image2dRGBA,val:&Float32)->Image2dRGBA{Image2D::per_pixel_op1(src1,|a| a.powf(val.0))}
    pub fn img_sin(src1:&Image2dRGBA,freq:&Float32)->Image2dRGBA{Image2D::per_pixel_op1(src1,|a| (a*freq.0).sin())}
    pub fn img_grainmerge(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2D::per_pixel_op2(src1,src2,|a,b|a+b-0.5)}
    pub fn img_min(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2D::per_pixel_op2(src1,src2,|a,b|if a<b{a}else{b})}
    pub fn img_max(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2D::per_pixel_op2(src1,src2,|a,b|if a>b{a}else{b})}
    pub fn img_noise(src1:&Float32)->Image2dRGBA{
		let mut rnd=Rnd(0xf983423); 
		Image2D::from_fn(v2make(64,64), |xy|rnd.float4())
	}
    pub fn img_fractal(dim:&Float32)->Image2dRGBA{let mut rnd=Rnd(0x8523912); Image2D::from_fn(v2make(64,64),|_xy| [rnd.float(),rnd.float(),rnd.float(),rnd.float()])  }
    pub fn img_blend(src1:&Image2dRGBA,src2:&Image2dRGBA,src3:&Image2dRGBA)->Image2dRGBA{
        Image2dRGBA::per_pixel_op3(src1,src2,src3,|a,b,f| (b-a)*f+a)
    }
    //pub fn img_warp(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{let rnd=Rnd::new(seed); Image::from_fn(src1.size,|_xy| [rnd.float(),rnd.float(),rnd.float(),rnd.float()])  }
 
    pub fn img_hardlight(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{ // from photoshop blend mdoes..
        Image2D::per_pixel_op2(src1,src2,
            |a,b| if b<0.5 { a*b*2.0} else { 1.0- (1.0-a)*(1.0-b) * 2.0}
        )
    }
    
}

