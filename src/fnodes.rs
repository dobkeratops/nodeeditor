#![allow(non_camel_case_types)]
use super::*;

macro_rules! define_fnodes{
    // quite fiddly to write but worth the effort!
    (
        $(
            pub fn $fnname:ident( 
                $($argname:ident : $argtype:ty),*
            )-> $returntype:ty
            $fnbody:block
        )*
	)=>
    {
		// implement the actual functions, they were input to this macro
		$(pub fn $fnname($($argname:$argtype),*)->$returntype {unimplemented!()})*

        #[derive(Clone, Debug, Copy, PartialEq, Serialize, Deserialize)]
        pub enum FNodeType{
            $($fnname),*
        }

        impl FNodeType {
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
            pub fn get_param_list(&self)->&'static [(&'static str,&'static str)]{
                match self {$(
                    FNodeType::$fnname=>&[$( (stringify!($argname),stringify!($argtype))  ),*]
                ),*}

            }
			pub fn num_input_slots(&self)->usize {
				self.get_param_list().len()
			}
			// initially only support one output.
			pub fn num_outputs(&self)->usize {1}
			pub fn num_slots(&self)->usize{self.num_input_slots()+self.num_outputs()}

			pub fn eval(&self, args:&[SlotTypeRef])->SlotTypeVal {
				assert!(args.len() == self.num_input_slots());
				let mut arg_iter=args.iter();
				match self{
					$(FNodeType::$fnname=>{
						let result = self::$fnname(
							$(
								{
									let optarg:Option<&SlotTypeRef> = arg_iter.next();
									let enum_arg:&SlotTypeRef=optarg.unwrap();
									//let actual_arg:&$argtype = (<&$argtype>::from(*enum_arg));
									let actual_arg:$argtype  = (*enum_arg).into();
									//<$argtype>::from(arg)
									actual_arg
								}
							),*
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
#[derive(Clone,Debug,Serialize,Deserialize)]

pub struct FloatDefaultOne(pub f32);
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct FloatDefaultZero(pub f32);

impl Default for FloatDefaultOne{fn default()->Self{Self(1.0)}}
impl Default for FloatDefaultZero{fn default()->Self{Self(0.0)}}

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

		//the trait bound `&Image2D<f32, 4>: From<&SlotTypeRef<'_>>` is not satisfie
		$(impl<'a> From<SlotTypeRef<'a> > for &'a $slot_type{
			fn from(src:SlotTypeRef<'a> )-> Self{
				match src{
					SlotTypeRef::$ename(r)=>r,
					_=>panic!(),
				}
			}
		})*

		#[derive(Debug,Serialize,Deserialize)]
		pub enum SlotTypeVal {
			$(
				$ename($slot_type)
			),*
		}


		// macro expect all pairs of types to have conversion operator defined.
		// and implements blanket conversion for the any slot-type enum.
		// TODO - is there a way to constrain compat check?
		//
		//he trait bound `FloatDefaultZero: From<SlotTypeRef<'_>>` is not satisfied
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
		//he trait bound `FloatDefaultZero: From<SlotTypeRef<'_>>` is not satisfied
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
		Empty(Empty),
		FloatDefaultOne(FloatDefaultOne),
		FloatDefaultZero(FloatDefaultZero),
		ImageRGBA(Image2D<f32,4>),
		ImageLuma(Image2D<f32,1>)
	}
}
#[derive(Default, Clone, Debug,Serialize,Deserialize)]
pub struct Empty();

impl Default for SlotTypeVal {
	fn default()->Self{Self::Empty(Empty())}
}


#[derive(Default, Clone, Debug,Serialize,Deserialize)]
pub struct Image2D<T,const Channels:usize>{pub data:Vec<T>, pub size:V2u}
pub type Image2dRGBA = Image2D<f32,4>;
pub type Image2dRGB = Image2D<f32,3>;
pub type Image2dLuma = Image2D<f32,1>;
pub type Image2dLumaAlpha = Image2D<f32,2>;
impl<T:Copy,const C:usize> Image2D<T,C> {
    pub fn per_pixel_binary_op<F>(a:&Image2D<T,C>,b:&Image2D<T,C>,func:F)->Image2D<T,C> where F:Fn(T,T)->T {unimplemented!()}
    pub fn per_pixel_trinary_blend<F>(a:&Image2D<T,C>,b:&Image2D<T,C>,c:Image2D<T,1>,func:F)->Image2D<T,C> where F:Fn(T,T,T)->T {unimplemented!()}
    pub fn per_pixel_unary_op<F>(a:&Image2D<T,C>,func:F)->Image2D<T,C> where F:Fn(T)->T {unimplemented!()}
    pub fn from_fn<F>(a:&Image2D<T,C>,func:F)->Image2D<T,C> where F:Fn((usize,usize)) {unimplemented!()}
}


define_fnodes!{
    // "node definitions" is just UI over FUNCTIONS
    // macro will generate the NodeTypes etc etc etc
    // caveat - parameter types must be listed in some kind of DataTypes enum?
    // guts of supporting implementation in impl Image{} etc
    // fn Image::per_pixel_binary_op(src1:&Image,src2:&Image, &Fn((usize,usize),a:f32,b:f32)->f32) ->Image
    pub fn img_add(
        src1:&Image2dRGBA,
        src2:&Image2dRGBA
		)->Image2dRGBA
    {
        Image2dRGBA::per_pixel_binary_op(src1,src2,|a,b|a+b) 
		
    }
	
    pub fn img_sub(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2dRGBA::per_pixel_binary_op(src1,src2,|a,b|a-b)}
    pub fn img_mul(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image2dRGBA::per_pixel_binary_op(src1,src2,|a,b|a*b)}
    pub fn img_mul_add(src1:&Image2dRGBA,src2:&Image2dRGBA,src3:&Image2dRGBA)->Image2dRGBA{
		img_add(img_mul(src1,src2), src2)
	}
	pub fn img_clamp(src1:&Image2dRGBA,lo:&FloatDefaultZero,hi:&FloatDefaultOne)->Image2dRGBA{Image2dRGBA::per_pixel_unary_op(src1,|a| std::cmp::clamp(a,lo,hi))}
    pub fn img_add_const(src1:&Image2dRGBA,val:&FloatDefaultZero)->Image2dRGBA{Image2dRGBA::per_pixel_unary_op(src1,|a| a+offset)}
    pub fn img_mul_const(src1:&Image2dRGBA,val:&FloatDefaultZero)->Image2dRGBA{Image2dRGBA::per_pixel_unary_op(src1,|a| a*offset)}
	pub fn img_add_mul_const(src1:&Image2dRGBA,val1:&FloatDefaultOne,val:&FloatDefaultZero)->Image2dRGBA{
		img_add_const(img_mul_const(src1,val1),val2)
	}
    pub fn img_pow(src1:&Image2dRGBA,val:&FloatDefaultOne)->Image2dRGBA{Image2dRGBA::per_pixel_unary_op(src1,|a| std::float::power(a,val))}
    pub fn img_sin(src1:&Image2dRGBA,freq:&FloatDefaultOne)->Image2dRGBA{Image2dRGBA::per_pixel_unary_op(src1,|a| (a*val).sin())}
    pub fn img_grainmerge(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image::per_pixel_binary_op(src1,src2,|a,b|a+b-0.5)}
    pub fn img_min(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image::per_pixel_binary_op(src1,src2,|a,b|std::cmp::min(a,b))}
    pub fn img_max(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{Image::per_pixel_binary_op(src1,src2,|a,b|std::cmp::max(a,b))}
    pub fn img_noise(src1:&FloatDefaultOne)->Image2dRGBA{let rnd=Rnd::new(seed); Image::from_fn(|_xy| [rnd.float(),rnd.float(),rnd.float(),rnd.float()])  }
    pub fn img_fractal(dim:&FloatDefaultOne)->Image2dRGBA{let rnd=Rnd::new(seed); Image::from_fn(|_xy| [rnd.float(),rnd.float(),rnd.float(),rnd.float()])  }
    pub fn img_blend(src1:&Image2dRGBA,src2:&Image2dRGBA,src3:&Image2dLuma)->Image2dRGBA{
        Image2dRGBA::per_pixel_trinary_blend(src1,src2,src3,|a,b,f| (b-a)*f+a)
    }
    pub fn img_warp(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{let rnd=Rnd::new(seed); Image::from_fn(|_xy| [rnd.float(),rnd.float(),rnd.float(),rnd.float()])  }
 
    pub fn img_hardlight(src1:&Image2dRGBA,src2:&Image2dRGBA)->Image2dRGBA{ // from photoshop blend mdoes..
        ImageRGBA::per_pixel_binary_op(src1,src2,
            |a,b| if b<0.5 { a*b*2.0} else { 1.0- (1.0-a)*(1.0-b) * 2.0}
        )
    }
    //pub fn img_split_channels(
        //src1:&Image2dRGBA
    //) -> Image4
    //{
		//unimplemented!()
    //}
	
    
}

