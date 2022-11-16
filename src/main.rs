#![allow(unused_variables,unused_imports,unused_macros,unused_mut)]

extern crate sdl2;
extern crate image;
extern crate num_traits;
extern crate serde;
extern crate serde_derive;
use std::borrow::BorrowMut;
use std::rc::Rc;
use std::slice::SliceIndex;
use sdl2::mouse;
use serde_derive::{Serialize,Deserialize};
use serde::de::DeserializeOwned;
mod util;
pub use util::*;
mod fnodes;
pub use fnodes::*;

use image::GenericImageView;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::video::WindowContext;
use sdl2::render::{WindowCanvas, TextureCreator,Texture, RendererContext};

fn from<A,B:From<A>>(src:A)->B{B::from(src)}

macro_rules! assert_less{
    ($a:expr,$b:expr)=>{if !($a<$b){dbg!(&$a,"<",&$b);assert!(false);}}
}



enum EdState{
    Nothing,
    DraggingFrom(Vec2,DragType),
    AdjustValue(NodeID),
    MoveNode(NodeID),
    Panning
}

type NodeID=usize;
type EdgeID=usize;
#[derive(Debug,Clone)]
enum DragType{
    MoveNode(NodeID),
    DrawEdge(String)
}
enum Action{
    Nothing(),
    CreateEdge(Edge),
    CreateNode(Node),
    PickSlot(SlotAddr),
    PickNode(NodeID,)
}
#[derive(Clone,Debug)]
enum NodeType{
    Function(fnodes::FNodeType),
    Value(fnodes::SlotTypeVal),
}
type GuiColor = [u8;4];
use NodeType::*;

impl NodeType {
    fn name(&self)->&'static str {match self{
        Function(x)=>x.name(),
        Value(x)=>x.type_id().name()
    }}
    fn num_inputs(&self)->usize{match self {
        Function(x)=>x.num_inputs(),
        Value(x)=>0,
    }}
    fn num_outputs(&self)->usize{match self{
        Function(x)=>x.num_outputs(),
        Value(x)=>1,
    }}
    fn num_slots(&self)->usize{self.num_inputs()+self.num_outputs()}
}

type OptRc<T> = Option<Rc<T>>;
fn somerc<T>(x:T)->OptRc<T>{Some(Rc::new(x))}

#[derive(Clone,Debug)]
struct Node {
    typ:NodeType,
    pos:Vec2,
    size:Vec2,
    color:GuiColor,
    text:String,
    cached:OptRc<SlotTypeVal>,
    bitmap:BitmapCache,
}
#[derive(Default)]
struct BitmapCache {
    data:std::cell::RefCell<Option<sdl2::render::Texture>>
}
impl std::fmt::Debug for BitmapCache{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        
        f.debug_struct("BitmapCache")
        .field("data",&())
        .finish()
    }
}

impl BitmapCache{
    fn update(&self, tc:&mut TextureCreator<WindowContext>, src:& Image2D<u8,4>){
        if src.size.0 ==0 || src.size.1 ==0{return;}
        let mut x=self.data.borrow_mut();
        
    
        if x.is_none() {
            *x = Some(tc.create_texture(
                sdl2::pixels::PixelFormatEnum::RGBA8888,
                sdl2::render::TextureAccess::Streaming, src.size.0 as _,src.size.1 as _).expect("ctex"));
        }
    
        unsafe {
            let bytelen = &src.data.len()*4;

            x.as_mut().unwrap().update(None, std::slice::from_raw_parts(&src.data[0] as *const _, bytelen), src.size.0*4);
        }
        
    }        
}
impl Clone for BitmapCache{
    fn clone(&self)->Self{Self{data:default()}}
}
fn default<T:Default>()->T{T::default()}

trait Render{fn render(&self, r:&mut Renderer);}

impl Node {
    fn rect(&self)->(Vec2,Vec2){ (self.pos, v2add(self.pos,self.size)) } 
    fn centre(&self)->Vec2{v2add(self.pos, v2div(self.size,v2splat(2)))}
    fn centre_left(&self)->Vec2{v2add(self.pos, (0,self.size.1/2))}
    fn centre_right(&self)->Vec2{v2add(self.pos, (self.size.0,self.size.1/2))}
    fn set_type(&mut self, nt:NodeType){self.typ=nt;}
}
impl Render for Node {
    fn render(&self, rc:&mut Renderer){
        let pos = v2add(self.pos,v2make(8,16));    // some margin, todo

        fn draw_float(val:f32, rc:&mut Renderer,pos:V2i){
            rc.draw_text(pos,[255,255,255,255],&format!("{:?}",val));
        }
        if let Some(x) = &self.cached {
            let x:&SlotTypeVal =&*x;
            match x {
                SlotTypeVal::Filename(x)=>rc.draw_text(pos, [192,192,192,255], from(x)),
                SlotTypeVal::Float32(x)=>rc.draw_text(pos, [192,192,192,255], &format!("{:?}",x)),
                SlotTypeVal::Image2dRGBA(b)=>{
                    
                    let img2 = b.map_channels(|x|(x.clamp(0.0,1.0)*255.0) as u8);
                    self.bitmap.update(&mut rc.tc,&img2);
                    let tex = self.bitmap.data.borrow();
                    
                    rc.canvas.copy((&*tex).as_ref().unwrap(),None,Rect::new(pos.0,pos.1, 64,64));
                }
                SlotTypeVal::Image2dLuma(x)=>{
                    unimplemented!()
                }
            }
        }
    }
}

trait Contains<X>{fn contains(&self,x:X)->bool;}
impl Contains<Vec2> for (Vec2,Vec2) {
    fn contains(&self, (x,y):Vec2)->bool {
        let ((l,t),(r,b))= *self;
        x>=l && x<r && y>=t && y<b
        
    }
}

struct Font{
    tex:std::cell::RefCell<sdl2::render::Texture>,
    xlat:[u8;256],
}

fn load_texture(filename:&str)->Result<(Vec<u8>,V2u,usize),()>{
    let img=image::open(filename).expect("texfont");

    let conv=|(a,b)| (a as usize, b as usize); 
    let dim= img.dimensions();
    match img {
        image::DynamicImage::ImageRgb8(img)=> Ok((img.into_vec(),conv(dim),3)),
        image::DynamicImage::ImageRgba8(img)=> Ok((img.into_vec(),conv(dim),4)),
        _=> {
            panic!("unknown texture format in dynamicimage")
        }
    }
}

impl Font{
    fn new(tc:&mut TextureCreator<WindowContext>,filename:&str, xlat_str:&str)->Self{
        let img=load_texture(filename).expect(filename);
        //let tex = tc.load_texture(filename).expect("texfont");
        let mut tex=tc.create_texture(sdl2::pixels::PixelFormatEnum::RGBA8888, sdl2::render::TextureAccess::Streaming, img.1 .0 as _,img.1 .1 as _).expect("ctex");
        let mut xlat = [0;256];
        println!("font - converting {:?} chars", xlat_str.len());
        for (i,c) in xlat_str.chars().enumerate() {
            xlat[c as usize] = i as u8;
        }

        tex.update(None, &img.0, img.1 .0 * img.2);
        tex.set_blend_mode(sdl2::render::BlendMode::Blend);


        Font {
            tex:std::cell::RefCell::new(tex),
            xlat
         }
    }
    fn draw_text_f(&mut self, canvas:&mut WindowCanvas, pos:Vec2,color:GuiColor, text:&str){
        let mut pos = pos;
        let screen_charsize=v2make(16,16);
        for c in text.chars(){
            let index=self.xlat[c as usize] as i32;
            let x=(index&15) as i32 *16;
            let y=(index>>4) as i32 *16;
            let rc=Rect::new(x,y, 16u32,16u32);
            let dstrc = Rect::new(pos.0, pos.1, screen_charsize.0 as u32, screen_charsize.1 as u32);
            canvas.set_blend_mode(sdl2::render::BlendMode::Blend);
            self.tex.get_mut().set_color_mod(color[0],color[1],color[2]);
            self.tex.get_mut().set_alpha_mod(color[3]);

            canvas.copy(self.tex.get_mut(), rc,dstrc );

            pos= v2add(pos,v2_x0(screen_charsize));
        }
    }
}
#[derive(Copy,Clone,Debug,std::hash::Hash,PartialEq)]
struct SlotAddr{node:NodeID,slot:usize}
#[derive(Copy,Clone,Debug,std::hash::Hash)]
struct Edge {
    start:SlotAddr,
    end:SlotAddr,
}

#[derive(Clone,Debug)]
struct World {
    nodes:Vec<Node>,
    edges:Vec<Edge>
}

fn draw_bitmap(canvas:&mut WindowCanvas,(data,imgsize):(&[u8],V2u), pos:V2i , screensize:V2i) {
    unimplemented!();
    //tex.update(None, &buffer,64*4);
    //canvas.copy(&tex, None, Rect::new(100,100,64,64));
}
impl World {
    fn remove_node(&mut self, id:NodeID){
        self.nodes.remove(id);
        self.edges.retain(|e|!(e.start.node==id || e.end.node==id));
        for e in self.edges.iter_mut(){
            if e.start.node>=id {e.start.node-=1;}
            if e.end.node>=id {e.end.node-=1;}

        }
    }
    fn pick_node(&self, pos:Vec2)->Option<NodeID>{
        for (i,n) in self.nodes.iter().enumerate(){
            if n.rect().contains(pos){
                
                return Some(i);
            }
        }
        None
    }
    fn slot_pos(&self,sa:&SlotAddr)->Vec2{
        let node=&self.nodes[sa.node];
        let inpn=node.typ.num_inputs();
        let (dx,syf,ns)=if sa.slot <inpn {
            (0,sa.slot,inpn)
        } else {
            (node.size.0, (sa.slot-inpn),node.typ.num_outputs())
        };
        v2make(node.pos.0 + dx, node.pos.1 + ((node.size.1 * (syf as i32*2+1))/(ns as i32*2)))
    }
    fn try_end_drag(&mut self, start:Vec2, end:Vec2)->Action{
        if let Some(x) = self.try_create_edge(start,end){ Action::CreateEdge(x)}
        else{
            Action::Nothing()
        }
    }
    fn try_create_edge(&mut self, start:Vec2,end:Vec2)->Option<Edge>{
        if let Some(eslot)=self.pick_slot(end){
            if let Some(sslot)=self.pick_slot(start) {
                dbg!("dragged between",sslot,eslot);
                if eslot.node!=sslot.node{
                    return Some(Edge{start:sslot,end:eslot})
                }
            }
        } 
        None
    }
    fn pick_elem(&mut self, pos:Vec2)->Elem {
        if let Some(ni) = self.pick_node(pos){
            Elem::Node(ni)
        }else if let Some(slot) =self.pick_slot(pos){
            Elem::Slot(slot)
        } else {
            Elem::Nothing
        }
    }

    fn drag_end(&mut self, start:Vec2, end:Vec2, dt:&DragType){
        println!("dragged{:?},{:?}",start,end);
        match dt{
            DragType::MoveNode(_id)=>{}
            DragType::DrawEdge(_caption)=>{
                let new_edge=if let Some(e)=self.try_create_edge(start,end){
                    // either craete edge to existing node..
                    Some(e)

                } else{

                    // or create a new node to connect to
                    let node_id=self.create_node_at(v2add(end,v2make(32,0)),&format!("Node{:?}",self.nodes.len()), [255,255,128,255], Function(FNodeType::img_add));
                    let slotpos=self.slot_pos(&SlotAddr{node:node_id,slot:0});
                    let node=&mut self.nodes[node_id];
                    let ofs =  v2sub(slotpos,node.pos);
                    v2acc(&mut self.nodes[node_id].pos,ofs);

                    if let Some(edge)=self.try_create_edge(start,end) {
                        Some(edge)
                    } else {None}
                };
                if let Some(edge)=new_edge{    
                    self.edges.retain(|e|e.end!=edge.end);
                    self.edges.push(edge);
                }
            }
        }
    }

    fn render_elem(&mut self, rc:&mut Renderer,el:&Elem, color:GuiColor){
        match el{
            Elem::Nothing=>{}
            Elem::Node(ni)=>{rc.draw_rect_c(v2expand(self.nodes[*ni].rect(),v2splat(2)),color);},
            Elem::Slot(slot)=>{rc.draw_square_centred(self.slot_pos(slot),slotsize as _, color);},
            Elem::Edge(ei)=> self.render_edge(rc,*ei,Some(color)),

        }
    }
    fn create_edge(&mut self, si:SlotAddr,ei:SlotAddr){
        self.edges.push(Edge{start:si,end:ei})
    }
    fn pick_slot(&mut self, pt:Vec2)->Option<SlotAddr>{
        let maxdist=32;
        assert!(self.nodes.len()>0);
        let mut bestdist=maxdist;
        let mut bestslot=None;
        
        for (i,n) in self.nodes.iter().enumerate() {
            for j in 0..n.typ.num_slots(){
                let pos = self.slot_pos(&SlotAddr{node:i,slot:j});
                let dist =v2manhattan_dist(pt,pos);
                let slot = SlotAddr{node:i,slot:j};
                if bestslot.is_none() && dist < bestdist{
                    bestdist=dist;
                    bestslot = Some((slot,pos));
                    
                }
                if dist < bestdist{
                    bestdist=dist;
                    bestslot=Some((slot,pos));
                }
            }
        }
        return bestslot.map(|(s,p)|s);
    }
    fn load_image_at(&mut self, pt:Vec2, filename:String) {
        let img = load_texture(&filename).expect("loading bitmap");
        let newnode= self.create_node_at(pt,&filename,[255,192,128,255], SlotTypeVal::Image2dRGBA(image_from_bitmap(&img)));
        let newnode = self.create_node_at(v2sub(pt, v2make(-64,00) ), &"fiename", [255,192,128,255], SlotTypeVal::Filename(Filename(filename)));
    }

    fn create_node_at<X>(&mut self, pt:Vec2, caption:&str, color:[u8;4],typ:X)->NodeID where NodeType : From <X>{
               let ns=(128,128);
        let newnode=Node{
            cached:None,
            typ:From::from(typ),
            pos:v2sub(pt,v2div(ns,v2splat(2))),
            size:ns,
            text:caption.to_string(),
            color,
            bitmap:default(),
        };
        self.nodes.push(newnode);
        self.nodes.len()-1
    }

    fn eval(&mut self) {
        
        // brute force
        // prepare input value buffer.
        let mut node_inputs:Vec<Vec<OptRc<SlotTypeVal>>> = Vec::new();
        for node in self.nodes.iter_mut(){
            node_inputs.push( vec![None; node.typ.num_inputs()]);
        }
        
        // gather inputs per node,through edges
        for (ei,edge) in self.edges.iter().enumerate(){
            
            let srcnode=&self.nodes[edge.start.node];
            let dstnode=&self.nodes[edge.end.node];
            assert!(srcnode.typ.num_outputs()==1);
            
            //assert_less!(edge.end.slot, dstnode.typ.num_inputs());
            
            if let Some(val)=&srcnode.cached{
                node_inputs[ edge.end.node ] [edge.end.slot] = Some(Rc::clone(val));
            }

        }
        
        for (id,node) in self.nodes.iter_mut().enumerate(){
            node.cached = match &node.typ {
                Value(x)=>{somerc(x.clone())},   // todo - this should be Rc or 
                Function(f)=>Self::eval_node(f, &node_inputs[id])
            }
        }
    }

    fn eval_node<'a>(f:&FNodeType, inputs:&'a [OptRc<SlotTypeVal>])->Option<Rc<SlotTypeVal>> {

        let mut inputs_xlat: Vec<SlotTypeRef<'a>> = Vec::with_capacity(inputs.len());
        for (i,inval) in inputs.iter().enumerate() {
            inputs_xlat.push(
                match inval{
                    None=>{return None;},
                    Some(x)=>SlotTypeRef::from(&**x)
                }
            )
        }
        somerc(f.eval(&inputs_xlat))
    }

    fn example_scene()->Self {
        let mut rnd= fnodes::Rnd(0x1412f0a);
        Self{
            nodes:vec![
                Node{
                    cached:None,typ:Function(FNodeType::img_add),
                    pos:v2make(100,150),size:v2make(64,64),
                    color:[255,0,0,255],text:format!("node0"),
                    bitmap:default(),
                },
                Node{cached:None, typ:Function(FNodeType::img_mul), 
                    pos:v2make(200,150), size:v2make(64,64), 
                    color:[0,255,255,255],
                    text:format!("node1"),
                    bitmap:default(),

                } ,
                Node{cached:None,
                    typ:Value(SlotTypeVal::Float32(from(0.75))),
                    pos:v2make(250,150), size:v2make(64,64), 
                    color:[0,255,255,255],
                    text:format!("node2"),
                    bitmap:default(),
                },
                Node{cached:None,
                    typ:Value(SlotTypeVal::Image2dRGBA(Image2D::from_fn (
                        v2make(64,64),
                        |xy|rnd.float4()
                        ))
                    ), 
                    pos:v2make(250,150), size:v2make(96,96), 
                    color:[0,255,255,255],
                    text:format!("node2"),
                    bitmap:default(),
                },
                Node{cached:None,
                    typ:Value(SlotTypeVal::Image2dRGBA(Image2D::from_fn(
                        v2make(64,64),
                        |xy|[0.5,0.75,1.0,1.0]
                        ))
                    ), 
                    pos:v2make(350,150), size:v2make(96,96), 
                    color:[0,255,255,255],
                    text:format!("node3"),
                    bitmap:default(),
                },

            ],
            edges:vec![],
        }
    }
    fn render_edge(&self, rc:&mut Renderer, id:EdgeID, override_col:Option<GuiColor>){
        let col = override_col.unwrap_or([192,192,192,255]);
        rc.canvas.set_draw_color((col[0],col[1],col[2]));
        let edge=&self.edges[id];   
        
        let endpos=v2add(self.slot_pos(&edge.end),(-slotsize/2,0));
        let startpos=v2add(self.slot_pos(&edge.start),(slotsize/2,0));
        rc.canvas.draw_line(startpos, endpos);
        let dir = (slotsize,0);
        
        //rc.draw_arrowhead(endpos, dir,1.0);
        //rc.draw_arrowhead(v2add(startpos,(-slotsize,0)), dir,1.0);
    }

    fn render_slot(&self, rc:&mut Renderer, sa:&SlotAddr,col:GuiColor){
        let pos = self.slot_pos(sa);
        rc.set_color(col);
        rc.draw_arrowhead(v2add(pos, v2make(-slotsize/2,0)), v2make(slotsize,0),1.0);

    }
}

const slotsize:i32=8;
impl Render for World {
    fn render(&self, rc:&mut Renderer){
        let rgb=|x:[u8;4]|(x[0],x[1],x[2]);
        
        for (ni,node) in self.nodes.iter().enumerate() {
            
            rc.draw_rect_c(node.rect(), node.color);
            
            rc.draw_text(node.pos, node.color, &node.typ.name());
            for i in 0..node.typ.num_slots(){
                self.render_slot(rc,&SlotAddr{node:ni,slot:i},node.color);
            }
            let client_tl =v2add(node.pos, v2make(8,16));// clieht rect of node, todo
            // node expand/contract, todo
            node.render(rc);
        }
        for (ei,edge) in self.edges.iter().enumerate() {
            self.render_edge(rc,ei,Some([192,192,192,255]));
        }    
    }    
}

fn image_from_bitmap(bitmap:&(Vec<u8>,V2u,usize)) ->Image2dRGBA{
    let data=&bitmap.0;
    let size = bitmap.1;
    let channels = bitmap.2;
    let mut buffer:Vec<[f32;4]> = vec![default(); v2hmul(size)*4];

    let ccopy = std::cmp::min(4,channels);
    for i in 0..data.len()/channels {

        for c in 0..ccopy {
            buffer[i][c] = data[i*channels+channels-1-c] as f32 * (1.0/255.0);
        }
        for c in ccopy..4{
            buffer[i][c] = 1.0;
        }
    }
    Image2D{data:buffer, size:size}
}

fn read_string(prompt:&str)->String{
    println!("{}",prompt);
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer);
    buffer
}

fn cellnone<T>()->std::cell::RefCell<Option<T>>{from(None)}

#[derive(Debug,Clone,Copy)]
enum Elem {
    Node(NodeID), Slot(SlotAddr), Edge(EdgeID), Nothing
}

impl From<FNodeType> for NodeType {    fn from(src:FNodeType)->Self{ NodeType::Function(src)}}
impl From<SlotTypeVal> for NodeType {    fn from(src:SlotTypeVal)->Self{ NodeType::Value(src)}}

pub struct Renderer{ canvas:sdl2::render::WindowCanvas, tc:sdl2::render::TextureCreator<WindowContext>, font:Font, hilightcol:[u8;4]}
impl Renderer{
    fn draw_text(&mut self, pos:Vec2, color:GuiColor,text:&str){
        self.font.draw_text_f(&mut self.canvas,pos,color,text);
    }
    fn set_color(&mut self, color:GuiColor){self.canvas.set_draw_color((color[0],color[1],color[2]));}
    fn draw_rect_c(&mut self, (tl,br):(Vec2,Vec2), col:GuiColor){
        self.canvas.set_draw_color((col[0],col[1],col[2]));
        let size = v2sub(br,tl);
        self.canvas.draw_rect(sdl2::rect::Rect::new(tl.0 as _, tl.1  as _, size.0 as _, size.1  as _));

    }
    fn draw_arrowhead(&mut self, pos:Vec2, dir:Vec2,f:f32){
        let perp=((dir.1 as f32*f) as _,(-dir.0 as f32 *f) as _);
        let vertices:[Vec2;3]=[v2add(pos,dir), v2add(pos,perp),v2sub(pos,perp)];
        for (s,e) in [(0,1),(1,2),(2,0)]{
            self.canvas.draw_line(vertices[s],vertices[e]);
        }
    }
    fn draw_square_centred(&mut self, pos:Vec2, radius:f32, color:GuiColor){
        let sz=v2splat(radius as _);
        self.draw_rect_c((v2sub(pos,sz),v2add(pos,sz)), color);
    }
    

}

pub fn main() -> Result<(), String> {
    dbg!(FNodeType::node_types());
    
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let filename = String::from("graph.json");
 
    let mut window = video_subsystem
        .window("Node Graph Editor", 1024, 768)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    window.set_title(&format!("{:?} -node graph editor",&filename));
    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;
    //let ttf = sdl2::ttf::init().expect("ttf init");
    //let font = ttf.load_font("assests/UFont Sans Medium",128).expect("font load");

    canvas.set_draw_color(Color::RGB(255, 0, 0));
    canvas.clear();
    canvas.present();
    let mut event_pump = sdl_context.event_pump()?;

    let mut state =EdState::Nothing;
    let mut mouse_pos =v2make(100,100);
    let mut some_pos=v2make(300,100);

    let mut world = World::example_scene();
    let mut tc:sdl2::render::TextureCreator<WindowContext> = canvas.texture_creator();
    let mut tex = tc.create_texture(None, sdl2::render::TextureAccess::Streaming, 64,64).expect("tex");
    let mut font = Font::new(&mut tc, "assets/font16.png",
        " ¬ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}+-<>@./\\#!?&*\",:;£$___");

    let x:Option<sdl2::render::TextureCreator<WindowContext>> = None;
    let mut rc=Renderer{canvas,tc,font,hilightcol:[128,255, 128,255]};
    
    let mut change_type = None;
    'running: loop {
        let mut mouse_delta=v2splat(0);
        
        
        let pick = world.pick_elem(mouse_pos);
        let node = world.pick_node(mouse_pos);
        

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }|
                Event::KeyDown {keycode: Some(Keycode::Escape),..} => break 'running,
                Event::KeyDown {keycode: Some(Keycode::Backspace),..} => {
                    if let Elem::Node(node)=pick {
                        world.remove_node(node);
                    }
                }
                Event::KeyDown {keycode: Some(Keycode::Num1),..} => change_type=Some(Function(FNodeType::img_add)), 
                Event::KeyDown {keycode: Some(Keycode::Num2),..} => change_type=Some(Function(FNodeType::img_mul)), 
                Event::KeyDown {keycode: Some(Keycode::Num3),..} => change_type=Some(Function(FNodeType::img_sin)), 
                Event::KeyDown {keycode: Some(Keycode::Num4),..} => change_type=Some(Function(FNodeType::img_fractal)), 
                Event::KeyDown {keycode: Some(Keycode::Num5),..} => change_type=Some(Function(FNodeType::img_grainmerge)), 
                Event::KeyDown {keycode: Some(Keycode::Num6),..} => change_type=Some(Function(FNodeType::img_sub)), 
                Event::KeyDown {keycode: Some(Keycode::Num7),..} => change_type=Some(Function(FNodeType::img_blend)), 
                Event::KeyDown {keycode: Some(Keycode::Num8),..} => change_type=Some(Function(FNodeType::img_min)), 
                //Event::KeyDown {keycode: Some(Keycode::Num0),..} => node_type=Some(Function(FNodeType::img_warp)), 
                
                Event::MouseButtonDown { timestamp, window_id, which, mouse_btn, clicks, x, y }=>{
                  
                    state=EdState::DraggingFrom(
                        (x,y),
                        if let Elem::Node(x)=pick{
                            DragType::MoveNode(x)
                        }else {
                            DragType::DrawEdge(format!("draw edge.."))
                        }
                    );
                }
                Event::MouseButtonUp { timestamp, window_id, which, mouse_btn, clicks, x, y }=> 
                    match state {
                        EdState::Nothing=>{},
                        EdState::DraggingFrom(spos,ref dt)=>{world.drag_end(spos, v2make(x,y),dt); state=EdState::Nothing;}
                        _=>{}
                    }
                Event::MouseMotion { timestamp, window_id, which, mousestate, x, y, xrel, yrel }=>{
                    mouse_pos=v2make(x,y);
                    mouse_delta= v2make(xrel,yrel);
                }
                Event::DropFile { timestamp, window_id, filename }=>{
                    world.load_image_at(mouse_pos, filename);
                }
                _ => {}
            }
        };
        if let Some(nt)=change_type{
            if let Elem::Node(id)=pick{
                world.nodes[id].set_type(nt);
            }
        }
        change_type=None;

        rc.canvas.set_draw_color(Color::RGB(128,128,128));
        
        rc.canvas.clear();
        rc.canvas.set_draw_color(Color::RGB(192,192,192));
        
        let mut buffer:Vec<u8> =Vec::new();
        buffer.resize(256*256*4,0);

        /*
        match state {
            EdState::DraggingFrom(spos,DragType::DrawEdge(caption))=> {
                canvas.draw_line( spos, mouse_pos );
                font.draw_text(canvas, v2avr(spos,mouse_pos), color, &caption)
            },
            _=>{}
        };
        */

        for i in 0..64*64{
            buffer[i]=i as u8;
        }

        world.eval();
        world.render(&mut rc);
        let hcol= rc.hilightcol;
        match state {
            EdState::Nothing=>{
                world.render_elem(&mut rc,&pick, hcol);
            },
            EdState::DraggingFrom(spos,ref dragtype)=>{
                match dragtype {
                    DragType::DrawEdge(caption)=>{
                        if let Some(slot) = world.pick_slot(mouse_pos){
                            rc.draw_square_centred(world.slot_pos(&slot), slotsize as _, hcol)
                        }
                        let dragcol=(0,255,0);
                        rc.canvas.set_draw_color(dragcol);
                        rc.canvas.draw_line(spos,mouse_pos);
                        rc.draw_text(v2avr(spos,mouse_pos), [0,255,0,255], &caption)
                    }
                    DragType::MoveNode(id)=>{
                        world .nodes[*id].pos=v2add(world.nodes[*id].pos,mouse_delta);
                    }
                }
            }
            _=>{}
        }
        rc.draw_text((10,10),[255,255,255,255], "node graph editor");

        



        if false {
            tex.set_color_mod(255,0,255);
            tex.set_alpha_mod(255);
            rc.canvas.copy(&tex, None, Rect::new(100,100,64,64));
            tex.set_color_mod(128,255,255);
            tex.set_alpha_mod(128);
            rc.canvas.copy(&tex, None, Rect::new(200,100,64,64));
        }

        rc.canvas.present();
        //::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
        // The rest of the game loop goes here...
    }

    Ok(())
}