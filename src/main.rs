#![allow(unused_variables)]
#![allow(unused_variables)]
extern crate sdl2;
extern crate image;
extern crate num_traits;
extern crate serde;
extern crate serde_derive;
use std::borrow::BorrowMut;
use std::rc::Rc;
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
use sdl2::render::{WindowCanvas, TextureCreator,Texture};

fn from<A,B:From<A>>(src:A)->B{B::from(src)}

macro_rules! assert_less{
    ($a:expr,$b:expr)=>{if !($a<$b){dbg!(&$a,"<",&$b);assert!(false);}}
}


enum EdState{
    None,
    DraggingFrom(Vec2,DragType),
}
type NodeID=usize;
#[derive(Debug,Copy,Clone)]
enum DragType{
    MoveNode(NodeID),
    DrawEdge()
}
#[derive(Clone,Debug)]
enum NodeType{
    Function(fnodes::FNodeType),
    Value(fnodes::SlotTypeVal),
}
use NodeType::*;

impl NodeType {
    fn name(&self)->&'static str {match self{
        Function(x)=>x.name(),
        Value(x)=>x.name()
    }}
    fn num_input_slots(&self)->usize{match self {
        Function(x)=>x.num_input_slots(),
        Value(x)=>0,
    }}
    fn num_outputs(&self)->usize{match self{
        Function(x)=>x.num_outputs(),
        Value(x)=>1,
    }}
    fn num_slots(&self)->usize{self.num_input_slots()+self.num_outputs()}
}

type OptRc<T> = Option<Rc<T>>;
fn somerc<T>(x:T)->OptRc<T>{Some(Rc::new(x))}

#[derive(Clone,Debug)]
struct Node {
    typ:NodeType,
    pos:Vec2,
    size:Vec2,
    color:[u8;4],
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


impl Node {
    fn rect(&self)->Rect{Rect::new(self.pos.0,self.pos.1, self.size.0 as u32,self.size.1 as u32)}
    fn centre(&self)->Vec2{v2add(self.pos, v2div(self.size,2))}
    fn centre_left(&self)->Vec2{v2add(self.pos, (0,self.size.1/2))}
    fn centre_right(&self)->Vec2{v2add(self.pos, (self.size.0,self.size.1/2))}

    fn render(&self, (wc,tc):(&mut WindowCanvas,&mut TextureCreator<WindowContext>),f:&mut Font ){
        let pos = v2add(self.pos,v2make(8,16));    // some margin, todo

        fn draw_float(val:f32, wc:&mut WindowCanvas,pos:V2i,font:&mut Font){
            font.draw_text(wc,pos,[255,255,255,255],&format!("{:?}",val));
        }
        if let Some(x) = &self.cached {
            let x:&SlotTypeVal =&*x;
            match x {
                SlotTypeVal::FloatDefaultOne(x)=>draw_float(from(x),wc,pos,f),
                SlotTypeVal::FloatDefaultZero(x)=>draw_float(from(x),wc,pos,f),
                SlotTypeVal::ImageRGBA(b)=>{
                    
                    let img2 = b.map_channels(|x|(x.clamp(0.0,1.0)*255.0) as u8);
                    self.bitmap.update(tc,&img2);
                    let tex = self.bitmap.data.borrow();
                    
                    wc.copy((&*tex).as_ref().unwrap(),None,Rect::new(pos.0,pos.1, 64,64));
                }
                SlotTypeVal::ImageLuma(x)=>{
                    
                }

                SlotTypeVal::Empty(_)=>{

                }
            }
        }
    }
}

trait Contains<X>{fn contains(&self,x:X)->bool;}
impl Contains<Vec2> for Rect {
    fn contains(&self, pos:Vec2)->bool {
        pos.0 > self.left() && pos.0 < self.right() && pos.1 > self.top() && pos.1<self.bottom()
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
    fn draw_text(&mut self, canvas:&mut WindowCanvas, pos:Vec2,color:[u8;4], text:&str){
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
fn draw_tri(canvas:&mut WindowCanvas,pos:Vec2, dir:Vec2){
    let perp=(dir.1,-dir.0);
    let vertices:[Vec2;3]=[v2add(pos,dir), v2add(pos,perp),v2sub(pos,perp)];
    for (s,e) in [(0,1),(1,2),(2,0)]{
        canvas.draw_line(vertices[s],vertices[e]);
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
    fn remove_node(&mut self, id:usize){
        self.nodes.remove(id);
        self.edges.retain(|e|!(e.start.node==id || e.end.node==id));
        for e in self.edges.iter_mut(){
            if e.start.node>=id {e.start.node-=1;}
            if e.end.node>=id {e.end.node-=1;}

        }
    }
    fn pick_node(&self, pos:Vec2)->Option<usize>{
        for (i,n) in self.nodes.iter().enumerate(){
            if n.rect().contains(pos){
                return Some(i)
            }
        }
        return None
    }
    fn slot_pos(&self,sa:&SlotAddr)->Vec2{
        let node=&self.nodes[sa.node];
        let inpn=node.typ.num_input_slots();
        let (dx,syf,ns)=if sa.slot <inpn {
            (0,sa.slot,inpn)
        } else {
            (node.size.0, (sa.slot-inpn),node.typ.num_outputs())
        };
        v2make(node.pos.0 + dx, node.pos.1 + ((node.size.1 * (syf as i32*2+1))/(ns as i32*2)))
    }
    fn try_create_edge(&mut self, start:Vec2,end:Vec2)->Option<(SlotAddr,SlotAddr)>{
        match self.pick_node(end){                
            Some(ei)=>{
                let (eslot,_)=self.pick_closest_node_slot(end);
                let (sslot,_)=self.pick_closest_node_slot(start);
                dbg!("dragged between",sslot,eslot);
                if sslot.node!=eslot.node{
                    self.edges.retain(|e|e.end!=eslot);
                    self.create_edge(sslot,eslot);
                } else {
                    println!("can't link node to self");
                }
                Some((sslot,eslot))
            }
            _=>{None}
        }
    }

    fn drag_end(&mut self, start:Vec2, end:Vec2, dt:DragType){
        println!("dragged{:?},{:?}",start,end);
        match dt{
            DragType::MoveNode(_id)=>{}
            DragType::DrawEdge()=>{
                if let Some(_)=self.try_create_edge(start,end){
                    // either craete edge to existing node..

                } else{

                    // or create a new node to connect to
                    let node_id=self.create_node_at(v2add(end,v2make(32,0)),&format!("Node{:?}",self.nodes.len()), [255,255,128,255], Function(FNodeType::img_add));

                    if let Some((si,ei))=self.try_create_edge(start,end) {
                        let cpos = self.nodes[node_id].centre();
                        let spos=self.slot_pos(&ei);
                        let diff = v2sub(end,spos);
                        v2acc(&mut self.nodes[node_id].pos, diff);
                    }

                }
            }
        }
    }
    fn create_edge(&mut self, si:SlotAddr,ei:SlotAddr){
        self.edges.push(Edge{start:si,end:ei})
    }
    fn pick_closest_node_slot(&mut self, pt:Vec2)->(SlotAddr,Vec2){
        assert!(self.nodes.len()>0);
        let mut bestdist=10000000;
        let mut bestslot=SlotAddr{node:0,slot:0};
        let mut bestpos=self.slot_pos(&bestslot);
        for (i,n) in self.nodes.iter().enumerate() {
            for j in 0..n.typ.num_slots(){
                let pos = self.slot_pos(&SlotAddr{node:i,slot:j});
                let dist =v2manhattan_dist(pt,pos);
                if dist < bestdist{
                    bestdist=dist;
                    bestslot=SlotAddr{node:i,slot:j};
                    bestpos=pos;
                }
            }
        }
        return (bestslot,bestpos);
    }

    fn create_node_at(&mut self, pt:Vec2, caption:&str, color:[u8;4],typ:NodeType)->NodeID{
               let ns=(128,128);
        let newnode=Node{
            cached:None,
            typ,
            pos:v2sub(pt,v2div(ns,2)),
            size:ns,
            text:caption.to_string(),
            color,
            bitmap:default(),
        };
        self.nodes.push(newnode);
        self.nodes.len()-1
    }

    fn eval(&mut self) {
        let empty=somerc(Empty());
        // brute force
        // prepare input value buffer.
        let mut node_inputs:Vec<Vec<OptRc<SlotTypeVal>>> = Vec::new();
        for node in self.nodes.iter_mut(){
            node_inputs.push( vec![None; node.typ.num_input_slots()]);
        }
        
        // gather inputs per node,through edges
        for (ei,edge) in self.edges.iter().enumerate(){
            
            let srcnode=&self.nodes[edge.start.node];
            let dstnode=&self.nodes[edge.end.node];
            assert!(srcnode.typ.num_outputs()==1);
            
            //assert_less!(edge.end.slot, dstnode.typ.num_input_slots());
            
            if let Some(val)=&srcnode.cached{
                node_inputs[ edge.end.node ] [edge.end.slot] = Some(Rc::clone(val));
            }

        }
        let empty= SlotTypeVal::Empty(Empty());
        for (id,node) in self.nodes.iter_mut().enumerate(){
            node.cached = match &node.typ {
                Value(x)=>{somerc(x.clone())},   // todo - this should be Rc or 
                Function(f)=>Self::eval_node(f, &node_inputs[id], &empty)
            }
        }
    }

    fn eval_node<'a>(f:&FNodeType, inputs:&'a [OptRc<SlotTypeVal>], empty:&'a SlotTypeVal)->Option<Rc<SlotTypeVal>> {

        let mut any_empty=false;
        let inputs_xlat:Vec<SlotTypeRef<'a>> =inputs.iter().map(
            |sv:&OptRc<SlotTypeVal>|->SlotTypeRef<'a>{
            match sv{
                None=>{any_empty=true;SlotTypeRef::from(empty)},
                Some(x)=>SlotTypeRef::from(&**x)
            }
        }).collect::<Vec<_>>();
        //dbg!(&inputs_xlat);
        if any_empty{None}
        else{somerc(f.eval(&inputs_xlat))}
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
                    typ:Value(SlotTypeVal::FloatDefaultOne(from(0.75))),
                    pos:v2make(250,150), size:v2make(64,64), 
                    color:[0,255,255,255],
                    text:format!("node2"),
                    bitmap:default(),
                },
                Node{cached:None,
                    typ:Value(SlotTypeVal::ImageRGBA(Image2D::from_fn (
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
                    typ:Value(SlotTypeVal::ImageRGBA(Image2D::from_fn(
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

    fn render(&self, (canvas,tc):(&mut WindowCanvas,&mut TextureCreator<WindowContext>), font:&mut Font){
        let rgb=|x:[u8;4]|(x[0],x[1],x[2]);
        let csize=8;
        for (ni,node) in self.nodes.iter().enumerate() {
            canvas.set_draw_color(rgb(node.color));
            canvas.draw_rect(node.rect());
            font.draw_text(canvas, node.pos, node.color, &node.typ.name());
            for i in 0..node.typ.num_slots(){
                draw_tri(canvas, self.slot_pos(&SlotAddr{slot:i,node:ni}), v2make(csize,0))
            }
            let client_tl =v2add(node.pos, v2make(8,16));// clieht rect of node, todo
            // node expand/contract, todo
            node.render((canvas,tc), font);
        }
        for edge in self.edges.iter() {
            canvas.set_draw_color((192,192,192));
            
            let endpos=v2add(self.slot_pos(&edge.end),(-csize,0));
            let startpos=v2add(self.slot_pos(&edge.start),(csize,0));
            canvas.draw_line(startpos, endpos);
            let dir = (csize,0);
            draw_tri(canvas, endpos, dir);
            draw_tri(canvas, v2add(startpos,(-csize,0)), dir);
        }    
    }
}


fn read_string(prompt:&str)->String{
    println!("{}",prompt);
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer);
    buffer
}

fn cellnone<T>()->std::cell::RefCell<Option<T>>{from(None)}

pub fn main() -> Result<(), String> {
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

    let mut state =EdState::None;
    let mut mouse_pos =v2make(100,100);
    let mut some_pos=v2make(300,100);


    let mut world = World::example_scene();
    let mut tc:sdl2::render::TextureCreator<WindowContext> = canvas.texture_creator();
    let mut tex = tc.create_texture(None, sdl2::render::TextureAccess::Streaming, 64,64).expect("tex");
    let mut font = Font::new(&mut tc, "assets/font16.png",
        " ¬ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}+-<>@./\\#!?&*\",:;£$___");

    let x:Option<sdl2::render::TextureCreator<WindowContext>> = None;
    
    let mut mouse_delta = v2make(0,0);
    let mut node_type = None;
    'running: loop {
        mouse_delta=(0,0);
        let pick_node=world.pick_node(mouse_pos);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }|
                Event::KeyDown {keycode: Some(Keycode::Escape),..} => break 'running,
                Event::KeyDown {keycode: Some(Keycode::A),..} => some_pos.0+=4,
                Event::KeyDown {keycode: Some(Keycode::D),..} => some_pos.0-=4,
                Event::KeyDown {keycode: Some(Keycode::W),..} => some_pos.1-=4,
                Event::KeyDown {keycode: Some(Keycode::S),..} => some_pos.1+=4,
                Event::KeyDown {keycode: Some(Keycode::Backspace),..} => {
                    if let Some(node)=world.pick_node(mouse_pos){
                        world.remove_node(node);
                    }
                }
                Event::KeyDown {keycode: Some(Keycode::Num1),..} => node_type=Some(Function(FNodeType::img_add)), 
/*                Event::KeyDown {keycode: Some(Keycode::Num2),..} => node_type=Some(NodeType::img_mul), 
                Event::KeyDown {keycode: Some(Keycode::Num3),..} => node_type=Some(NodeType::img_sin), 
                Event::KeyDown {keycode: Some(Keycode::Num4),..} => node_type=Some(NodeType::img_fractal), 
                Event::KeyDown {keycode: Some(Keycode::Num5),..} => node_type=Some(NodeType::img_add_mul_const), 
                Event::KeyDown {keycode: Some(Keycode::Num6),..} => node_type=Some(NodeType::img_warp), 
                Event::KeyDown {keycode: Some(Keycode::Num7),..} => node_type=Some(NodeType::img_blend), 
                Event::KeyDown {keycode: Some(Keycode::Num8),..} => node_type=Some(NodeType::img_min), 
                Event::KeyDown {keycode: Some(Keycode::Num9),..} => node_type=Some(NodeType::img_max), 
*/                
                Event::MouseButtonDown { timestamp, window_id, which, mouse_btn, clicks, x, y }=>{
                        state=EdState::DraggingFrom(
                            (x,y),
                            if let Some(x)=world.pick_node((x,y)){
                                DragType::MoveNode(x)
                            }else {
                                DragType::DrawEdge()
                            }
                        );
                    }
                Event::MouseButtonUp { timestamp, window_id, which, mouse_btn, clicks, x, y }=> 
                    match state {
                        EdState::None=>{},
                        EdState::DraggingFrom(spos,ref dt)=>{world.drag_end(spos, v2make(x,y),*dt); state=EdState::None;}
                    }
                Event::MouseMotion { timestamp, window_id, which, mousestate, x, y, xrel, yrel }=>{
                    mouse_pos=v2make(x,y);
                    mouse_delta= v2make(xrel,yrel);
                }
                _ => {}
            }
        };
        if let Some(nt)=node_type{
            if let Some(pick)=pick_node{
                let node = &mut world.nodes[pick];
                node.typ=nt;
            }
        }
        node_type=None;

        canvas.set_draw_color(Color::RGB(128,128,128));
        
        canvas.clear();
        canvas.set_draw_color(Color::RGB(192,192,192));
        
        
        
        let mut buffer:Vec<u8> =Vec::new();
        buffer.resize(256*256*4,0);
        match state {
            
            EdState::DraggingFrom(spos,DragType::DrawEdge())=> {canvas.draw_line( spos, mouse_pos );},
            _=>{}
        };
        

        for i in 0..64*64{
            buffer[i]=i as u8;
        }


        match state {
            EdState::None=>{},
            EdState::DraggingFrom(spos,ref dragtype)=>{
                match dragtype {
                    DragType::DrawEdge()=>{
                        canvas.set_draw_color((0,255,0));
                        canvas.draw_line(spos,mouse_pos);
                    }
                    DragType::MoveNode(id)=>{
                        world .nodes[*id].pos=v2add(world.nodes[*id].pos,mouse_delta);
                    }
                }
            }
        }
        font.draw_text(&mut canvas, (10,10),[255,255,255,255], "node graph editor");

        

        world.eval();
        world.render((&mut canvas,&mut tc),&mut font);


        if false {
            tex.set_color_mod(255,0,255);
            tex.set_alpha_mod(255);
            canvas.copy(&tex, None, Rect::new(100,100,64,64));
            tex.set_color_mod(128,255,255);
            tex.set_alpha_mod(128);
            canvas.copy(&tex, None, Rect::new(200,100,64,64));
        }

        canvas.present();
        //::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
        // The rest of the game loop goes here...
    }

    Ok(())
}