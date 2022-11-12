#![allow(unused_variables)]
extern crate sdl2;
extern crate image;
extern crate num_traits; 
use image::GenericImageView;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::render::WindowCanvas;
use std::ops::{Add,Sub,Mul,Div,Neg,Rem};

//use sdl2::image::LoadTexture;
type V2<T> = (T,T);
type V2i = V2<i32>;
type V2f = V2<f32>;
type V2u = V2<usize>;
type Vec2 = V2i;
fn v2make<T>(a:T,b:T)->V2<T> {(a,b)}
trait VecElem : Copy+PartialOrd+std::fmt::Debug+Mul<Output=Self> + Add<Output=Self>+ Sub<Output=Self>+ Div<Output=Self>+Rem<Output=Self>+Div<Output=Self>+Neg<Output=Self> {}
impl VecElem for i32{}
impl VecElem for f32{}
impl VecElem for isize{}
fn v2add<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0+b.0, a.1+b.1)}
fn v2sub<T:VecElem >(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0-b.0, a.1-b.1)}
fn v2div<T:VecElem>(a:V2<T>,d:T)->V2<T>{v2make(a.0/d, a.1/d)}
fn v2mod<T:VecElem>(a:V2<T>,d:T)->V2<T>{v2make(a.0%d, a.1%d)}
fn v2dot<T:VecElem>(a:V2<T>,b:V2<T>)->T{v2hsum(v2mul(a,b))}
fn v2hsum<T:VecElem>(a:V2<T>)->T{a.0+a.1}
fn v2lerp<T:VecElem>(a:V2<T>,b:V2<T>,f:T)->V2<T>{v2madd(a,v2sub(b,a),f)}
fn v2madd<T:VecElem>(a:V2<T>,b:V2<T>,f:T)->V2<T>{v2make(a.0 + b.0*f, a.1+ b.1*f)}
fn v2hmul<T:VecElem>(a:V2<T>)->T{a.0*a.1}
fn v2maxcomp<T:VecElem>(a:V2<T>)->(usize,T){if a.0 > a.1 {(0,a.0)}else{(1,a.1)}}
fn v2mul<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(a.0*b.0, a.1*b.1)}
fn minp<T:PartialOrd>(a:T,b:T)->T{if a<b{a} else {b}}
fn maxp<T:PartialOrd>(a:T,b:T)->T{if a>b{a} else {b}}
fn v2max<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(maxp(a.0,b.0), maxp(a.1,b.1))}
fn v2min<T:VecElem>(a:V2<T>,b:V2<T>)->V2<T>{v2make(minp(a.0,b.0), minp(a.1,b.1))}
fn v2scale<T:VecElem>(a:V2<T>,f:T)->V2<T>{v2make(a.0*f, a.1*f)}
fn v2abs<T:VecElem+num_traits::Signed>(a:V2<T>)->V2<T>{v2make(a.0 .abs(), a.1 .abs())}
fn v2ymx<T:VecElem>(a:V2<T>)->V2<T>{v2make(a.1,-a.0)}
fn v2myx<T:VecElem>(a:V2<T>)->V2<T>{v2make(-a.1,a.0)}
fn v2mxmy<T:VecElem>(a:V2<T>)->V2<T>{v2make(-a.0,-a.1)}

fn v2manhattan_dist<T:VecElem+num_traits::Signed>(a:V2<T>,b:V2<T>)->T {
    v2hsum(v2abs(v2sub(b,a)))
}
fn v2sqr<T:VecElem>(a:V2<T>)->T{v2dot(a,a)}
fn v2distSqr<T:VecElem>(a:V2<T>,b:V2<T>)->T{v2sqr(v2sub(b,a))}

fn v2muldiv<T:VecElem>(a:V2<T>,m:T,d:T)->V2<T>{ v2make( (a.0*m)/d, (a.0*m)/d ) }

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

enum NodeType {
    MulAdd,Add,Sub,Mul,Div,Neg,Lerp,Max,Min,Clamp,
}
impl Default for NodeType{fn default()->Self{Self::MulAdd}}
impl NodeType {
    fn name(&self)->&'static str {
        match self{
            Self::Lerp=>&"Lerp",
            Self::Add=>&"Add",
            Self::MulAdd=>&"MulAdd",
            Self::Clamp=>&"Clamp",
            Self::Mul=>&"Mul",
            Self::Div=>&"Div",
            Self::Sub=>&"Sub",
            Self::Min=>&"Min",
            Self::Max=>&"Max",
            Self::Neg=>&"Neg",
        }
    }
    fn num_input_slots(&self)->usize{
        match self {
            Self::Lerp|Self::MulAdd|Self::Clamp=>3,
            Self::Neg=>1,
            _=>2
        }
    }
}
struct Node {
    typ:NodeType,
    pos:Vec2,
    size:Vec2,
    color:[u8;4],
    text:String,
}
impl Node {
    fn rect(&self)->Rect{Rect::new(self.pos.0,self.pos.1, self.size.0 as u32,self.size.1 as u32)}
    fn centre(&self)->Vec2{v2add(self.pos, v2div(self.size,2))}
    fn centre_left(&self)->Vec2{v2add(self.pos, (0,self.size.1/2))}
    fn centre_right(&self)->Vec2{v2add(self.pos, (self.size.0,self.size.1/2))}
}
trait Contains<X>{fn contains(&self,x:X)->bool;}
impl Contains<Vec2> for Rect {
    fn contains(&self, pos:Vec2)->bool {
        pos.0 > self.left() && pos.0 < self.right() && pos.1 > self.top() && pos.1<self.bottom()
    }
}

struct Font{
    tex:std::cell::Cell<sdl2::render::Texture>,
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
    fn new(tc:&mut sdl2::render::TextureCreator<sdl2::video::WindowContext>,filename:&str, xlat_str:&str)->Self{
        let img=load_texture(filename).expect(filename);
        //let tex = tc.load_texture(filename).expect("texfont");
        let mut tex=tc.create_texture(sdl2::pixels::PixelFormatEnum::RGBA8888, sdl2::render::TextureAccess::Streaming, img.1 .0 as _,img.1 .1 as _).expect("ctex");
        let mut xlat = [0;256];
        for (i,c) in xlat_str.chars().enumerate() {
            xlat[c as usize] = i as u8;
        }

        tex.update(None, &img.0, img.1 .0 * img.2);
        tex.set_blend_mode(sdl2::render::BlendMode::Blend);


        Font {
            tex:std::cell::Cell::new(tex),
            xlat
         }
    }
    fn draw_text(&mut self, canvas:&mut sdl2::render::WindowCanvas, pos:Vec2,color:[u8;4], text:&str){
        let mut pos = pos;
        for c in text.chars(){
            let index=self.xlat[c as usize] as i32;
            let x=(index&15) as i32 *16;
            let y=(index>>4) as i32 *16;
            let rc=Rect::new(x,y, 16u32,16u32);
            let dstrc = Rect::new(pos.0, pos.1, 16u32, 16u32);
            canvas.set_blend_mode(sdl2::render::BlendMode::Blend);
            self.tex.get_mut().set_color_mod(color[0],color[1],color[2]);
            self.tex.get_mut().set_alpha_mod(color[3]);

            canvas.copy(self.tex.get_mut(), rc,dstrc );

            
            pos= v2add(pos,(16,0));
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
struct Edge {
    start:NodeID,
    end:NodeID,
}

struct World {
    nodes:Vec<Node>,
    edges:Vec<Edge>
}
impl World {
    fn pick_node(&self, pos:Vec2)->Option<usize>{
        for (i,n) in self.nodes.iter().enumerate(){
            if n.rect().contains(pos){
                return Some(i)
            }
        }
        return None
    }
    fn drag_end(&mut self, start:Vec2, end:Vec2, dt:DragType){
        println!("dragged{:?},{:?}",start,end);
        match dt{
            DragType::MoveNode(_id)=>{}
            DragType::DrawEdge()=>{
                match self.pick_node(end){
                
                    Some(ei)=>{
                        let si=self.pick_closest_node(start);
                        dbg!("dragged between",si,ei);
                        if si!=ei{
                            self.create_edge(si,ei);
                        } else {
                            println!("can't link node to self");
                        }
                    }
                    _=>{dbg!(self.create_node_at(end, &format!("Node{:?}",self.nodes.len()), [255,255,128,255], NodeType::Mul));}
                }
            }
        }
    }
    fn create_edge(&mut self, si:NodeID,ei:NodeID){
        self.edges.push(Edge{start:si,end:ei})
    }
    fn pick_closest_node(&mut self, pt:Vec2)->NodeID{
        assert!(self.nodes.len()>0);
        let mut bestdist=10000000;
        let mut bestnode=0;
        for (i,n) in self.nodes.iter().enumerate() {
            let dist =v2manhattan_dist(pt,n.centre());
            if dist < bestdist{
                bestdist=dist;
                bestnode=i;
            }
        }
        return bestnode;
    }

    fn create_node_at(&mut self, pt:Vec2, caption:&str, color:[u8;4],typ:NodeType)->NodeID{
               let ns=(64,64);
        self.nodes.push(Node{typ,pos:v2sub(pt,v2div(ns,2)), size:ns, text:caption.to_string(), color});
        self.nodes.len()-1
    }
}

fn read_string(prompt:&str)->String{
    println!("{}",prompt);
    let mut buffer = String::new();
    std::io::stdin().read_line(&mut buffer);
    buffer
}

pub fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let filename = String::from("graph.json");

    let mut window = video_subsystem
        .window("Node Graph Editor", 800, 600)
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


    let mut world = World{
        nodes:vec![Node{typ:NodeType::Mul,pos:v2make(100,150),size:v2make(64,64),color:[255,0,0,255],text:format!("node0")}, Node{typ:NodeType::Add, pos:v2make(200,150), size:v2make(64,64), color:[0,255,255,255],text:format!("node1")} ],
        edges:vec![],
    };
    let mut tc:sdl2::render::TextureCreator<sdl2::video::WindowContext> = canvas.texture_creator();
    let mut tex = tc.create_texture(None, sdl2::render::TextureAccess::Streaming, 64,64).expect("tex");
    let mut font = Font::new(&mut tc, "assets/font16.png"," _ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}+-<>@./\\#!?&*\",:;£$");

    let x:Option<sdl2::render::TextureCreator<sdl2::video::WindowContext>> = None;
    
    let mut mouse_delta = v2make(0,0);
    'running: loop {
        mouse_delta=(0,0);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }|
                Event::KeyDown {keycode: Some(Keycode::Escape),..} => break 'running,
                Event::KeyDown {keycode: Some(Keycode::A),..} => some_pos.0+=4,
                Event::KeyDown {keycode: Some(Keycode::D),..} => some_pos.0-=4,
                Event::KeyDown {keycode: Some(Keycode::W),..} => some_pos.1-=4,
                Event::KeyDown {keycode: Some(Keycode::S),..} => some_pos.1+=4,
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
                        EdState::DraggingFrom(spos,ref dt)=>{world.drag_end(spos, (x,y),*dt); state=EdState::None;}
                    }
                Event::MouseMotion { timestamp, window_id, which, mousestate, x, y, xrel, yrel }=>{
                    mouse_pos=(x,y);
                    mouse_delta= (xrel,yrel);
                }
                _ => {}
            }
        };

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

        tex.update(None, &buffer,64*4);

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

        let rgb=|x:[u8;4]|(x[0],x[1],x[2]);
        for node in world.nodes.iter() {
            canvas.set_draw_color(rgb(node.color));
            canvas.draw_rect(node.rect());
            font.draw_text(&mut canvas, node.pos, node.color, &node.typ.name());
        }
        for edge in world.edges.iter() {
            canvas.set_draw_color((192,192,192));
            let sz=8;
            let endpos=v2add(world.nodes[edge.end].centre_left(),(-sz,0));
            let startpos=v2add(world.nodes[edge.start].centre_right(),(sz,0));
            canvas.draw_line(startpos, endpos);
            let dir = (sz,0);
            draw_tri(&mut canvas, endpos, dir);
            draw_tri(&mut canvas, v2add(startpos,(-sz,0)), dir);
        }

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