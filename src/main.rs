extern crate sdl2;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use std::time::Duration;

type Point2d= (i32,i32);
fn v2add(a:Point2d,b:Point2d)->Point2d{(a.0+b.0, a.1+b.1)}
fn v2sub(a:Point2d,b:Point2d)->Point2d{(a.0-b.0, a.1-b.1)}
fn v2div(a:Point2d,d:i32)->Point2d{(a.0/d, a.1/d)}
fn v2muldiv(a:Point2d,b:Point2d,m:i32,d:i32)->Point2d{( ((a.0 as i64*m as i64)/d as i64) as i32, ((a.0 as i64*m as i64)/d as i64) as i32) }

enum EdState{
    None,
    DraggingFrom(Point2d,DragType),
}
type NodeID=usize;
#[derive(Debug,Copy,Clone)]
enum DragType{
    MoveNode(NodeID),
    DrawEdge()
}

struct Node {
    pos:Point2d,
    size:Point2d,
    color:(u8,u8,u8),
}
impl Node {
    fn rect(&self)->Rect{Rect::new(self.pos.0,self.pos.1, self.size.0 as u32,self.size.1 as u32)}
}
trait Contains<X>{fn contains(&self,x:X)->bool;}
impl Contains<Point2d> for Rect {
    fn contains(&self, pos:Point2d)->bool {
        pos.0 > self.left() && pos.0 < self.right() && pos.1 > self.top() && pos.1<self.bottom()
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
    fn pick_node(&self, pos:Point2d)->Option<usize>{
        for (i,n) in self.nodes.iter().enumerate(){
            if n.rect().contains(pos){
                return Some(i)
            }
        }
        return None
    }
    fn drag_end(&mut self, start:Point2d, end:Point2d, dt:DragType){
        println!("dragged{:?},{:?}",start,end);
        match dt{
            DragType::MoveNode(id)=>{}
            DragType::DrawEdge()=>{
                match (self.pick_node(start),self.pick_node(end)){
                
                    (Some(si),Some(ei))=>{
                        dbg!("dragged between",si,ei);
                    }
                    _=>{dbg!(self.create_node_at(end));}
                }
            }
        }
    }
    fn create_node_at(&mut self, pt:Point2d)->NodeID{
        let ns=(64,64);
        self.nodes.push(Node{pos:v2sub(pt,v2div(ns,2)), size:ns,color:(128,128,255)});
        self.nodes.len()-1
    }
    
}
pub fn main() -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("rust-sdl2 demo: Video", 800, 600)
        .position_centered()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().build().map_err(|e| e.to_string())?;

    canvas.set_draw_color(Color::RGB(255, 0, 0));
    canvas.clear();
    canvas.present();
    let mut event_pump = sdl_context.event_pump()?;

    let mut state =EdState::None;
    let mut mouse_pos:Point2d =(100,100);
    let mut some_pos=(300,100);


    let mut world = World{
        nodes:vec![Node{pos:(100,150),size:(64,64),color:(255,0,0)}, Node{pos:(200,150), size:(64,64), color:(0,255,255)} ],
        edges:vec![],
    };
    let mut mouse_delta:Point2d = (0,0);
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
        
        canvas.draw_line((100,100),(200,200));
        canvas.draw_rect(Rect::new(200,100,50,50));
        let mut tc= canvas.texture_creator();
        let mut tex = tc.create_texture(None, sdl2::render::TextureAccess::Streaming, 64,64).expect("tex");
        let mut buffer:Vec<u8> =Vec::new();
        buffer.resize(256*256*4,0);
        match state {
            EdState::None=>{}
            EdState::DraggingFrom(spos,dt)=> {canvas.draw_line( spos, mouse_pos );},
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

        for node in world.nodes.iter() {
            canvas.set_draw_color(node.color);
            canvas.draw_rect(node.rect());
        }

        tex.set_color_mod(255,0,255);
        tex.set_alpha_mod(255);
        canvas.copy(&tex, None, Rect::new(100,100,64,64));
        tex.set_color_mod(128,255,255);
        tex.set_alpha_mod(128);
        canvas.copy(&tex, None, Rect::new(200,100,64,64));

        canvas.present();
        //::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 30));
        // The rest of the game loop goes here...
    }

    Ok(())
}