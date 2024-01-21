use std::{
    f32,
};
use minifb::*;
use vek::*;

use nalgebra::{
    Rotation3, 
    Unit, 
    Vector3
};

const W: usize = 100;
const H: usize = 100;

trait Helper {
    fn rotate_around(&self, axis: Vec3<f32>, angle: f32) -> Vec3<f32>;
    fn toVector3(&self) -> Vector3<f32>;
    fn rotate_vector(&self, axis: Vec3<f32>, angle: f32) -> Vec3<f32>;
}

impl Helper for Vec3<f32> {
    fn toVector3(&self) -> Vector3<f32> {
        return Vector3::new(self.x, self.y, self.z)
    }
    fn rotate_around(&self, axis: Vec3<f32>, angle: f32) -> Vec3<f32> {
        let axis = Unit::new_normalize(axis.toVector3());
        let rot = Rotation3::from_axis_angle(&axis, angle);
        let prod = rot * self.toVector3();
        return Vec3::new(prod.x,prod.y,prod.z)
    }
    fn rotate_vector(&self, axis: Vec3<f32>, angle: f32) -> Vec3<f32> {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        let axisn = axis.normalized();
    
        let term1 = self * cos_angle;
        let term2 = axisn.cross(*self) * sin_angle;
        let term3 = axisn * axisn.dot(*self) * (1.0 - cos_angle);
    
        return term1 + term2 + term3;
    }
}
trait ColorHelper {
    fn u32color(&self) -> u32;
}
impl ColorHelper for Rgba<u8> {
    fn u32color(&self) -> u32 {
        let (r, g, b) = (self.r as u32, self.g as u32, self.b as u32);
        return (r << 16) | (g << 8) | b;
    }
}
trait Shape {
    fn colliding(&self, r: Ray, dist: &mut f32, normal: &mut Ray) -> bool;
}

#[derive(Copy, Clone, Debug)]
struct Triangle {
    a: Vec3<f32>,
    b: Vec3<f32>,
    c: Vec3<f32>,
}
impl Triangle {
    fn new(a: Vec3<f32>, b: Vec3<f32>, c: Vec3<f32>) -> Triangle {
        return Triangle {a, b, c}
    }
}
#[derive(Copy, Clone, Debug)]
struct Sphere {
    center: Vec3<f32>, 
    rad: f32
}
impl Sphere {
    fn new(center: Vec3<f32>, rad: f32) -> Sphere {
        return Sphere {center, rad}
    }
}
impl Shape for Sphere {
    fn colliding(&self, r: Ray, dist: &mut f32, normal: &mut Ray) -> bool {
        let v = self.center - r.o;
        let t = v.dot(r.dir);
        
        if t < 0.0 {
            return false;
        }
    
        let v2 = v.dot(v);
        let d = self.rad * self.rad - v2 + t * t;
    
        if d >= 0.0 {
    
            let bruh = r.o + t * r.dir;
    
            *dist = bruh.distance(r.o);


            let dir = (bruh - &self.center).normalized();

            *normal = Ray::new(bruh, dir); 
    
            return true;
        }
    
        return false;
    }
}

#[derive(Copy, Clone, Debug)]
struct Plane {
    a: f32,
    b: f32,
    c: f32, 
    k: f32,
}
impl Plane {
    fn new(a: f32, b: f32, c: f32, k: f32) -> Plane {
        return Plane {a, b, c, k}
    }
}
#[derive(Copy, Clone, Debug)]
struct Camera {
    pos: Vec3<f32>,
    pitch: f32,
    yaw:  f32,
    fov: f32,
    screen: Vec2<u32>,
}
impl Camera {
    fn new(pos: Vec3<f32>, pitch: f32, yaw: f32, fov: f32, screen: Vec2<u32>) -> Camera {
        return Camera {pos, pitch, yaw, fov, screen}
    }
    fn normal(&self) -> Vec3<f32> {
        let axis_y = Vec3::new(1.0,0.0,0.0);
        let axis_p = Vec3::new(0.0,1.0,0.0);

        let f = Vec3::new(0.0,0.0,1.0);

        let r1 = f.rotate_vector(axis_y, self.yaw);
        let r2 = r1.rotate_vector(axis_p, self.pitch);

        return r2;
    }
    fn yawtate(&mut self, angle: f32) {
        self.yaw += angle;
    }
    fn pitchtate(&mut self, angle: f32) {
        self.pitch += angle;
    }
    fn refov(&mut self, fov: f32) {
        self.fov = fov;
    }
}
#[derive(Copy, Clone, Debug)]
struct Light {
    pos: Vec3<f32>,
    intensity: f32,
}
impl Light {
    fn new(pos: Vec3<f32>, intensity: f32) -> Light {
        return Light {pos, intensity}
    }
    fn colliding(&self, _r: Ray, _dist: &mut f32) -> bool {
        //calculate plane
        return false;
    }
}
#[derive(Clone)]
struct Scene<'a> {
    light: Light,
    objs: Vec<&'a dyn Shape>,
}
impl<'a> Scene<'a> {
    fn new(light: Light, objs: Vec<&dyn Shape>) -> Scene {
        return Scene {light, objs};
    }
}
#[derive(Copy, Clone, Debug)]
struct Ray {
    o: Vec3<f32>,
    dir: Vec3<f32>,
}
impl Ray {
    fn new(o: Vec3<f32>, dir: Vec3<f32>) -> Ray {
        return Ray {o, dir}
    }
    fn empty() -> Ray {
        let o = Vec3::new(0.0,0.0,0.0);
        let dir = Vec3::new(0.0,0.0,0.0);
        return Ray {
            o,
            dir,
        }
    }
}

fn cast_rays(c: Camera, scene: Scene) -> Vec<Rgba<u8>> {
    let k = c.screen.x as f32;
    let m = c.screen.y as f32;

    let n = c.normal(); //view plane normal 

    let size = c.fov;
    let zr = Vec3::new(1.0,0.0,0.0);
    let v = n.rotate_around(zr, 1.5707963268);
    let b = v.cross(n);
            
    let gx = (k / 2.0).floor() * size;
    let gy = (m / 2.0).floor() * size;

    let qx = 2.0 * gx / (k - 1.0) * b;
    let qy = 2.0 * gy / (m - 1.0) * v;

    let p1m = n - gx * b - gy * v;

    let mut buf = Vec::new();
    for x in 1..(k as u32) + 1 {
        for y in 1..(m as u32) + 1 {
            
            let xf = x as f32;
            let yf = y as f32;
            let p = p1m + qx * (xf - 1.0) + qy * (yf - 1.0);
            let r = (c.pos - p).normalized();
            
            let mut red = Rgba::new(255,0,0,0);
            let mut blue = Rgba::new(0,0,255,0);
            let black = Rgba::new(0,0,0,0);
            let primray = Ray::new(c.pos, r); //primary ray

            let mut dist = 0.0;
            let mut bright = 0.0;
            if intersect(primray, 99999.9, scene.clone(), &mut dist, &mut bright) {
                if y as f32 > k / 2.0 {
                    blue.b = ((255.0 - (blue.b as f32 / dist.powf(2.0))) * bright) as u8;
                    buf.push(blue);
                } else {
                    red.r = ((255.0 - (red.r as f32 / dist.powf(2.0))) * bright) as u8;
                    buf.push(red);
                }
            } else {
                buf.push(black);
            }
        }
    }
    return buf;
}
fn intersect(r: Ray, max_dist: f32, scene: Scene, dist: &mut f32, brightness: &mut f32) -> bool {
    let zero = Vec3::new(0.0,0.0,0.0);
    let mut min_dist = max_dist;
    let mut phit = Vec3::new(0.0,0.0,0.0);
    let mut nhit = Ray::new(zero, zero);
    for obj in scene.clone().objs {
        let mut n = Ray::new(zero, zero);
        let mut c_dist = 0.0;
        let _c = obj.colliding(r, &mut c_dist, &mut n);
        if c_dist < min_dist {
            min_dist = c_dist;
            nhit = n;
            phit = r.o + r.dir * c_dist;
        }
    }
    let mut shadow = true;
    let d = phit.distance(scene.light.pos);

    if min_dist < max_dist {
        *dist = min_dist;
        let idir = (scene.light.pos - phit).normalized();
        let ilray = Ray::new(phit, idir);
        
        for obj in scene.clone().objs {
            let mut _d = 0.0;
            let mut _n = Ray::empty();

            if obj.colliding(r, &mut _d, &mut _n) {
                shadow = false;
                break;
            }
        }
        if shadow {
            *brightness = 0.25;
        } else {
            *brightness = 1.0;
        }

        return true;
    }
    return false;
}
fn render_scene(cam: Camera, scene: Scene) -> Vec<u32> {
    let mut u32buff = Vec::new();
    for i in cast_rays(cam, scene) {
        u32buff.push(i.u32color());
    }
    return u32buff;
}
fn main() {
    let p4 = Vec3::new(0.0, 0.0, 42.0);
    let s = Sphere::new(p4, 40.0);

    let p5 = Vec3::new(0.0, 0.0, -40.0);
    let s2 = Sphere::new(p5, 2.0);

    let pos = Vec3::new(0.0, 0.0, 0.0);
    let screen = Vec2::new(W as u32, H as u32);
    let mut cam = Camera::new(pos, 0.0, 0.0, 1.0, screen);

    let lr = Vec3::new(0.0,12.0,0.0);
    let l = Light::new(lr, 0.0);

    let scene = Scene::new(l, vec!(&s, &s2));

    let mut count = 0;

    let mut opts = WindowOptions::default();
    opts.scale = minifb::Scale::X8;
    let mut buf = vec![0; H * W];
    let mut win = Window::new("Canvas", W, H, opts).unwrap();


    while win.is_open() {
        count += 1;
            if count > 1000 {
            let (_width, _height) = win.get_size();

            if win.is_key_down(Key::A) {
                cam.yawtate(-0.01);
            }
            if win.is_key_down(Key::D) {
                cam.yawtate(0.01);
            }
            if win.is_key_down(Key::W) {
                cam.pitchtate(0.01);
            }
            if win.is_key_down(Key::S) {
                cam.pitchtate(-0.01);
            }
            if win.is_key_down(Key::Up) {
                cam.pos += cam.normal()/10.0;
            }
            if win.is_key_down(Key::Down) {
                cam.pos -= cam.normal()/10.0;
            }
            if win.is_key_pressed(Key::X, KeyRepeat::No) {
                cam.yawtate(3.1415926536);
            }
            if win.is_key_pressed(Key::R, KeyRepeat::No) {
                cam.pos = Vec3::new(0.0,0.0,0.0);
                cam.pitch = 0.0;
                cam.yaw = 0.0;
                cam.fov = 1.0;
            }
            if win.is_key_pressed(Key::Space, KeyRepeat::No) {
                println!("{}, ({}, {})", s.center, cam.pos, cam.normal());
            }
            if win.is_key_down(Key::Equal) {
                if cam.fov > 0.0 {
                    cam.fov -= 0.05;
                }
            }
            if win.is_key_down(Key::Minus) {

                cam.fov += 0.05;
            }
            
            buf = render_scene(cam, scene.clone());

            win.update_with_buffer(&buf, W, H).unwrap();
        }
    }
}