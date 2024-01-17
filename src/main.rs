use std::{
    f32,
    io,
    ops::Mul,
    mem,
};
use minifb::*;
use vek::*;

use nalgebra::{
    Rotation3, 
    Unit, 
    Vector3
};

const W: usize = 70;
const H: usize = 70;

trait Helper {
    fn rotate_around(&self, axis: Vec3<f32>, angle: f32) -> Vec3<f32>;
    fn toVector3(&self) -> Vector3<f32>;
}

impl Helper for Vec3<f32> {
    fn toVector3(&self) -> Vector3<f32> {
        return Vector3::new(self.x, self.y, self.z)
    }
    fn rotate_around(&self, axis: Vec3<f32>, angle: f32) -> Vec3<f32> {
        let axis = Unit::new_normalize(Vector3::new(axis.x, axis.y, axis.z));
        let rot = Rotation3::from_axis_angle(&axis, angle);
        let prod = rot * self.toVector3();
        return Vec3::new(prod.x,prod.y,prod.z)
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
trait Collision {
    fn colliding(&self, p: Vec3<f32>, normal: &mut Vec3<f32>) -> bool;
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
    fn repos(&mut self, center: Vec3<f32>) {
        self.center = center;
    }
}
impl Collision for Sphere {
    fn colliding(&self, p: Vec3<f32>, normal: &mut Vec3<f32>) -> bool {
        if p.distance(self.center) <= self.rad {
            let n = (p - self.center).normalized();
            *normal = n;
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
        let y = Vec3::new(1.0,0.0,0.0);
        let p = Vec3::new(0.0,1.0,0.0);
        let mut f = Vec3::new(0.0,0.0,1.0);
        let mut r1 = f.rotate_around(y, self.yaw);
        return r1.rotate_around(p, -self.pitch);
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
struct Light {
    pos: Vec3<f32>,
    intensity: f32,
}
impl Light {
    fn new(pos: Vec3<f32>, intensity: f32) -> Light {
        return Light {pos, intensity}
    }
}
struct Ray {

    dir: Vec3<f32>,
}
impl Ray {
    fn new(dir: Vec3<f32>) -> Ray {
        return Ray {dir}
    }
}

fn cast_rays(c: Camera, obj: Sphere) -> Vec<Rgba<u8>> {
    let k = c.screen.x as f32;
    let m = c.screen.y as f32;

    let n = c.normal(); //view plane normal 

    let size = 1.0;
    let zr = Vec3::new(1.0,0.0,0.0);
    let v = n.rotate_around(zr, 1.5707963268);
    let b = v.cross(n);
            
    let gx = (k / 2.0).floor() * size;
    let gy = (m / 2.0).floor() * size;

    let qx = 2.0 * gx / (k - 1.0) * b * size;
    let qy = 2.0 * gy / (m - 1.0) * v * size;

    let p1m = n * c.fov - gx * b - gy * v;

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
            let primray = Ray::new(r); //primary ray

            let mut dist = 0.0;
            if intersect(c.pos, primray, 25.0, obj, &mut dist) {
                if c.pos.z > obj.center.z {
                    blue.b -= (blue.b as f32 / dist.powf(2.0)) as u8;
                    buf.push(blue);
                } else {
                    red.r -= (red.r as f32 / dist.powf(2.0)) as u8;
                    buf.push(red);
                }
            } else {
                buf.push(black);
            }

            

        }
    }
    return buf;
}
fn intersect(p: Vec3<f32>, r: Ray, max_dist: f32, sphere: Sphere, dist: &mut f32) -> bool {
    let mut c = 0.0;
    let mut l = p;
    while c < max_dist {

        let mut normal = Vec3::new(0.0,0.0,0.0);
        if sphere.colliding(l, &mut normal) {
            *dist = c;
            return true;
        }
        l += r.dir * 0.2;
        c += 0.2;
    }
    return false;
}
fn render_bruh(cam: Camera, s: Sphere) -> Vec<u32> {
    let mut u32buff = Vec::new();
    for i in cast_rays(cam, s) {
        u32buff.push(i.u32color());
    }
    return u32buff;
}
fn main() {
    
    /*let p1 = Vec3::new(-1.0, 0.0, 2.0);
    let p2 = Vec3::new(1.0, 0.0, 2.0);
    let p3 = Vec3::new(0.0, 1.0, 3.0);

    let t = Triangle::new(p1, p2, p3);*/

    let p4 = Vec3::new(0.0, 0.0, 30.0);
    let mut s = Sphere::new(p4, 20.0);

    let pos = Vec3::new(0.0, 0.0, 0.0);
    let screen = Vec2::new(W as u32, H as u32);
    let mut cam = Camera::new(pos, 0.0, 0.0, 1.0, screen);

    let pos2 = Vec3::new(0.0,3.0,0.0);
    let l = Light::new(pos2, 1.0);

    let mut count = 0;

    let mut opts = WindowOptions::default();
    opts.scale = minifb::Scale::X8;
    let mut buf = render_bruh(cam, s);
    let mut win = Window::new("Canvas", W, H, opts).unwrap();

    while win.is_open() {
        count += 1;
            if count > 1000 {
            let (width, height) = win.get_size();

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
                cam.pos == Vec3::new(0.0,0.0,0.0);
            }
            if win.is_key_pressed(Key::Space, KeyRepeat::No) {
                println!("{}, ({}, {})", s.center, cam.pos, cam.normal());
            }
            
            buf = render_bruh(cam, s);

            win.update_with_buffer(&buf, W, H).unwrap();
        }
    }
}