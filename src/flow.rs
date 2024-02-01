use std::io::Read;
use std::path::Path;

pub const X_CELLS: usize = 400;
pub const X_START: f32 = -0.5;
pub const X_END: f32 = 7.5;
pub const X_STEP: f32 = (X_END - X_START) / X_CELLS as f32;
pub const Y_CELLS: usize = 50;
pub const Y_START: f32 = -0.5;
pub const Y_END: f32 = 0.5;
pub const Y_STEP: f32 = (Y_END - Y_START) / Y_CELLS as f32;
pub const T_CELLS: usize = 1001;
pub const T_START: f32 = 15.0;
pub const T_END: f32 = 23.0;
pub const T_STEP: f32 = (T_END - T_START) / T_CELLS as f32;
pub const FRAME_SIZE: usize = X_CELLS * Y_CELLS;
pub const TOTAL_ELEMS: usize = FRAME_SIZE * T_CELLS;

pub struct Field {
    pub max_velocity: f32,
    data: Vec<Vec2>,
}

impl Field {
    #[cfg(target_endian = "little")]
    pub fn read(path: &Path) -> anyhow::Result<Field> {
        let mut file = std::fs::File::open(path)?;
        let mut data = vec![Vec2::ZERO; TOTAL_ELEMS];
        let raw = bytemuck::cast_slice_mut(data.as_mut_slice());
        file.read_exact(raw)?;

        let max_velocity = data
            .iter()
            .copied()
            .map(|v| v.norm())
            .max_by(f32::total_cmp)
            .unwrap();
        Ok(Field { max_velocity, data })
    }

    pub fn frame(&self, t: usize) -> Frame<'_> {
        Frame(&self.data[t * FRAME_SIZE..(t + 1) * FRAME_SIZE])
    }
}

pub struct Frame<'a>(&'a [Vec2]);

impl<'a> Frame<'a> {
    pub fn get(&self, x: usize, y: usize) -> Vec2 {
        debug_assert!(x < X_CELLS);
        debug_assert!(y < Y_CELLS);
        self.0[y * X_CELLS + x]
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vec2> {
        self.0.iter()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn norm(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Add<Vec2> for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct Pos2 {
    pub x: f32,
    pub y: f32,
}

impl Pos2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };
}

impl std::ops::Add<Vec2> for Pos2 {
    type Output = Pos2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub<Pos2> for Pos2 {
    type Output = Vec2;

    fn sub(self, rhs: Pos2) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::AddAssign<Vec2> for Pos2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}
