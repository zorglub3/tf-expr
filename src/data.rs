use std::marker::PhantomData;
use tensorflow::DataType;
use tensorflow::Shape;
use tensorflow::TensorType;

pub trait Data<const RANK: usize>: Clone {
    type Element: TensorType;

    fn rank(&self) -> usize {
        RANK
    }

    fn data_type(&self) -> DataType;
    fn shape(&self) -> Shape;
    fn dimensions(&self) -> Vec<u64>;
}

pub trait ScalarData: Data<0> + Clone {}

impl ScalarData for FloatData<0> {}

#[derive(PartialEq, Clone)]
pub struct NoData {
    phantom: PhantomData<usize>,
}

impl NoData {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl Data<0> for NoData {
    type Element = i32;

    fn data_type(&self) -> DataType {
        DataType::Int32
    }

    fn shape(&self) -> Shape {
        [0; 0].into()
    }

    fn dimensions(&self) -> Vec<u64> {
        Vec::new()
    }
}

#[derive(PartialEq, Clone)]
pub struct FloatData<const D: usize> {
    shape: [usize; D],
}

/*
// TODO: hide behind feature flag `dodgy` :-)

impl<const D: usize> From<&[usize]> for FloatData<D> {
    fn from(v: &[usize]) -> Self {
        debug_assert_eq!(v.len(), D);

        let mut shape = [0_usize; D];

        for i in 0..D {
            shape[i] = v[i];
        }

        FloatData { shape }
    }
}
*/

impl<const RANK: usize> From<[usize; RANK]> for FloatData<RANK> {
    fn from(shape: [usize; RANK]) -> Self {
        FloatData { shape }
    }
}

impl<const RANK: usize> From<&[usize; RANK]> for FloatData<RANK> {
    fn from(shape: &[usize; RANK]) -> Self {
        FloatData {
            shape: shape.clone(),
        }
    }
}

impl<const RANK: usize> Data<RANK> for FloatData<RANK> {
    type Element = f32;

    fn data_type(&self) -> DataType {
        DataType::Float
    }

    fn shape(&self) -> Shape {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0..self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64[..].into()
    }

    fn dimensions(&self) -> Vec<u64> {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0..self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64
    }
}

#[derive(PartialEq, Clone)]
pub struct DoubleData<const RANK: usize> {
    shape: [usize; RANK],
}

impl<const RANK: usize> Data<RANK> for DoubleData<RANK> {
    type Element = f64;

    fn data_type(&self) -> DataType {
        DataType::Double
    }

    fn shape(&self) -> Shape {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0..self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64[..].into()
    }

    fn dimensions(&self) -> Vec<u64> {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0..self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64
    }
}
