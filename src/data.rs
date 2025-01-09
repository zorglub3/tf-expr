use tensorflow::DataType;
use tensorflow::TensorType;
use tensorflow::Shape;

pub trait Data: Clone {
    type Element: TensorType;

    fn rank(&self) -> usize;
    fn data_type(&self) -> DataType;
    fn shape(&self) -> Shape;
    fn dimensions(&self) -> Vec<u64>;
}

#[derive(PartialEq, Clone)]
pub struct FloatData<const D: usize> {
    shape: [usize; D],
}

impl<const D: usize> From<&[usize]> for FloatData<D> {
    fn from(v: &[usize]) -> Self {
        todo!()
    }
}

impl<const D: usize> From<[usize; D]> for FloatData<D> {
    fn from(v: [usize; D]) -> Self {
        todo!()
    }
}

impl<const D: usize> Data for FloatData<D> {
    type Element = f32;

    fn rank(&self) -> usize {
        D
    }

    fn data_type(&self) -> DataType {
        DataType::Float
    }

    fn shape(&self) -> Shape {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0 .. self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64[..].into()
    }

    fn dimensions(&self) -> Vec<u64> {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0 .. self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64
    }
}

#[derive(PartialEq, Clone)]
pub struct DoubleData<const D: usize> {
    shape: [usize; D],
}

impl<const D: usize> Data for DoubleData<D> {
    type Element = f64;

    fn rank(&self) -> usize {
        D
    }

    fn data_type(&self) -> DataType {
        DataType::Double
    }

    fn shape(&self) -> Shape {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0 .. self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64[..].into()
    }

    fn dimensions(&self) -> Vec<u64> {
        let mut shape_u64: Vec<u64> = Vec::new();

        for i in 0 .. self.shape.len() {
            shape_u64.push(self.shape[i] as u64);
        }

        shape_u64
    }
}