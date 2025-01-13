use crate::data;
use crate::data::Data;
use tensorflow::Tensor;
use tensorflow::Status;

pub struct TensorData<D: data::Data> {
    pub(crate) data: Option<Vec<D::Element>>,
    pub(crate) data_type: D,
}

impl<const D: usize> TensorData<data::FloatData<D>> {
    pub fn new<S: Into<data::FloatData<D>>>(shape: S, values: &[<data::FloatData<D> as data::Data>::Element]) -> Self {
        let data_type = shape.into();
        let mut data = Vec::new();
        data.extend_from_slice(values);
        TensorData { data: Some(data), data_type }
    }

    pub fn new_with_zero<S: Into<data::FloatData<D>>>(shape: S) -> Self {
        let data_type = shape.into();
        TensorData { data: None, data_type }
    }
}

impl<D: Data + 'static> TensorData<D> {
    pub fn make_tensor(&self) -> Result<Tensor<D::Element>, Status> {
        match &self.data {
            None => Ok(Tensor::new(&self.data_type.dimensions())),
            Some(values) => Tensor::new(&self.data_type.dimensions()).with_values(values),
        }
    }
}

impl<const D: usize> From<&[f32; D]> for TensorData<data::FloatData<1>> {
    fn from(values: &[f32; D]) -> Self {
        TensorData::new([D], values)
    }
}

impl From<&[f32]> for TensorData<data::FloatData<1>> {
    fn from(values: &[f32]) -> Self {
        TensorData::new([values.len()], values)
    }
}

impl From<f32> for TensorData<data::FloatData<0>> {
    fn from(value: f32) -> Self {
        TensorData::new([], &[value])
    }
}
