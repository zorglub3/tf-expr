use crate::data;
use tensorflow::Status;
use tensorflow::Tensor;

pub struct TensorData<const RANK: usize, D: data::Data<RANK>> {
    pub(crate) data: Option<Vec<D::Element>>,
    pub(crate) data_type: D,
}

pub struct TaggedTensor<const RANK: usize, D: data::Data<RANK>> {
    pub(crate) tensor: Tensor<D::Element>,
    #[allow(dead_code)]
    pub(crate) data_type: D,
}

impl<const RANK: usize, D: data::Data<RANK> + 'static> TryFrom<&TensorData<RANK, D>> for TaggedTensor<RANK, D> {
    type Error = Status;

    fn try_from(tensor_data: &TensorData<RANK, D>) -> Result<Self, Status> {
        Ok(Self {
            tensor: tensor_data.make_tensor()?,
            data_type: tensor_data.data_type.clone(),
        })
    }
}

impl<const RANK: usize, D: data::Data<RANK> + 'static> TensorData<RANK, D> {
    pub fn new<S: Into<D>>(shape: S, values: &[D::Element]) -> Self {
        let data_type = shape.into();
        let mut data = Vec::new();
        data.extend_from_slice(values);
        TensorData {
            data: Some(data),
            data_type,
        }
    }

    pub fn new_with_zero<S: Into<D>>(shape: S) -> Self {
        let data_type = shape.into();
        TensorData {
            data: None,
            data_type,
        }
    }

    pub fn make_tensor(&self) -> Result<Tensor<D::Element>, Status> {
        match &self.data {
            None => Ok(Tensor::new(&self.data_type.dimensions())),
            Some(values) => Tensor::new(&self.data_type.dimensions()).with_values(values),
        }
    }

    pub fn tag(&self) -> Result<TaggedTensor<RANK, D>, Status> {
        TaggedTensor::try_from(self)
    }
}

impl<const D: usize> From<&[f32; D]> for TensorData<1, data::FloatData<1>> {
    fn from(values: &[f32; D]) -> Self {
        TensorData::new([D], values)
    }
}

impl From<&[f32]> for TensorData<1, data::FloatData<1>> {
    fn from(values: &[f32]) -> Self {
        TensorData::new([values.len()], values)
    }
}

impl From<f32> for TensorData<0, data::FloatData<0>> {
    fn from(value: f32) -> Self {
        TensorData::new([], &[value])
    }
}
