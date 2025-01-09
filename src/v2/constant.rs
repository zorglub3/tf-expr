use tensorflow::Scope;
use tensorflow::ops;
use tensorflow::Status;
use tensorflow::Tensor;
use crate::data::*;
use super::*;
use super::binop::BinOpExpr;
use std::rc::Rc;
use std::ops::{Add, Sub, Div, Mul};

pub struct ConstantExpr<D: Data> {
    values: Vec<D::Element>,
    data_type: D,
}

impl<D: Data + 'static, E: Expr<D> + 'static> Add<E> for ConstantExpr<D> {
    type Output = BinOpExpr<D>;

    fn add(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::add(self, rhs, data_type)
    }
}

impl<D: Data + 'static> Expr<D> for ConstantExpr<D> {
    fn to_operation(&self, scope: &mut Scope) -> Result<CompiledExpr<D>, Status> {
        let operation = ops::constant(
            Tensor::new(&self.data_type().dimensions()[..]).with_values(&self.values)?, 
            scope,
        )?;

        let data_type = self.data_type.clone();

        Ok(CompiledExpr {
            expr: Rc::new(ConstantExpr {
                values: self.values.clone(),
                data_type,
            }),
            operation,
        })
    }

    fn data_type(&self) -> D {
        self.data_type.clone()
    }
}

impl From<f32> for ConstantExpr<FloatData<0>> {
    fn from(v: f32) -> Self {
        ConstantExpr {
            values: vec![v],
            data_type: FloatData::from([]),
        }
    }
}

impl From<&[f32]> for ConstantExpr<FloatData<1>> {
    fn from(v: &[f32]) -> Self {
        let mut values = Vec::new();

        values.extend_from_slice(v);
        let len = values.len();

        ConstantExpr { values, data_type: FloatData::from([len]) }
    }
}
