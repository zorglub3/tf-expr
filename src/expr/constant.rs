use super::{ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use crate::tensordata::TensorData;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;

pub(crate) struct ConstantExpr<D: Data> {
    pub(crate) id: Id,
    pub(crate) value: TensorData<D>,
}

impl<D: Data + 'static> ExprImpl<D> for ConstantExpr<D> {
    fn data_type(&self) -> D {
        self.value.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.value.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.value.data_type.dimensions()
    }

    fn id(&self) -> Id {
        self.id
    }

    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status> {
        let operation = ops::constant(self.value.make_tensor()?, compiler.borrow_scope_mut())?;
        // let operation = ops::constant(self.value.data.clone(), compiler.borrow_scope_mut())?;

        Ok(CompiledElement::Operation(operation))
    }
}
