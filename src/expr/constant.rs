use super::{CompiledElement, CompilerScope, ExprImpl, Id};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;
use tensorflow::Tensor;

pub(crate) struct ConstantExpr<D: Data> {
    pub(crate) id: Id,
    pub(crate) values: Vec<D::Element>,
    pub(crate) data_type: D,
}

impl<D: Data> ExprImpl<D> for ConstantExpr<D> {
    fn data_type(&self) -> D {
        self.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type.dimensions()
    }

    fn id(&self) -> Id {
        self.id
    }

    fn make_operation(
        &self,
        compiler_scope: &mut CompilerScope,
    ) -> Result<CompiledElement, Status> {
        let tensor = Tensor::new(&self.data_type().dimensions()[..]).with_values(&self.values)?;
        let operation = ops::constant(tensor, compiler_scope.borrow_scope_mut())?;

        Ok(CompiledElement::Operation(operation))
    }
}
