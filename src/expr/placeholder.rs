use super::{CompilerScope, Expr, Id};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Shape;
use tensorflow::Status;

pub struct PlaceholderExpr<D: Data> {
    pub(crate) id: Id,
    pub(crate) name: String,
    pub(crate) data_type: D,
}

impl<D: Data> Expr<D> for PlaceholderExpr<D> {
    fn id(&self) -> Id {
        self.id
    }

    fn data_type(&self) -> D {
        self.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type().dimensions()
    }

    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status> {
        ops::Placeholder::new()
            .dtype(self.data_type.data_type())
            .shape(self.data_type.shape())
            .build(&mut compiler_scope.borrow_scope_mut().with_op_name(&self.name))
    }
}
