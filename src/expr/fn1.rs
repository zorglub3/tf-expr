use super::{CompilerScope, Expr, Id, WrappedExpr};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Shape;
use tensorflow::Status;

pub struct Fn1Expr<D0: Data, D1: Data> {
    pub(crate) id: Id,
    pub(crate) function: TFFunction,
    pub(crate) arg: WrappedExpr<D1>,
    pub(crate) data_type: D0,
}

impl<D0: Data, D1: Data> Expr<D0> for Fn1Expr<D0, D1> {
    fn id(&self) -> Id {
        self.id
    }

    fn data_type(&self) -> D0 {
        self.data_type.clone()
    }

    fn shape(&self) -> Shape {
        self.data_type.shape()
    }

    fn dimensions(&self) -> Vec<u64> {
        self.data_type.dimensions()
    }

    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status> {
        let arg_output = compiler_scope.get_output(&self.arg)?;

        match self.function {
            TFFunction::Tanh => ops::tanh(arg_output, compiler_scope.borrow_scope_mut()),
            TFFunction::Exp => ops::exp(arg_output, compiler_scope.borrow_scope_mut()),
        }
    }
}

pub(crate) enum TFFunction {
    Tanh,
    Exp,
}