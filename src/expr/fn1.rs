use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::Data;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;

pub(crate) struct Fn1Expr<const RANK0: usize, D0: Data<RANK0>, const RANK1: usize, D1: Data<RANK1>>
{
    pub(crate) id: Id,
    pub(crate) function: TFFunction1,
    pub(crate) arg: Expr<RANK1, D1>,
    pub(crate) data_type: D0,
}

impl<const RANK0: usize, D0: Data<RANK0>, const RANK1: usize, D1: Data<RANK1>> ExprImpl<RANK0, D0>
    for Fn1Expr<RANK0, D0, RANK1, D1>
{
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

    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status> {
        let arg_output = compiler.get_output(&self.arg)?;

        let operation = match self.function {
            TFFunction1::Tanh => ops::tanh(arg_output, compiler.borrow_scope_mut())?,
            TFFunction1::Exp => ops::exp(arg_output, compiler.borrow_scope_mut())?,
        };

        Ok(CompiledElement::Operation(operation))
    }
}

pub(crate) enum TFFunction1 {
    Tanh,
    Exp,
}
