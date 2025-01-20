use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::Data;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;

pub(crate) struct Fn2Expr<const RANK0: usize, D0: Data<RANK0>, const RANK1: usize, D1: Data<RANK1>, const RANK2: usize, D2: Data<RANK2>>
{
    pub(crate) id: Id,
    pub(crate) function: TFFunction2,
    pub(crate) arg1: Expr<RANK1, D1>,
    pub(crate) arg2: Expr<RANK2, D2>,
    pub(crate) data_type: D0,
}

impl<const RANK0: usize, D0: Data<RANK0>, const RANK1: usize, D1: Data<RANK1>, const RANK2: usize, D2: Data<RANK2>> ExprImpl<RANK0, D0>
    for Fn2Expr<RANK0, D0, RANK1, D1, RANK2, D2>
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
        let arg1_output = compiler.get_output(&self.arg1)?;
        let arg2_output = compiler.get_output(&self.arg2)?;

        let operation = match self.function {
            TFFunction2::MatMul => ops::mat_mul(arg1_output, arg2_output, compiler.borrow_scope_mut())?,
        };

        Ok(CompiledElement::Operation(operation))
    }
}

pub(crate) enum TFFunction2 {
    MatMul,
}
