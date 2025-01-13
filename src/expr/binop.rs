use super::{Expr, ExprImpl, Id};
use crate::compiler::{CompiledElement, Compiler};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Shape;
use tensorflow::Status;

pub(crate) struct BinOpExpr<const RANK: usize, D: Data<RANK>> {
    pub(crate) id: Id,
    pub(crate) op: BinaryOperator,
    pub(crate) left: Expr<RANK, D>,
    pub(crate) right: Expr<RANK, D>,
    pub(crate) data_type: D,
}

impl<const RANK: usize, D: Data<RANK>> ExprImpl<RANK, D> for BinOpExpr<RANK, D> {
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

    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status> {
        let left_output = compiler.get_output(&self.left)?;
        let right_output = compiler.get_output(&self.right)?;

        let operation = match self.op {
            BinaryOperator::Add => {
                ops::add(left_output, right_output, compiler.borrow_scope_mut())?
            }
            BinaryOperator::Sub => {
                ops::sub(left_output, right_output, compiler.borrow_scope_mut())?
            }
            BinaryOperator::Mul => {
                ops::mul(left_output, right_output, compiler.borrow_scope_mut())?
            }
            BinaryOperator::Div => {
                ops::div(left_output, right_output, compiler.borrow_scope_mut())?
            }
        };

        Ok(CompiledElement::Operation(operation))
    }
}

pub(crate) enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}
