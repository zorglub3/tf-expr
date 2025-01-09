use super::{CompilerScope, Expr, Id, WrappedExpr};
use crate::data::*;
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Shape;
use tensorflow::Status;

pub struct BinOpExpr<D: Data> {
    pub(crate) id: Id,
    pub(crate) op: BinaryOperator,
    pub(crate) left: WrappedExpr<D>,
    pub(crate) right: WrappedExpr<D>,
    pub(crate) data_type: D,
}

impl<D: Data> Expr<D> for BinOpExpr<D> {
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
        let left_output = compiler_scope.get_output(&self.left)?;
        let right_output = compiler_scope.get_output(&self.right)?;

        match self.op {
            BinaryOperator::Add => {
                ops::add(left_output, right_output, compiler_scope.borrow_scope_mut())
            }
            BinaryOperator::Sub => {
                ops::sub(left_output, right_output, compiler_scope.borrow_scope_mut())
            }
            BinaryOperator::Mul => {
                ops::mul(left_output, right_output, compiler_scope.borrow_scope_mut())
            }
            BinaryOperator::Div => {
                ops::div(left_output, right_output, compiler_scope.borrow_scope_mut())
            }
        }
    }
}

pub(crate) enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}
