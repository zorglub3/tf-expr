use tensorflow::Scope;
use tensorflow::ops;
use tensorflow::Status;
use crate::data::*;
use super::*;
use std::ops::{Add, Sub, Div, Mul};

pub struct BinOpExpr<D: Data> {
    op: BinaryOperator,
    left: Rc<dyn Expr<D>>,
    right: Rc<dyn Expr<D>>,
    data_type: D,
}

impl<D: Data> BinOpExpr<D> {
    pub fn add<E1: Expr<D> + 'static, E2: Expr<D> + 'static>(
        left: E1,
        right: E2,
        data_type: D,
    ) -> Self {
        BinOpExpr {
            op: BinaryOperator::Add,
            left: Rc::new(left),
            right: Rc::new(right),
            data_type,
        }
    }

    pub fn mul<E1: Expr<D> + 'static, E2: Expr<D> + 'static>(
        left: E1,
        right: E2,
        data_type: D,
    ) -> Self {
        BinOpExpr {
            op: BinaryOperator::Mul,
            left: Rc::new(left),
            right: Rc::new(right),
            data_type,
        }
    }

    pub fn sub<E1: Expr<D> + 'static, E2: Expr<D> + 'static>(
        left: E1,
        right: E2,
        data_type: D,
    ) -> Self {
        BinOpExpr {
            op: BinaryOperator::Sub,
            left: Rc::new(left),
            right: Rc::new(right),
            data_type,
        }
    }

    pub fn div<E1: Expr<D> + 'static, E2: Expr<D> + 'static>(
        left: E1,
        right: E2,
        data_type: D,
    ) -> Self {
        BinOpExpr {
            op: BinaryOperator::Div,
            left: Rc::new(left),
            right: Rc::new(right),
            data_type,
        }
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Add<E> for BinOpExpr<D> {
    type Output = BinOpExpr<D>;

    fn add(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::add(self, rhs, data_type)
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Mul<E> for BinOpExpr<D> {
    type Output = BinOpExpr<D>;

    fn mul(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::mul(self, rhs, data_type)
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Sub<E> for BinOpExpr<D> {
    type Output = BinOpExpr<D>;

    fn sub(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::sub(self, rhs, data_type)
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Div<E> for BinOpExpr<D> {
    type Output = BinOpExpr<D>;

    fn div(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::div(self, rhs, data_type)
    }
}


impl<D: Data + 'static> Expr<D> for BinOpExpr<D> {
    fn to_operation(&self, scope: &mut Scope) -> Result<CompiledExpr<D>, Status> {
        let left = self.left.to_operation(scope)?;
        let right = self.right.to_operation(scope)?;

        let left_output = left.operation.output(0);
        let right_output = right.operation.output(0);
        let op = self.op;
        let data_type = self.data_type.clone();

        let operation = match self.op {
            BinaryOperator::Add => ops::add(left_output, right_output, scope)?,
            BinaryOperator::Sub => ops::sub(left_output, right_output, scope)?,
            BinaryOperator::Mul => ops::mul(left_output, right_output, scope)?,
            BinaryOperator::Div => ops::div(left_output, right_output, scope)?,
        };

        Ok(CompiledExpr {
            expr: Rc::new(BinOpExpr {
                op,
                left: Rc::new(left),
                right: Rc::new(right),
                data_type,
            }),
            operation,
        })
    }

    fn data_type(&self) -> D {
        self.data_type.clone()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
}
