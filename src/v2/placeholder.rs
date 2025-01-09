use tensorflow::Scope;
use tensorflow::ops;
use tensorflow::Status;
use crate::data::*;
use super::*;
use super::binop::BinOpExpr;
use std::rc::Rc;
use std::ops::{Add, Sub, Div, Mul};

pub struct PlaceholderExpr<D: Data> {
    name: String,
    data_type: D,
}

impl<D: Data + 'static, E: Expr<D> + 'static> Add<E> for PlaceholderExpr<D> {
    type Output = BinOpExpr<D>;

    fn add(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::add(self, rhs, data_type)
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Mul<E> for PlaceholderExpr<D> {
    type Output = BinOpExpr<D>;

    fn mul(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::mul(self, rhs, data_type)
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Sub<E> for PlaceholderExpr<D> {
    type Output = BinOpExpr<D>;

    fn sub(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::sub(self, rhs, data_type)
    }
}

impl<D: Data + 'static, E: Expr<D> + 'static> Div<E> for PlaceholderExpr<D> {
    type Output = BinOpExpr<D>;

    fn div(self, rhs: E) -> BinOpExpr<D> {
        let data_type = self.data_type.clone();

        BinOpExpr::div(self, rhs, data_type)
    }
}

impl<D: Data + 'static> Expr<D> for PlaceholderExpr<D> {
    fn to_operation(&self, scope: &mut Scope) -> Result<CompiledExpr<D>, Status> {
        let operation = ops::Placeholder::new()
            .dtype(self.data_type.data_type())
            .shape(self.data_type.shape())
            .build(&mut scope.with_op_name(&self.name))?;

        let data_type = self.data_type().clone();

        Ok(CompiledExpr {
            expr: Rc::new(PlaceholderExpr {
                name: self.name.clone(),
                data_type,
            }),
            operation,
        })
    }

    fn data_type(&self) -> D {
        self.data_type.clone()
    }
}
