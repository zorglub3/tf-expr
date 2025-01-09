use tensorflow::Scope;
use tensorflow::ops;
use tensorflow::Status;
use crate::data::*;
use super::*;
use super::binop::BinOpExpr;
use std::rc::Rc;
use std::ops::{Add, Sub, Div, Mul};

pub struct Fn1Expr<D0: Data, D1: Data> {
    function: TFFun,
    arg: Rc<dyn Expr<D1>>,
    data_type: D0,
}

impl<const D: usize> Fn1Expr<FloatData<D>, FloatData<D>> {
    pub fn tanh<E: Expr<FloatData<D>> + 'static>(arg: E, data_type: FloatData<D>) -> Self {
        Fn1Expr {
            function: TFFun::Tanh,
            arg: Rc::new(arg),
            data_type,
        }
    }

    pub fn exp<E: Expr<FloatData<D>> + 'static>(arg: E, data_type: FloatData<D>) -> Self {
        Fn1Expr {
            function: TFFun::Exp,
            arg: Rc::new(arg),
            data_type,
        }
    }
}

impl<D0: Data + 'static, D1: Data + 'static, E: Expr<D0> + 'static> Add<E> for Fn1Expr<D0, D1> {
    type Output = BinOpExpr<D0>;

    fn add(self, rhs: E) -> BinOpExpr<D0> {
        let data_type = self.data_type.clone();

        BinOpExpr::add(self, rhs, data_type)
    }
}

impl<D0: Data + 'static, D1: Data + 'static, E: Expr<D0> + 'static> Mul<E> for Fn1Expr<D0, D1> {
    type Output = BinOpExpr<D0>;

    fn mul(self, rhs: E) -> BinOpExpr<D0> {
        let data_type = self.data_type.clone();

        BinOpExpr::mul(self, rhs, data_type)
    }
}

impl<D0: Data + 'static, D1: Data + 'static, E: Expr<D0> + 'static> Sub<E> for Fn1Expr<D0, D1> {
    type Output = BinOpExpr<D0>;

    fn sub(self, rhs: E) -> BinOpExpr<D0> {
        let data_type = self.data_type.clone();

        BinOpExpr::sub(self, rhs, data_type)
    }
}

impl<D0: Data + 'static, D1: Data + 'static, E: Expr<D0> + 'static> Div<E> for Fn1Expr<D0, D1> {
    type Output = BinOpExpr<D0>;

    fn div(self, rhs: E) -> BinOpExpr<D0> {
        let data_type = self.data_type.clone();

        BinOpExpr::mul(self, rhs, data_type)
    }
}

impl<D0: Data + 'static, D1: Data + 'static> Expr<D0> for Fn1Expr<D0, D1> {
    fn to_operation(&self, scope: &mut Scope) -> Result<CompiledExpr<D0>, Status> {
        let arg = self.arg.to_operation(scope)?;

        let arg_output = arg.operation.output(0);
        let function = self.function;
        let data_type = self.data_type.clone();

        let operation = match self.function {
            TFFun::Tanh => ops::tanh(arg_output, scope)?,
            TFFun::Exp => ops::exp(arg_output, scope)?, 
        };

        Ok(CompiledExpr {
            expr: Rc::new(Fn1Expr {
                function,
                arg: Rc::new(arg),
                data_type,
            }),
            operation,
        })
    }

    fn data_type(&self) -> D0 {
        self.data_type.clone()
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TFFun {
    Tanh,
    Exp,
}
