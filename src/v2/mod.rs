use tensorflow::Scope;
use tensorflow::Operation;
use tensorflow::Status;
use crate::data::*;
use std::rc::Rc;
use std::ops::Deref;

pub mod binop;
pub mod fn1;
pub mod placeholder;
pub mod constant;

pub use binop::BinOpExpr;
pub use fn1::Fn1Expr;
pub use placeholder::PlaceholderExpr;
pub use constant::ConstantExpr;

pub trait Expr<D: Data> {
    fn to_operation(&self, scope: &mut Scope) -> Result<CompiledExpr<D>, Status>;
    fn data_type(&self) -> D;
}

impl<D: Data> Expr<D> for Rc<dyn Expr<D>> {
    fn to_operation(&self, scope: &mut Scope) -> Result<CompiledExpr<D>, Status> {
        self.deref().to_operation(scope)
    }

    fn data_type(&self) -> D {
        self.deref().data_type()
    }
}

pub struct CompiledExpr<D: Data> {
    expr: Rc<dyn Expr<D>>,
    operation: Operation,
}

impl<D: Data> Expr<D> for CompiledExpr<D> {
    fn to_operation(&self, _scope: &mut Scope) -> Result<CompiledExpr<D>, Status> {
        Ok(CompiledExpr {
            expr: Rc::clone(&self.expr),
            operation: self.operation.clone(),
        })
    }

    fn data_type(&self) -> D {
        self.expr.data_type()
    }
}

pub fn tanh<const D: usize, E: Expr<FloatData<D>> + 'static>(expr: E) -> fn1::Fn1Expr<FloatData<D>, FloatData<D>> {
    let data_type = expr.data_type();

    fn1::Fn1Expr::tanh(expr, data_type)
}

pub fn exp<const D: usize, E: Expr<FloatData<D>> + 'static>(expr: E) -> fn1::Fn1Expr<FloatData<D>, FloatData<D>> {
    let data_type = expr.data_type();

    fn1::Fn1Expr::exp(expr, data_type)
}
