use crate::data::*;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use tensorflow::Operation;
use tensorflow::Shape;
use tensorflow::Status;

mod binop;
mod compiler_scope;
mod constant;
mod fn1;
mod placeholder;
mod variable;

pub use compiler_scope::CompilerScope;

static COUNTER: AtomicUsize = AtomicUsize::new(1);

pub(crate) type Id = usize;

pub(crate) fn get_id() -> Id {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub trait Expr<D: Data> {
    fn data_type(&self) -> D;
    fn shape(&self) -> Shape;
    fn dimensions(&self) -> Vec<u64>;
    fn id(&self) -> Id;
    fn make_operation(&self, compiler_scope: &mut CompilerScope) -> Result<Operation, Status>;
}

#[derive(Clone)]
pub struct WrappedExpr<D: Data>(Rc<dyn Expr<D>>);

impl<D: Data + 'static> Add<WrappedExpr<D>> for WrappedExpr<D> {
    type Output = WrappedExpr<D>;

    fn add(self, rhs: WrappedExpr<D>) -> WrappedExpr<D> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Add,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<D: Data + 'static> Sub<WrappedExpr<D>> for WrappedExpr<D> {
    type Output = WrappedExpr<D>;

    fn sub(self, rhs: WrappedExpr<D>) -> WrappedExpr<D> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Sub,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<D: Data + 'static> Mul<WrappedExpr<D>> for WrappedExpr<D> {
    type Output = WrappedExpr<D>;

    fn mul(self, rhs: WrappedExpr<D>) -> WrappedExpr<D> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Mul,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<D: Data + 'static> Div<WrappedExpr<D>> for WrappedExpr<D> {
    type Output = WrappedExpr<D>;

    fn div(self, rhs: WrappedExpr<D>) -> WrappedExpr<D> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Div,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<const D: usize> WrappedExpr<FloatData<D>> {
    pub fn tanh(self) -> WrappedExpr<FloatData<D>> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(fn1::Fn1Expr {
            id: get_id(),
            function: fn1::TFFunction::Tanh,
            arg: self.clone(),
            data_type,
        }))
    }

    pub fn exp(self) -> WrappedExpr<FloatData<D>> {
        let data_type = self.0.data_type();

        WrappedExpr(Rc::new(fn1::Fn1Expr {
            id: get_id(),
            function: fn1::TFFunction::Exp,
            arg: self.clone(),
            data_type,
        }))
    }
}

pub fn scalar_float(v: f32) -> WrappedExpr<FloatData<0>> {
    WrappedExpr(Rc::new(constant::ConstantExpr {
        id: get_id(),
        values: vec![v],
        data_type: FloatData::from([]),
    }))
}

pub fn vector_float(v: &[f32]) -> WrappedExpr<FloatData<1>> {
    let mut values = Vec::new();

    values.extend_from_slice(v);
    let len = values.len();

    WrappedExpr(Rc::new(constant::ConstantExpr {
        id: get_id(),
        values,
        data_type: FloatData::from([len]),
    }))
}

pub fn float_variable<const D: usize, S: Into<FloatData<D>>>(
    name: &str,
    initial_value: WrappedExpr<FloatData<D>>,
    shape: S,
) -> WrappedExpr<FloatData<D>> {
    WrappedExpr(Rc::new(variable::ReadVariable {
        id: get_id(),
        name: name.to_string(),
        initial_value,
        data_type: shape.into(),
    }))
}
