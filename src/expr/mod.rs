use crate::compiler::CompiledElement;
use crate::compiler::Compiler;
use crate::data::*;
use crate::tensordata::TensorData;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use tensorflow::Shape;
use tensorflow::Status;

mod binop;
mod constant;
mod fn0;
mod fn1;
mod optimize;
mod placeholder;
mod variable;

pub use placeholder::Placeholder;
pub use placeholder::PlaceholderRef;
pub use variable::Variable;
pub use variable::VariableRef;

static COUNTER: AtomicUsize = AtomicUsize::new(1);

pub(crate) type Id = usize;

pub(crate) fn get_id() -> Id {
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub(crate) trait ExprImpl<const RANK: usize, D: Data<RANK>> {
    fn data_type(&self) -> D;
    fn shape(&self) -> Shape;
    fn dimensions(&self) -> Vec<u64>;
    fn id(&self) -> Id;
    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status>;
}

#[derive(Clone)]
pub struct Expr<const RANK: usize, D: Data<RANK>>(pub(crate) Rc<dyn ExprImpl<RANK, D>>);

impl<const RANK: usize, D: Data<RANK> + 'static> Add<Expr<RANK, D>> for Expr<RANK, D> {
    type Output = Expr<RANK, D>;

    fn add(self, rhs: Expr<RANK, D>) -> Expr<RANK, D> {
        let data_type = self.0.data_type();

        Expr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Add,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<const RANK: usize, D: Data<RANK> + 'static> Sub<Expr<RANK, D>> for Expr<RANK, D> {
    type Output = Expr<RANK, D>;

    fn sub(self, rhs: Expr<RANK, D>) -> Expr<RANK, D> {
        let data_type = self.0.data_type();

        Expr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Sub,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<const RANK: usize, D: Data<RANK> + 'static> Mul<Expr<RANK, D>> for Expr<RANK, D> {
    type Output = Expr<RANK, D>;

    fn mul(self, rhs: Expr<RANK, D>) -> Expr<RANK, D> {
        let data_type = self.0.data_type();

        Expr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Mul,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<const RANK: usize, D: Data<RANK> + 'static> Div<Expr<RANK, D>> for Expr<RANK, D> {
    type Output = Expr<RANK, D>;

    fn div(self, rhs: Expr<RANK, D>) -> Expr<RANK, D> {
        let data_type = self.0.data_type();

        Expr(Rc::new(binop::BinOpExpr {
            id: get_id(),
            op: binop::BinaryOperator::Div,
            left: self.clone(),
            right: rhs.clone(),
            data_type,
        }))
    }
}

impl<const R: usize> Expr<R, FloatData<R>> {
    pub fn tanh(self) -> Expr<R, FloatData<R>> {
        let data_type = self.0.data_type();

        Expr(Rc::new(fn1::Fn1Expr {
            id: get_id(),
            function: fn1::TFFunction1::Tanh,
            arg: self.clone(),
            data_type,
        }))
    }

    pub fn exp(self) -> Expr<R, FloatData<R>> {
        let data_type = self.0.data_type();

        Expr(Rc::new(fn1::Fn1Expr {
            id: get_id(),
            function: fn1::TFFunction1::Exp,
            arg: self.clone(),
            data_type,
        }))
    }

    pub fn minimize(self, variables: Vec<VariableRef>) -> Expr<0, NoData> {
        Expr(Rc::new(optimize::AdaDeltaMinimizeExpr::<
            R,
            FloatData<R>,
            FloatData<0>,
        > {
            id: get_id(),
            loss: self.clone(),
            variables,
            learning_rate: None,
            rho: None,
            epsilon: None,
        }))
    }
}

impl<const RANK: usize, D: Data<RANK> + 'static, T: Into<TensorData<RANK, D>>> From<T>
    for Expr<RANK, D>
{
    fn from(t: T) -> Self {
        Expr(Rc::new(constant::ConstantExpr {
            id: get_id(),
            value: t.into(),
        }))
    }
}

pub fn scalar<D: Data<0> + From<[usize; 0]> + 'static, T: Into<D::Element>>(v: T) -> Expr<0, D> {
    Expr(Rc::new(constant::ConstantExpr {
        id: get_id(),
        value: TensorData::new([], &[v.into()]),
    }))
}

pub fn vector<D: Data<1> + From<[usize; 1]> + 'static>(v: &[D::Element]) -> Expr<1, D> {
    Expr(Rc::new(constant::ConstantExpr {
        id: get_id(),
        value: TensorData::new([v.len()], v),
    }))
}

pub fn float_variable<const D: usize, S: Into<FloatData<D>>>(
    name: &str,
    initial_value: Expr<D, FloatData<D>>,
    shape: S,
) -> Variable<D, FloatData<D>> {
    Variable {
        id: get_id(),
        name: name.to_string(),
        initial_value,
        data_type: shape.into(),
    }
}

pub fn float_feed<const D: usize, S: Into<FloatData<D>>>(
    name: &str,
    shape: S,
) -> Placeholder<D, FloatData<D>> {
    Placeholder {
        id: get_id(),
        name: name.to_string(),
        data_type: shape.into(),
    }
}

pub fn random_uniform<const D: usize, S: Into<FloatData<D>>>(shape: S) -> Expr<D, FloatData<D>> {
    Expr(Rc::new(fn0::Fn0Expr {
        id: get_id(),
        function: fn0::TFFunction0::RandomUniform,
        data_type: shape.into(),
    }))
}

pub fn random_standard_normal<const D: usize, S: Into<FloatData<D>>>(
    shape: S,
) -> Expr<D, FloatData<D>> {
    Expr(Rc::new(fn0::Fn0Expr {
        id: get_id(),
        function: fn0::TFFunction0::RandomStandardNormal,
        data_type: shape.into(),
    }))
}
