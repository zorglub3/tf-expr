use crate::compiler::CompiledElement;
use crate::compiler::Compiler;
use crate::data::*;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use tensorflow::Output;
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

pub(crate) trait ExprImpl<D: Data> {
    fn data_type(&self) -> D;
    fn shape(&self) -> Shape;
    fn dimensions(&self) -> Vec<u64>;
    fn id(&self) -> Id;
    fn make_operation(&self, compiler: &mut Compiler) -> Result<CompiledElement, Status>;
}

#[derive(Clone)]
pub struct Expr<D: Data>(pub(crate) Rc<dyn ExprImpl<D>>);

impl CompiledElement {
    pub fn output(&self) -> Output {
        match self {
            CompiledElement::Operation(operation) => operation.output(0),
            CompiledElement::Variable(variable) => variable.output().clone(),
        }
    }
}

impl<D: Data + 'static> Add<Expr<D>> for Expr<D> {
    type Output = Expr<D>;

    fn add(self, rhs: Expr<D>) -> Expr<D> {
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

impl<D: Data + 'static> Sub<Expr<D>> for Expr<D> {
    type Output = Expr<D>;

    fn sub(self, rhs: Expr<D>) -> Expr<D> {
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

impl<D: Data + 'static> Mul<Expr<D>> for Expr<D> {
    type Output = Expr<D>;

    fn mul(self, rhs: Expr<D>) -> Expr<D> {
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

impl<D: Data + 'static> Div<Expr<D>> for Expr<D> {
    type Output = Expr<D>;

    fn div(self, rhs: Expr<D>) -> Expr<D> {
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

impl<const D: usize> Expr<FloatData<D>> {
    pub fn tanh(self) -> Expr<FloatData<D>> {
        let data_type = self.0.data_type();

        Expr(Rc::new(fn1::Fn1Expr {
            id: get_id(),
            function: fn1::TFFunction1::Tanh,
            arg: self.clone(),
            data_type,
        }))
    }

    pub fn exp(self) -> Expr<FloatData<D>> {
        let data_type = self.0.data_type();

        Expr(Rc::new(fn1::Fn1Expr {
            id: get_id(),
            function: fn1::TFFunction1::Exp,
            arg: self.clone(),
            data_type,
        }))
    }
}

pub fn scalar_float(v: f32) -> Expr<FloatData<0>> {
    Expr(Rc::new(constant::ConstantExpr {
        id: get_id(),
        values: vec![v],
        data_type: FloatData::from([]),
    }))
}

pub fn vector_float(v: &[f32]) -> Expr<FloatData<1>> {
    let mut values = Vec::new();

    values.extend_from_slice(v);
    let len = values.len();

    Expr(Rc::new(constant::ConstantExpr {
        id: get_id(),
        values,
        data_type: FloatData::from([len]),
    }))
}

pub fn float_variable<const D: usize, S: Into<FloatData<D>>>(
    name: &str,
    initial_value: Expr<FloatData<D>>,
    shape: S,
) -> Variable<FloatData<D>> {
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
) -> Placeholder<FloatData<D>> {
    Placeholder {
        id: get_id(),
        name: name.to_string(),
        data_type: shape.into(),
    }
}

pub fn random_uniform<const D: usize, S: Into<FloatData<D>>>(shape: S) -> Expr<FloatData<D>> {
    Expr(Rc::new(fn0::Fn0Expr {
        id: get_id(),
        function: fn0::TFFunction0::RandomUniform,
        data_type: shape.into(),
    }))
}

pub fn random_standard_normal<const D: usize, S: Into<FloatData<D>>>(
    shape: S,
) -> Expr<FloatData<D>> {
    Expr(Rc::new(fn0::Fn0Expr {
        id: get_id(),
        function: fn0::TFFunction0::RandomStandardNormal,
        data_type: shape.into(),
    }))
}
