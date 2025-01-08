use tensorflow::DataType;
use tensorflow::Shape;
use tensorflow::Scope;
use tensorflow::ops;
use tensorflow::Operation;
use tensorflow::Status;
use core::ops::{Add, Sub, Mul, Div};
use crate::var::*;
use crate::val::*;

#[derive(Debug)]
pub enum Expr<'a> {
    BinOp(BinaryOperator, Box<Expr<'a>>, Box<Expr<'a>>),
    UnOp(UnaryOperator, Box<Expr<'a>>),
    Placeholder(DataType, Shape, String),
    Variable(&'a PersistentVariable),
    Value(Value<'a>),
}

impl<'a> PartialEq for Expr<'a> {
    fn eq(&self, other: &Expr<'a>) -> bool {
        use Expr::*;

        match (self, other) {
            (BinOp(o1, e11, e21), BinOp(o2, e12, e22)) => {
                o1 == o2 && e11 == e12 && e21 == e22
            }
            (UnOp(o1, e1), UnOp(o2, e2)) => {
                o1 == o2 && e1 == e2
            }
            (Placeholder(dt1, s1, n1), Placeholder(dt2, s2, n2)) => {
                dt1 == dt2 && s1 == s2 && n1 == n2
            }
            (Variable(v1), Variable(v2)) => {
                v1 == v2
            }
            (Expr::Value(v1), Expr::Value(v2)) => {
                v1 == v2
            }
            _ => false,
        }
    }
}

impl<'a> Expr<'a> {
    pub fn placeholder<S: Into<Shape>>(label: String, shape: S, data_type: DataType) -> Self {
        Expr::Placeholder(data_type, shape.into(), label)
    }

    pub fn random_normal<S: Into<Shape>>(data_type: DataType, shape: S) -> Self {
        Expr::Value(Value::RandomNormal(data_type, shape.into()))
    }

    pub fn to_operation(&self, scope: &mut Scope) -> Result<Operation, Status> {
        use Expr::*;
        use BinaryOperator::*;

        match self {
            BinOp(Add, e1, e2) => {
                ops::add(
                    e1.to_operation(scope)?, 
                    e2.to_operation(scope)?,
                    scope)
            }
            BinOp(Sub, e1, e2) => {
                ops::sub(
                    e1.to_operation(scope)?,
                    e2.to_operation(scope)?,
                    scope)
            }
            BinOp(Mul, e1, e2) => {
                ops::mul(
                    e1.to_operation(scope)?,
                    e2.to_operation(scope)?,
                    scope)
            }
            BinOp(Div, e1, e2) => {
                ops::div(
                    e1.to_operation(scope)?,
                    e2.to_operation(scope)?,
                    scope)
            }
            Placeholder(data_type, shape, name) => {
                ops::Placeholder::new()
                    .dtype(*data_type)
                    .shape(shape.clone())
                    .build(&mut scope.with_op_name(&name))
            }
            Variable(v) => { 
                ops::read_variable_op(v.output().clone(), scope)
            }
            Value(v) => {
                v.to_operation(scope)
            }
            _ => todo!(),
        }
    }
}

impl<'a, V: Into<f64>> From<V> for Expr<'a> {
    fn from(v: V) -> Self {
        Expr::Value(Value::Constant(v.into(), DataType::Double, [1].into()))
    }
}

/*
impl<'a> From<f64> for Expr<'a, DataType::Double> {
    fn from(v: f64) -> Self {
        Expr::ScalarFloat(v)
    }
}

impl<'a> From<f32> for Expr<'a> {
    fn from(v: f32) -> Self {
        Expr::ScalarFloat(f64::from(v))
    }
}
*/

impl<'a, E: Into<Expr<'a>>> Mul<E> for Expr<'a> {
    type Output = Expr<'a>;

    fn mul(self, rhs: E) -> Self::Output {
        Expr::BinOp(BinaryOperator::Mul, Box::new(self), Box::new(rhs.into()))
    }
}

impl<'a, E: Into<Expr<'a>>> Add<E> for Expr<'a> {
    type Output = Expr<'a>;

    fn add(self, rhs: E) -> Self::Output {
        Expr::BinOp(BinaryOperator::Add, Box::new(self), Box::new(rhs.into()))
    }
}

impl<'a> Add<Expr<'a>> for f64 {
   type Output = Expr<'a>;

   fn add(self, rhs: Expr<'a>) -> Self::Output {
       Expr::BinOp(BinaryOperator::Add, Box::new(self.into()), Box::new(rhs))
   }
}

impl<'a, E: Into<Expr<'a>>> Sub<E> for Expr<'a> {
    type Output = Expr<'a>;

    fn sub(self, rhs: E) -> Self::Output {
        Expr::BinOp(BinaryOperator::Sub, Box::new(self), Box::new(rhs.into()))
    }
}

impl<'a, E: Into<Expr<'a>>> Div<E> for Expr<'a> {
    type Output = Expr<'a>;

    fn div(self, rhs: E) -> Self::Output {
        Expr::BinOp(BinaryOperator::Div, Box::new(self), Box::new(rhs.into()))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum BinaryOperator {
    Mul,
    Div,
    Add,
    Sub,
}

#[derive(Debug, PartialEq, Eq)]
pub enum UnaryOperator {
    Inv,
    Neg,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_simple_expr() {
        let i = Expr::scalar_input("test".to_string());
        let e = 2.0_f64 + i;

        let expected =
            Expr::BinOp(
                BinaryOperator::Add,
                Box::new(Expr::ScalarFloat(2.0)),
                Box::new(Expr::ScalarVariable(DataType::Float, "test".to_string())));

        assert_eq!(e, expected);
    }
}
