use tensorflow::DataType;
use tensorflow::Shape;
use core::ops::{Add, Sub, Mul, Div};

#[derive(Debug, PartialEq)]
pub enum Expr {
    BinOp(BinaryOperator, Box<Expr>, Box<Expr>),
    UnOp(UnaryOperator, Box<Expr>),
    ScalarFloat(f64),
    ScalarInput(String),
    Input(DataType, Shape, String),
}

impl Expr {
    pub fn scalar_input(label: String) -> Self {
        Expr::ScalarInput(label)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinaryOperator::Mul, Box::new(self), Box::new(rhs))
    }
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        Expr::BinOp(BinaryOperator::Add, Box::new(self), Box::new(rhs))
    }
}

impl Add<Expr> for f64 {
   type Output = Expr;

   fn add(self, rhs: Expr) -> Self::Output {
       Expr::BinOp(BinaryOperator::Add, Box::new(Expr::ScalarFloat(self)), Box::new(rhs))
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
                Box::new(Expr::ScalarInput("test".to_string())));

        assert_eq!(e, expected);
    }
}
