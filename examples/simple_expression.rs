// use tf_expr::expr::*;
// use tf_expr::var::*;
use tensorflow::Scope;
use tensorflow::DataType;
use tf_expr::v2::*;
use tf_expr::data::*;

pub fn main() {
    let c1: &dyn Expr<FloatData<1>> = &[1.0_f32, 2.][..].into();
    let c2: &dyn Expr<FloatData<1>> = &[2.0_f32, 3.][..].into();

    let expr = c1 + c2;
    /*
    let mut scope = Scope::new_root_scope();
    let initial_value = Expr::random_normal(DataType::Double, [2, 2]);
    let variable = variable("hello", DataType::Double, [2, 2], &initial_value, &mut scope).unwrap();
    let expression = 2.0_f64 + variable.read();
    */
}
