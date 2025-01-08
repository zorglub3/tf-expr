use tf_expr::expr::*;
use tf_expr::var::*;
use tensorflow::Scope;
use tensorflow::DataType;

pub fn main() {
    let mut scope = Scope::new_root_scope();
    let initial_value = Expr::random_normal(DataType::Double, [2, 2]);
    let variable = variable("hello", DataType::Double, [2, 2], &initial_value, &mut scope).unwrap();
    let expression = 2.0_f64 + variable.read();
}
