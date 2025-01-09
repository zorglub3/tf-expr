// use tf_expr::expr::*;
// use tf_expr::var::*;
use tensorflow::Scope;
use tensorflow::DataType;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tf_expr::v3::*;
use tf_expr::data::*;

pub fn main() {
    let v1 = vector_float(&[1.0_f32, 2., 3.][..]);
    let v2 = vector_float(&[4.0_f32, 5., 6.][..]);

    let e: WrappedExpr<FloatData<1>> = v1 + v2;

    let mut scope = Scope::new_root_scope();
    let mut compiler = CompilerScope::new(scope);

    let operation = compiler.get_operation(&e).unwrap();

    let graph = compiler.borrow_scope_mut().graph();
    let session_options = SessionOptions::new();
    let session = Session::new(&session_options, &graph).unwrap();

    let mut args = SessionRunArgs::new();
    let token = args.request_fetch(&operation, 0);

    session.run(&mut args);

    let output = args.fetch::<f32>(token);

    println!("got output: {:?}", output);
    /*
    let mut scope = Scope::new_root_scope();
    let initial_value = Expr::random_normal(DataType::Double, [2, 2]);
    let variable = variable("hello", DataType::Double, [2, 2], &initial_value, &mut scope).unwrap();
    let expression = 2.0_f64 + variable.read();
    */
}
