use tensorflow::Scope;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tf_expr::data::*;
use tf_expr::expr::*;

pub fn main() {
    let v1 = vector_float(&[1.0_f32, 2., 3.][..]);
    let v2 = vector_float(&[4.0_f32, 5., 6.][..]);
    let v3 = float_variable("hello", v2.clone(), [3]);

    let e: WrappedExpr<FloatData<1>> = v1 + v2 * v3.clone();

    let scope = Scope::new_root_scope();
    let mut compiler = CompilerScope::new(scope);

    let operation = compiler.get_operation(&e).unwrap();
    let variable = compiler.get_variable(&v3).unwrap();
    let initializer = variable.initializer();

    let graph = compiler.borrow_scope_mut().graph();
    let session_options = SessionOptions::new();
    let session = Session::new(&session_options, &graph).unwrap();

    let mut args = SessionRunArgs::new();
    let _ = args.request_fetch(&initializer, 0);
    let token = args.request_fetch(&operation, 0);

    session.run(&mut args).expect("Error running session!");

    let output = args.fetch::<f32>(token);

    println!("got output: {:?}", output);
}
