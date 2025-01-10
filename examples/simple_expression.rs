use tf_expr::data::*;
use tf_expr::expr::*;
use tf_expr::RuntimeSession;

pub fn main() {
    let v1 = vector_float(&[1.0_f32, 2., 3.][..]);
    let v2 = vector_float(&[4.0_f32, 5., 6.][..]);
    let v3 = float_variable("hello", v2.clone(), [3]);

    let e: Expr<FloatData<1>> = v1 + v2 * v3.clone();

    let mut compiler = CompilerScope::new_with_root_scope();

    let _ = compiler.get_operation(&e).unwrap();

    let session = RuntimeSession::new(compiler).unwrap();

    session
        .run_initializers()
        .expect("Failed to run initializer");

    let mut args = session.session_run_args();

    let token = session.request_fetch(&mut args, &e).unwrap();

    // let token = args.request_fetch(&operation, 0);

    session.run(&mut args).expect("Error running session!");

    let output = args.fetch::<f32>(token);

    println!("got output: {:?}", output);
}
