use tf_expr::data::*;
use tf_expr::expr::*;
use tf_expr::Compiler;
use tf_expr::RuntimeSession;

pub fn main() {
    // let v1 = vector_float(&[1.0_f32, 2., 3.][..]);
    // let v2 = vector_float(&[4.0_f32, 5., 6.][..]);
    let v1: Expr<FloatData<1>> = (&[1.0_f32, 2., 3.]).into();
    let v2: Expr<FloatData<1>> = (&[4.0_f32, 5., 6.]).into();
    // let v1 = (&[1.0_f32, 2., 3.]).into();
    // let v2 = (&[4.0_f32, 5., 6.]).into();
    let v3 = float_variable("hello", v2.clone(), [3]);

    let e: Expr<FloatData<1>> = v1 + v2 * v3.read();

    let mut compiler = Compiler::new_with_root_scope();

    let _ = compiler.get_operation(&e).unwrap();

    let session = RuntimeSession::new(compiler).unwrap();

    session
        .run_initializers()
        .expect("Failed to run initializer");

    let mut args = session.session_run_args();

    let token = session.request_fetch(&mut args, &e).unwrap();

    session.run(&mut args).expect("Error running session!");

    let output = args.fetch::<f32>(token);

    println!("got output: {:?}", output);
}
