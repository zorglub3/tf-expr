use tf_expr::data::*;
use tf_expr::expr::*;
use tf_expr::*;
use tf_expr::tensordata::TensorData;

pub fn main() {
    let v1: Expr<1, FloatData<1>> = (&[1.0_f32, 2., 3.]).into();
    let v2 = float_feed("the_feed", [3]);

    let e: Expr<1, FloatData<1>> = v1 + v2.read();

    let mut compiler = Compiler::new_with_root_scope();

    let _ = compiler.get_operation(&e).unwrap();

    let session = RuntimeSession::new(compiler).unwrap();

    session
        .run_initializers()
        .expect("Failed to run initializer");

    let feed_data = TensorData::<1, FloatData<1>>::new(&[3], &[4., 5., 6.]);
    let feed_tensor = feed_data.tag().unwrap();

    let mut args = session.session_run_args();
    let token = session.request_fetch(&mut args, &e).unwrap();
    session
        .add_feed(&mut args, &v2.refer(), &feed_tensor)
        .expect("Failed to add feed");

    session.run(&mut args).expect("Error running session");

    let output = args.fetch::<f32>(token);

    println!("got output: {:?}", output);
}
