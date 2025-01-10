use tensorflow::Tensor;
use tf_expr::data::*;
use tf_expr::expr::*;
use tf_expr::*;

pub fn main() {
    let v1 = vector_float(&[1.0_f32, 2., 3.][..]);
    let v2 = float_feed("the_feed", [3]);

    let e: Expr<FloatData<1>> = v1 + v2.read();

    let mut compiler = Compiler::new_with_root_scope();

    let _ = compiler.get_operation(&e).unwrap();

    let session = RuntimeSession::new(compiler).unwrap();

    session
        .run_initializers()
        .expect("Failed to run initializer");

    let mut feed_data = Tensor::<f32>::new(&[3]);
    feed_data[0] = 4.;
    feed_data[1] = 5.;
    feed_data[2] = 6.;

    let mut args = session.session_run_args();
    let token = session.request_fetch(&mut args, &e).unwrap();
    session
        .add_feed(&mut args, &v2.refer(), &feed_data)
        .expect("Failed to add feed");

    session.run(&mut args).expect("Error running session");

    let output = args.fetch::<f32>(token);

    println!("got output: {:?}", output);
}
