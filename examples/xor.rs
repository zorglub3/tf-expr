use tf_expr::data::*;
use tf_expr::expr::*;
use tf_expr::*;
use tf_expr::tensordata::TensorData;

pub fn main() {
    let input_shape: [usize; 2] = [2, 1];
    let label_shape: [usize; 2] = [1, 1];

    let input = float_feed("input", &input_shape);
    let label = float_feed("label", &label_shape);

    let weight1_shape: [usize; 2] = [3, 2];
    let weight2_shape: [usize; 2] = [1, 3];

    let w1_init = random_standard_normal(weight1_shape);
    let weight1 = float_variable(
        "weight_layer_1",
        w1_init,
        weight1_shape,
    );

    let w2_init = random_standard_normal(weight2_shape);
    let weight2 = float_variable(
        "weight_layer_2",
        w2_init,
        weight2_shape,
    );

    let output = weight2.read().mat_mul(weight1.read().mat_mul(input.read()).tanh()).tanh();
    let error  = (output.clone() - label.read()) * (output.clone() - label.read());

    let mut compiler = Compiler::new_with_root_scope();

    // let error_op = compiler.get_operation(&error).unwrap();

    let min_error = error.clone().minimize(&[weight1.refer(), weight2.refer()]);

    let _ = compiler.compile(&min_error).unwrap();

    let session = RuntimeSession::new(compiler).unwrap();

    println!("initializing variables");

    session.run_initializers().expect("Failed to run initializers");

    println!("running training iterations");

    for i in 0 .. 10000 {
        let a = i & 1;
        let b = (i & 2) >> 1;
        let l = a ^ b;

        let input_feed = TensorData::<2, FloatData<2>>::new(&input_shape, &[a as f32, b as f32]);
        let label_feed = TensorData::<2, FloatData<2>>::new(&label_shape, &[l as f32]);

        let input_tensor = input_feed.tag().unwrap();
        let label_tensor = label_feed.tag().unwrap();

        let mut args = session.session_run_args();

        session.add_feed(&mut args, &input.refer(), &input_tensor).expect("Could not add input feed");
        session.add_feed(&mut args, &label.refer(), &label_tensor).expect("Could not add label feed");
        session.add_target(&mut args, &min_error);

        let err_token = session.request_fetch(&mut args, &error).unwrap();

        session.run(&mut args).expect("Error running training iteration");

        let output = args.fetch::<f32>(err_token);

        println!("current error: {:?}", output);
    }

    println!("All done");
}

