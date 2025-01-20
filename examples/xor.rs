use tf_expr::data::*;
use tf_expr::expr::*;
use tf_expr::*;
use tf_expr::tensordata::TensorData;

const HIDDEN_SIZE: usize = 8;
const TRAINING_ITERATIONS: usize = 100_000;

pub fn main() {
    let input_shape: [usize; 2] = [2, 1];
    let label_shape: [usize; 2] = [1, 1];

    let input = float_feed("input", &input_shape);
    let label = float_feed("label", &label_shape);

    let weight1_shape: [usize; 2] = [HIDDEN_SIZE, 2];
    let weight2_shape: [usize; 2] = [1, HIDDEN_SIZE];

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
    let error = output.clone() - label.read();
    let error_sqr = error.clone() * error;

    let mut compiler = Compiler::new_with_root_scope();

    let min_error = error_sqr.clone().minimize(&[weight2.refer(), weight1.refer()]);

    let _ = compiler.compile(&min_error).unwrap();

    let session = RuntimeSession::new(compiler).unwrap();

    println!("initializing variables");

    session.run_initializers().expect("Failed to run initializers");

    println!("running training iterations");

    for j in 0 .. TRAINING_ITERATIONS {
        let mut error_sum: f32 = 0.;

        for i in 0 .. 4 {
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

            let err_token = session.request_fetch(&mut args, &error_sqr).unwrap();

            session.run(&mut args).expect("Error running training iteration");

            let output = args.fetch::<f32>(err_token).unwrap();

            error_sum += output[0].sqrt();
        }

        if j % 100 == 0 {
            println!("Iteration {}. Error: {}", j, error_sum);
        }
    }

    println!("All done");
}

