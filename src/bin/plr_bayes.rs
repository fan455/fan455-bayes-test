//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_util::*;
use fan455_util_macro::*;
use fan455_arrf64::*;
use fan455_bayes::*;
use rand::{RngCore, SeedableRng};

type MyRng = rand_xoshiro::Xoshiro256StarStar;


#[allow(non_snake_case)]
fn main()  {

    parse_cmd_args!();

    cmd_arg!(y_path, String, "data/data_y.npy".to_string());
    cmd_arg!(X_path, String, "data/data_X.npy".to_string());
    cmd_arg!(T, usize, 10);

    cmd_arg!(seed, u64, 10000);
    cmd_arg!(random_seed, bool, false);

    cmd_arg!(max_depth, usize, 10);
    cmd_arg!(max_deltaH, f64, 1000.);
    cmd_arg!(delta, f64, 0.8);
    cmd_arg!(gamma, f64, 0.05);
    cmd_arg!(kappa, f64, 0.75);
    cmd_arg!(t0, f64, 10.);
    cmd_arg!(epsilon_init, f64, 0.1);

    cmd_arg!(n_warmup, usize, 10000);
    cmd_arg!(n_sample, usize, 10000);
    cmd_arg!(init_buffer, usize, 75);
    cmd_arg!(base_window, usize, 50);
    cmd_arg!(term_buffer, usize, 25);
    cmd_arg!(n_prog_adapt, usize, 10);
    cmd_arg!(n_prog_draw, usize, 10);

    cmd_vec!(x_index, usize, vec![0]);
    cmd_vec!(x_name, String, Vec::new());
    cmd_arg!(y_name, String, "y".to_string());

    cmd_arg!(do_infer, bool, true);
    cmd_vec_mut!(H0, f64, vec![0.]);
    cmd_arg!(infer_data_path, String, "data/data_inference.csv".to_string());

    cmd_arg!(print_name_width, usize, 15);
    cmd_arg!(print_width, usize, 12);
    cmd_arg!(print_prec, usize, 4);
    cmd_arg!(csv_name_width, usize, 15);
    cmd_arg!(csv_width, usize, 12);
    cmd_arg!(csv_prec, usize, 4);
    
    cmd_arg!(write_sample_data, bool, true);
    cmd_arg!(sample_data_path, String, "data/data_beta.npy".to_string());

    unknown_cmd_args!();

    let y = Arr1::<f64>::read_npy(&y_path);
    let X = Arr2::<f64>::read_npy(&X_path);

    let model = PlrBayes::new(y, X, T).unwrap();
    //println!("model.T = {}", model.T);
    //println!("model.N = {}", model.N);
    //println!("model.K = {}", model.K);

    let mut rng = match random_seed {
        false => MyRng::seed_from_u64(seed),
        true => MyRng::from_entropy(),
    };
    rng.next_u64();
    
    let mut buffer: Arr2<f64> = Arr2::<f64>::new(model.K, n_sample);

    let timer = std::time::Instant::now();

    nuts_dense(
        rng, model, max_depth, max_deltaH, delta, gamma, 
        kappa, t0, epsilon_init, n_warmup, init_buffer, base_window, 
        term_buffer, n_prog_adapt, n_prog_draw, &mut buffer
    );

    let duration = timer.elapsed();
    println!("\nTime elapsed: {:.2?}\n", duration);

    let n_param: usize = x_index.len();
    let mut sample = Arr2::<f64>::new(n_sample, n_param);
    sample.get_rows_as_cols(&buffer, &x_index);

    if do_infer {
        if H0 == vec![0.] {
            H0.resize(n_param, 0.);
        }
        let mut posterior_mean: Vec<f64> = vec![0.; n_param];
        let mut posterior_var: Vec<f64> = vec![0.; n_param];
        let mut H0_percent: Vec<f64> = vec![0.; n_param];
        let mut H0_pvalue: Vec<f64> = vec![0.; n_param];
    
        for i in 0..n_param {
            (posterior_mean[i], posterior_var[i]) = welford(&sample.col(i));
            sample.col_mut(i).sort_ascend();
            (H0_percent[i], H0_pvalue[i]) = infer_by_percent(&sample.col(i), H0[i]);
        }
        let pr0 = VecPrinter{ name_width: print_name_width, width: print_width, prec: print_prec };
        println!("y name: {y_name}");
        pr0.print(&x_index, "x index");
        pr0.print_string(&x_name, "x name");
        pr0.print(&posterior_mean, "posterior mean");
        pr0.print(&posterior_var, "posterior var");
        pr0.print(&H0, "H0");
        pr0.print(&H0_percent, "H0 percent");
        pr0.print(&H0_pvalue, "p-value");

        let mut csv = CsvWriter::new(&infer_data_path);
        csv.name_width = csv_name_width;
        csv.width = csv_width;
        csv.prec = csv_prec;

        csv.write_str(format!("y name: {y_name},\n").as_str());
        csv.write_vec(&x_index, "x index");
        csv.write_vec_string(&x_name, "x name");
        csv.write_vec(&posterior_mean, "posterior mean");
        csv.write_vec(&posterior_var, "posterior var");
        csv.write_vec(&H0, "H0");
        csv.write_vec(&H0_percent, "H0 percent");
        csv.write_vec(&H0_pvalue, "p-value");

        println!("\nInference data saved to: {infer_data_path}");
    }

    if write_sample_data {
        sample.write_npy(&sample_data_path);
        println!("Sample data saved to: {sample_data_path}");
    }

    //let _ = std::process::Command::new("cmd.exe").arg("/c").arg("pause").status();
}
