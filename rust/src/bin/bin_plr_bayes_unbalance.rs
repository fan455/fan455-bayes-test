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

    cmd_arg!(y_aug_path, String, "data/data_y_aug.npy".to_string());
    cmd_arg!(X_aug_path, String, "data/data_X_aug.npy".to_string());
    cmd_arg!(pat_path, String, "data/data_pat.npy".to_string());
    cmd_arg!(num_path, String, "data/data_unit_range".to_string());

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
    cmd_arg!(y_name, String, "".to_string());

    cmd_arg!(do_infer, bool, true);
    cmd_arg!(confidence_interval, f64, 0.95);
    cmd_vec_mut!(H0, f64, vec![0.]);
    cmd_arg!(infer_data_path, String, "data/data_inference.csv".to_string());

    cmd_arg!(run_freqs, bool, true);
    cmd_arg!(freqs_data_path, String, "data/data_inference_freq.csv".to_string());

    cmd_arg!(print_name_width, usize, 15);
    cmd_arg!(print_width, usize, 12);
    cmd_arg!(print_prec, usize, 4);
    cmd_arg!(csv_name_width, usize, 15);
    cmd_arg!(csv_width, usize, 12);
    cmd_arg!(csv_prec, usize, 4);
    
    cmd_arg!(write_model_data, bool, true);
    cmd_arg!(write_sample_data, bool, true);
    cmd_arg!(sample_data_path, String, "data/data_beta.npy".to_string());
    cmd_arg!(model_data_folder, String, "data/model_data".to_string());

    unknown_cmd_args!();
    print!("\n");

    let n_param: usize = x_index.len();

    let mut model = BayesPlrUnbalance::new(
        &y_aug_path, &X_aug_path, &pat_path, &num_path,
    );
    if write_model_data {
        model.write_data(&model_data_folder);
    }

    // RUn frequentists models
    if run_freqs {
        let CI = confidence_interval;
        let (fgls_beta, fgls_cov) = model.run_fgls();
        let (ols_beta, ols_cov, sols_cov) = model.run_ols();

        if H0 == vec![0.] {
            H0.resize(n_param, 0.);
        }
        let mut fgls_beta_sel = vec![0.; n_param];
        let mut ols_beta_sel = fgls_beta_sel.clone();
        let mut sols_beta_sel = fgls_beta_sel.clone();

        let mut fgls_pvalue = vec![0.; n_param];
        let mut ols_pvalue = fgls_pvalue.clone();
        let mut sols_pvalue = fgls_pvalue.clone();

        let mut fgls_lb = vec![0.; n_param];
        let mut fgls_ub = fgls_lb.clone();
        let mut ols_lb = fgls_lb.clone();
        let mut ols_ub = fgls_lb.clone();
        let mut sols_lb = fgls_lb.clone();
        let mut sols_ub = fgls_lb.clone();

        let df = (model.n_obs - model.K) as f64;

        for (i, k_ref) in x_index.iter().enumerate() {
            let k = *k_ref;
            let H0_ = H0[k];
            {
                let beta_ = fgls_beta[k];
                let se_ = fgls_cov[(k, k)].sqrt();
                fgls_beta_sel[i] = beta_;
                fgls_pvalue[i] = t_test_twotail(beta_, H0_, df, se_).1;
                (fgls_lb[i], fgls_ub[i]) = t_test_ci(beta_, CI, df, se_);
            }
            {
                let beta_ = ols_beta[k];
                let se_ = ols_cov[(k, k)].sqrt();
                ols_beta_sel[i] = beta_;
                ols_pvalue[i] = t_test_twotail(beta_, H0_, df, se_).1;
                (ols_lb[i], ols_ub[i]) = t_test_ci(beta_, CI, df, se_);
            }
            {
                let beta_ = ols_beta[k];
                let se_ = sols_cov[(k, k)].sqrt();
                sols_beta_sel[i] = beta_;
                sols_pvalue[i] = t_test_twotail(beta_, H0_, df, se_).1;
                (sols_lb[i], sols_ub[i]) = t_test_ci(beta_, CI, df, se_);
            }
        }

        let pr0 = VecPrinter{ name_width: print_name_width, width: print_width, prec: print_prec };
        println!("Freqs PLR inference results:");
        println!("y name: {y_name}");
        pr0.print(&x_index, "x index");
        pr0.print_string(&x_name, "x name");
        pr0.print(&H0, "H0");
        pr0.print(&ols_beta_sel, "beta, ols");
        pr0.print(&ols_beta_sel, "beta, sols");
        pr0.print(&fgls_beta_sel, "beta, fgls");
        print!("\n");
        pr0.print(&ols_pvalue, "p-value, ols");
        pr0.print(&sols_pvalue, "p-value, sols");
        pr0.print(&fgls_pvalue, "p-value, fgls");
        print!("\n");
        pr0.print(&ols_lb, &format!("{CI:.2} lb, ols"));
        pr0.print(&ols_ub, &format!("{CI:.2} ub, ols"));
        pr0.print(&sols_lb, &format!("{CI:.2} lb, sols"));
        pr0.print(&sols_ub, &format!("{CI:.2} ub, sols"));
        pr0.print(&fgls_lb, &format!("{CI:.2} lb, fgls"));
        pr0.print(&fgls_ub, &format!("{CI:.2} ub, fgls"));

        let mut csv = CsvWriter::new(&freqs_data_path);
        csv.name_width = csv_name_width;
        csv.width = csv_width;
        csv.prec = csv_prec;

        csv.write_str(format!("y name: {y_name}\n").as_str());
        csv.write_vec(&x_index, "x index");
        csv.write_vec_string(&x_name, "x name");
        csv.write_vec(&H0, "H0");
        csv.write_vec(&ols_beta_sel, "beta, ols");
        csv.write_vec(&ols_beta_sel, "beta, sols");
        csv.write_vec(&fgls_beta_sel, "beta, fgls");
        csv.write_newline();
        csv.write_vec(&ols_pvalue, "p-value, ols");
        csv.write_vec(&sols_pvalue, "p-value, sols");
        csv.write_vec(&fgls_pvalue, "p-value, fgls");
        csv.write_newline();
        csv.write_vec(&ols_lb, &format!("{CI:.2} lb, ols"));
        csv.write_vec(&ols_ub, &format!("{CI:.2} ub, ols"));
        csv.write_vec(&sols_lb, &format!("{CI:.2} lb, sols"));
        csv.write_vec(&sols_ub, &format!("{CI:.2} ub, sols"));
        csv.write_vec(&fgls_lb, &format!("{CI:.2} lb, fgls"));
        csv.write_vec(&fgls_ub, &format!("{CI:.2} ub, fgls"));

        println!("\nInference data (frequentists) saved to: {freqs_data_path}");
    }

    // Run NUTS.
    let mut buffer: Mat<f64> = Mat::<f64>::new(n_param, n_sample);
    let mut rng = match random_seed {
        false => MyRng::seed_from_u64(seed),
        true => MyRng::from_entropy(),
    };
    rng.next_u64();

    let timer = std::time::Instant::now();

    model = nuts_dense(
        rng, model, max_depth, max_deltaH, delta, gamma, 
        kappa, t0, epsilon_init, n_warmup, init_buffer, base_window, 
        term_buffer, n_prog_adapt, n_prog_draw, &mut buffer
    );
    println!("Number of model updates: {}", model.n_update);

    let duration = timer.elapsed();
    println!("\nTime elapsed: {:.2?}\n", duration);

    let mut sample = Mat::<f64>::new(n_sample, n_param);
    sample.get_trans(&buffer);
    //sample.get_rows_as_cols(&buffer, &x_index);
    
    if do_infer {
        let CI = confidence_interval;
        if H0 == vec![0.] {
            H0.resize(n_param, 0.);
        }
        let mut posterior_mean: Vec<f64> = vec![0.; n_param];
        let mut posterior_var: Vec<f64> = vec![0.; n_param];
        let mut H0_percent: Vec<f64> = vec![0.; n_param];
        let mut H0_pvalue: Vec<f64> = vec![0.; n_param];
        let mut ci_lb: Vec<f64> = vec![0.; n_param];
        let mut ci_ub: Vec<f64> = vec![0.; n_param];
    
        for i in 0..n_param {
            (posterior_mean[i], posterior_var[i]) = welford(&sample.col(i));
            sample.col_mut(i).sort_ascend();
            (H0_percent[i], H0_pvalue[i]) = infer_by_percent(&sample.col(i), H0[i]);
            (ci_lb[i], ci_ub[i]) = infer_by_quantile(&sample.col(i), confidence_interval);
        }
        let pr0 = VecPrinter{ name_width: print_name_width, width: print_width, prec: print_prec };
        println!("Bayes PLR inference results:");
        println!("y name: {y_name}");
        pr0.print(&x_index, "x index");
        pr0.print_string(&x_name, "x name");
        pr0.print(&posterior_mean, "posterior mean");
        pr0.print(&posterior_var, "posterior var");
        pr0.print(&H0, "H0");
        pr0.print(&H0_percent, "H0 percent");
        pr0.print(&H0_pvalue, "p-value");
        pr0.print(&ci_lb, &format!("{CI:.2} lb"));
        pr0.print(&ci_ub, &format!("{CI:.2} ub"));

        let mut csv = CsvWriter::new(&infer_data_path);
        csv.name_width = csv_name_width;
        csv.width = csv_width;
        csv.prec = csv_prec;

        csv.write_str(format!("y name: {y_name}\n").as_str());
        csv.write_vec(&x_index, "x index");
        csv.write_vec_string(&x_name, "x name");
        csv.write_vec(&posterior_mean, "posterior mean");
        csv.write_vec(&posterior_var, "posterior var");
        csv.write_vec(&H0, "H0");
        csv.write_vec(&H0_percent, "H0 percent");
        csv.write_vec(&H0_pvalue, "p-value");

        println!("\nInference data (Bayes) saved to: {infer_data_path}");
    }

    if write_sample_data {
        sample.write_npy(&sample_data_path);
        println!("Sample data saved to: {sample_data_path}");
    }

    //let _ = std::process::Command::new("cmd.exe").arg("/c").arg("pause").status();
}
