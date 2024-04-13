//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_util::*;
use fan455_util_macro::*;
use fan455_arrf64::*;
use fan455_arrf64_macro::*;
use fan455_bayes::*;
use rand::{RngCore, SeedableRng};
//use rand::distributions::{Bernoulli, Standard, Distribution};
//use rand_distr::{Distribution, Uniform};

type MyRng = rand_xoshiro::Xoshiro256StarStar;


#[allow(non_snake_case)]
fn main()  {

    let timer = std::time::Instant::now();

    parse_cmd_args!();

    cmd_arg!(T, usize, 10);
    cmd_arg!(N, usize, 100);
    cmd_arg!(u_rho, f64, 0.75);
    cmd_arg!(u_scale, f64, 1.);
    cmd_arg!(u_nonnormal, bool, false); // Default is normal.
    cmd_arg!(u_skew, f64, 0.); // Default is normal.
    cmd_arg!(u_kurt, f64, 3.); // Default is normal.
    cmd_vec!(beta, f64);
    cmd_arg!(beta_other, f64, 1.);
    cmd_vec!(X_loc, f64);
    cmd_vec!(X_scale, f64);
    cmd_vec!(X_rho, f64);
    cmd_vec!(H0, f64);

    cmd_arg!(seed0, u64, 1000);
    cmd_arg!(seed, u64, 10000);
    cmd_arg!(random_seed0, bool, false);
    cmd_arg!(random_seed, bool, false);
    cmd_arg!(n_sim, usize, 100);
    cmd_arg!(n_prog_sim, usize, 10);
    cmd_arg!(significance, f64, 0.05);

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

    cmd_arg!(print_name_width, usize, 20);
    cmd_arg!(print_width, usize, 12);
    cmd_arg!(print_prec, usize, 4);
    cmd_arg!(csv_name_width, usize, 20);
    cmd_arg!(csv_width, usize, 12);
    cmd_arg!(csv_prec, usize, 4);
    cmd_arg!(result_data_path, String, "data/comparison_results.csv".to_string());

    cmd_arg!(X_path, String, "data/data_sim_X.npy".to_string());
    cmd_arg!(use_saved_X, bool, true);
    cmd_arg!(save_X, bool, false);

    unknown_cmd_args!();


    // Set up the rng.
    let mut rng0 = match random_seed0 {
        false => MyRng::seed_from_u64(seed0),
        true => MyRng::from_entropy(),
    };
    let mut rng = match random_seed {
        false => MyRng::seed_from_u64(seed),
        true => MyRng::from_entropy(),
    };
    rng0.next_u64();
    rng.next_u64();


    // Generate beta.
    let K0 = beta.len();
    assert_multi_eq!(K0, X_loc.len(), X_scale.len(), X_rho.len(), H0.len());
    let K = K0 + T;
    let mut beta_full = Arr1::<f64>::new_set(K, beta_other);
    for elem!(a, b) in mzip!(beta_full.subvec_mut(0, K0), &beta) {*a = *b;}

    // Generate X.
    let mut X = match use_saved_X {
        true => Arr2::<f64>::read_npy(&X_path),
        false => Arr2::<f64>::new(N*T, K),
    };
    if !use_saved_X {
        for t in 0..T {
            for x in X.subvec2_mut(t*N, t+K0, (t+1)*N, t+K0) {
                *x = 1.;
            }
        }
        let mut rv0: Vec<MvNormal> = Vec::with_capacity(K0);
        for rho in X_rho {
            let cor = get_cor_ar1(T, rho);
            rv0.push( MvNormal::new(cor) );
        }
        let mut sample = Arr2::<f64>::new(N, T);
        let mut loc = Arr1::<f64>::new(T);
        let mut scale = Arr1::<f64>::new(T);

        for elem!(k, rv, loc_, scale_) in mzip!(0..K0, &rv0, &X_loc, &X_scale) {
            loc.set(*loc_);
            scale.set(*scale_);
            rv.ndraw_with_loc_scale(&mut rng0, &mut sample, &loc, &scale);
            X.col_mut(k).copy(&sample);
        }
    }
    if save_X {
        X.write_npy(&X_path);
    }
    
    // Generate u.
    let u_cor = get_cor_ar1(T, u_rho);
    let u_scale_vec = Arr1::<f64>::new_set(T, u_scale);
    let V = cor_to_cov(&u_cor, &u_scale_vec);

    let pr0 = RMatPrinter::new(8, 2);
    pr0.print("V", &V);

    let rv1 = MvNormal::new( V.clone() );
    let rv2 = MvNonNormal::new(
        Arr1::<f64>::new_set(T, u_skew), Arr1::<f64>::new_set(T, u_kurt), u_cor.clone()
    );

    let mut u = Arr2::<f64>::new(N, T);
    let mut y = Arr1::<f64>::new(N*T);

    let n_simulation: Vec<usize> = vec![n_sim; K0];
    let mut params: Vec<usize> = Vec::with_capacity(K0);
    for i in 0..K0 {params.push(i);}


    // Set up the models.
    // PLR Bayes
    let M0 = PlrBayes::new(y, X, T).unwrap();
    let mut nuts = NutsDense::new(
        rng, M0, max_depth, max_deltaH, delta, gamma, kappa, t0, epsilon_init
    );
    let mut nuts_tmp = NutsBuf::new(K);
    let mut buffer = Arr2::<f64>::new(K, n_sample);
    let mut sample = Arr2::<f64>::new(n_sample, K0);

    // PLR GLS
    y = nuts.model.y;
    X = nuts.model.X;
    let mut M1 = PlrGls::new(y, X, T, V.clone()).unwrap();

    // PLR FGLS
    y = M1.base.y;
    X = M1.base.X;
    let mut M2 = PlrGls::new_uninit(y, X, T).unwrap();
    
    // PLR OLS
    y = M2.base.y;
    X = M2.base.X;
    let mut M3 = PlrOls::new(y, X, T).unwrap();
    let mut V_tmp = Arr2::<f64>::new(T, T);

    // Buffer to store results
    let mut n_reject_0: Vec<usize> = vec![0; K0];
    let mut n_reject_1: Vec<usize> = n_reject_0.clone();
    let mut n_reject_2 = n_reject_0.clone();
    let mut n_reject_3 = n_reject_0.clone();
    let mut n_reject_4 = n_reject_0.clone();

    let mut beta_avg_0: Vec<f64> = vec![0.; K0];
    let mut beta_avg_1 = beta_avg_0.clone();
    let mut beta_avg_2 = beta_avg_0.clone();
    let mut beta_avg_3 = beta_avg_0.clone();
    let mut beta_avg_4 = beta_avg_0.clone();


    // Run simulations.
    for i_sim in 1..n_sim+1 {
        y = M3.base.y;
        X = M3.base.X;

        // Generate y.
        rng = nuts.rng;
        if u_nonnormal {
            rv2.ndraw_with_scale(&mut rng, &mut u, &u_scale_vec);
        } else {
            rv1.ndraw(&mut rng, &mut u);
        }
        y.copy(&u);
        dgemv!(1., &X, &beta_full, 1., &mut y);

        // PLR Bayes
        //#[cfg(not(feature="freqs_only"))]
        nuts.model.y = y;
        nuts.model.X = X;
        nuts.rng = rng;
        nuts.init();
        nuts.adapt(&mut nuts_tmp, n_warmup, n_prog_adapt, init_buffer, base_window, term_buffer);
        nuts.draw(&mut nuts_tmp, &mut buffer, n_prog_draw);
        nuts.reset();
        sample.get_rows_as_cols(&buffer, &params[..]);
        for k in 0..K0 {
            let (mean, _) = welford(&mut sample.col_mut(k));
            beta_avg_0[k] += mean;
            sample.col_mut(k).sort_ascend();
            let (_, pvalue) = infer_by_percent(&sample.col(k), H0[k]);
            if pvalue < significance {n_reject_0[k] += 1;}
            //let (_, _, reject) = infer_by_quantile(sample.col_slice(k), H0[k], significance);
            //if reject {n_reject_0[k] += 1;}
        }
        
        // PLR GLS
        M1.base.y = nuts.model.y;
        M1.base.X = nuts.model.X;
        M1.estimate();
        for k in 0..K0 {
            beta_avg_1[k] += M1.base.beta[k];
            let (_, pvalue) = M1.base.t_test(k, H0[k]);
            if pvalue < significance {n_reject_1[k] += 1;}
        }
        M1.base.reset();
        
        // PLR FGLS
        M2.base.y = M1.base.y;
        M2.base.X = M1.base.X;
        M2 = M2.get_V();
        M2.solve_V();
        M2.estimate();
        for k in 0..K0 {
            beta_avg_2[k] += M2.base.beta[k];
            let (_, pvalue) = M2.base.t_test(k, H0[k]);
            if pvalue < significance {n_reject_2[k] += 1;}
        }
        M2.base.reset();
        
        // PLR OLS
        M3.base.y = M2.base.y;
        M3.base.X = M2.base.X;
        M3.estimate();
        for k in 0..K0 {
            beta_avg_3[k] += M3.base.beta[k];
            let (_, pvalue) = M3.base.t_test(k, H0[k]);
            if pvalue < significance {n_reject_3[k] += 1;}
        }
        M3.get_V(&mut V_tmp);
        M3.get_beta_cov_sandwich(&mut V_tmp);
        for k in 0..K0 {
            beta_avg_4[k] += M3.base.beta[k];
            let (_, pvalue) = M3.base.t_test(k, H0[k]);
            if pvalue < significance {n_reject_4[k] += 1;}
        }
        M3.base.reset();
        
        // Print progress
        if i_sim % n_prog_sim == 0 {
            println!("Simulation: {i_sim} / {n_sim}");
        }
    }

    for k in 0..K0 {
        beta_avg_0[k] /= n_sim as f64;
        beta_avg_1[k] /= n_sim as f64;
        beta_avg_2[k] /= n_sim as f64;
        beta_avg_3[k] /= n_sim as f64;
        beta_avg_4[k] /= n_sim as f64;
    }

    // Print results.
    {
        let pr1 = VecPrinter{ name_width: print_name_width, width: print_width, prec: print_prec };
        print!("\nM0: Bayes; M1: GLS; M2: feasible GLS; M3: OLS; M4: sandwich OLS\n\n");
        pr1.print(&params, "parameter index");
        pr1.print(&beta, "true beta");
        pr1.print(&beta_avg_0, "average beta of M0");
        pr1.print(&beta_avg_1, "average beta of M1");
        pr1.print(&beta_avg_2, "average beta of M2");
        pr1.print(&beta_avg_3, "average beta of M3");
        pr1.print(&beta_avg_4, "average beta of M4");
        println!();
        pr1.print(&params, "parameter index");
        pr1.print(&n_simulation, "total simulations");
        pr1.print(&n_reject_0, "rejections of M0");
        pr1.print(&n_reject_1, "rejections of M1");
        pr1.print(&n_reject_2, "rejections of M2");
        pr1.print(&n_reject_3, "rejections of M3");
        pr1.print(&n_reject_4, "rejections of M4");
    }
    // Save results.
    {
        let mut csv = CsvWriter::new(&result_data_path);
        csv.name_width = csv_name_width;
        csv.width = csv_width;
        csv.prec = csv_prec;
        
        csv.write_str("M0: Bayes; M1: GLS; M2: feasible GLS; M3: OLS; M4: sandwich OLS\n\n");
        csv.write_vec(&params, "parameter index");
        csv.write_vec(&beta, "true beta");
        csv.write_vec(&beta_avg_0, "average beta of M0");
        csv.write_vec(&beta_avg_1, "average beta of M1");
        csv.write_vec(&beta_avg_2, "average beta of M2");
        csv.write_vec(&beta_avg_3, "average beta of M3");
        csv.write_vec(&beta_avg_4, "average beta of M4");
        csv.write_newline();
        csv.write_vec(&params, "parameter index");
        csv.write_vec(&n_simulation, "total simulations");
        csv.write_vec(&n_reject_0, "rejections of M0");
        csv.write_vec(&n_reject_1, "rejections of M1");
        csv.write_vec(&n_reject_2, "rejections of M2");
        csv.write_vec(&n_reject_3, "rejections of M3");
        csv.write_vec(&n_reject_4, "rejections of M4");

        println!("\nResult data saved to: {result_data_path}\n");
    }

    let duration = timer.elapsed();
    println!("Time elapsed: {:.2?}", duration);
    //let _ = std::process::Command::new("cmd.exe").arg("/c").arg("pause").status();

}
