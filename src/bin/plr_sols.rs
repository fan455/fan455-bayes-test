//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_util::*;
use fan455_util_macro::*;
use fan455_arrf64::*;
use fan455_bayes::*;


#[allow(non_snake_case)]
fn main() {

    parse_cmd_args!();

    cmd_arg!(y_path, String, "data/data_y.npy".to_string());
    cmd_arg!(X_path, String, "data/data_X.npy".to_string());
    cmd_arg!(T, usize, 10);

    cmd_vec!(x_index, usize, vec![0]);
    cmd_vec!(x_name, String, Vec::new());
    cmd_arg!(y_name, String, "y".to_string());
    cmd_vec_mut!(H0, f64, vec![0.]);

    cmd_arg!(print_name_width, usize, 15);
    cmd_arg!(print_width, usize, 12);
    cmd_arg!(print_prec, usize, 4);
    cmd_arg!(csv_name_width, usize, 15);
    cmd_arg!(csv_width, usize, 12);
    cmd_arg!(csv_prec, usize, 4);
    cmd_arg!(infer_data_path, String, "data/data_inference.csv".to_string());

    unknown_cmd_args!();

    let y = Arr1::<f64>::read_npy(&y_path);
    let X = Arr2::<f64>::read_npy(&X_path);

    let mut model = PlrOls::new(y, X, T).unwrap();
    model.estimate_sandwich();
    let n_param = x_index.len();
    let mut beta_hat: Vec<f64> = vec![0.; n_param];
    let mut H0_pvalue: Vec<f64> = vec![0.; n_param];
    for i in 0..n_param {
        beta_hat[i] = model.base.beta[i];
        let (_, pvalue) = model.base.t_test(x_index[i], H0[i]);
        H0_pvalue[i] = pvalue;
    }

    let pr0 = VecPrinter{ name_width: print_name_width, width: print_width, prec: print_prec };
    println!("y name: {y_name}");
    pr0.print(&x_index, "x index");
    pr0.print_string(&x_name, "x name");
    pr0.print(&beta_hat, "beta estimated");
    pr0.print(&H0, "H0");
    pr0.print(&H0_pvalue, "p-value");

    let mut csv = CsvWriter::new(&infer_data_path);
    csv.name_width = csv_name_width;
    csv.width = csv_width;
    csv.prec = csv_prec;

    csv.write_str(format!("y name: {y_name},\n").as_str());
    csv.write_vec(&x_index, "x index");
    csv.write_vec_string(&x_name, "x name");
    csv.write_vec(&beta_hat, "beta estimated");
    csv.write_vec(&H0, "H0");
    csv.write_vec(&H0_pvalue, "p-value");

    println!("\nInference data saved to: {infer_data_path}");
}
