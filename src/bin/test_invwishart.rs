//#![allow(warnings)]
//#[allow(dead_code, unused_imports, non_snake_case, non_upper_case_globals, nonstandard_style)]

use fan455_util::*;
use fan455_util_macro::*;
use fan455_arrf64::*;
use fan455_bayes::*;
use rand::SeedableRng;

type MyRng = rand_xoshiro::Xoshiro256StarStar;


#[allow(non_snake_case)]
fn main() {

    parse_cmd_args!();

    //cmd_arg!(dim, usize, 5);
    cmd_arg!(df, usize, 30);
    cmd_arg!(T, usize, 5);
    cmd_arg!(N, usize, 1000);
    cmd_arg!(rho, f64, 0.75);
    cmd_arg!(seed, u64, 1000);

    unknown_cmd_args!();

    let mut rng = MyRng::seed_from_u64(seed);
    let cor = get_cor_ar1(T, rho);
    let mut S = cor.clone();
    S.scale((df-T-1) as f64);
    let rv = InvWishart::new(S, df);
    let mut X = Arr2::<f64>::new(T, T);
    let mut B = X.clone();
    let mut M = Arr2::<f64>::new(T, T);

    for _ in 0..N {
        rv.draw_with_buf(&mut rng, &mut X, &mut B);
        M.addassign(&X);
    }
    M.scale(1./N as f64);

    let pr0 = RMatPrinter::new(9, 3);
    pr0.print("cor", &cor);
    pr0.print("M", &M);

}
