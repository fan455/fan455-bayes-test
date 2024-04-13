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
    cmd_arg!(skew, f64, 0.);
    cmd_arg!(kurt, f64, 3.);
    cmd_arg!(N, usize, 1000);
    cmd_arg!(seed, u64, 1000);

    unknown_cmd_args!();

    let mut rng = MyRng::seed_from_u64(seed);
    let mut x = Arr1::<f64>::new(N);
    let rv = NonNormal::new(skew, kurt+3.);
    rv.ndraw(&mut rng, &mut x);

    let (mean, var) = welford(&x);
    let sd = var.sqrt();

    println!("rv.b = {:12.4}", rv.b);
    println!("rv.c = {:12.4}", rv.c);
    println!("rv.d = {:12.4}", rv.d);

    println!("mean = {:12.4}", mean);
    println!("var = {:12.4}", var);
    println!("sd = {:12.4}", sd);

}
