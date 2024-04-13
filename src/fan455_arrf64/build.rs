fn main() {

    #[cfg(all(feature="x86-64-windows", feature="mkl"))]
    {
        println!("cargo:rustc-link-search=native=D:/Intel/oneAPI/mkl/2024.0/lib");
        println!("cargo:rustc-link-lib=static=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=static=mkl_sequential");
        println!("cargo:rustc-link-lib=static=mkl_core");
    }
    
    #[cfg(all(feature="x86-64-windows", feature="openblas"))]
    {
        println!("cargo:rustc-link-search=native=D:/sofs/openblas/lib");
        println!("cargo:rustc-link-lib=static=libopenblas");
    }
}