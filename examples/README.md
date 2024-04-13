This folder contains test examples. `plr_bayes.exe` and `plr_bayes_freqs.exe` are binary programs compiled with these configurations: windows x86-64 platform, Rust (MSVC), OpenBLAS static linking. If you are on 64-bit Windows, you can simply follow these steps: 

**Step 1**: Run `plr simulate data.py` to generate data in `./data/` using the `.npy` format.

**Step 2**: Run `plr_bayes.bat`. This runs the Bayes panel linear regression model with command line arguments passed to `plr_bayes.exe`, reads data from `./data/`, and generates posterior samples in `./data/` using the `.npy` format.

**Step 3**: Run `plr plot.py` to plot the probability density function of generated samples in `./data/`.

Note that the compiled Rust binary file may not run correctly on your computer if, say, your operating system is not 64-bit Windows, or you have not handled the BLAS/LAPACK libraries requirement. What's more, you may want to utilize your CPU's specific SIMD features (in this library, no explicit SIMD code is written, only relying on Rust compiler's auto-vectorization). All these means that you need to compile the Rust code on your own. To do so, you need to modify [fan455_bayes_test/Cargo.toml](../Cargo.toml), [fan455_arrf64/Cargo.toml](../src/fan455_arrf64/Cargo.toml) and [fan455_arrf64/build.rs](../src/fan455_arrf64/build.rs) to fit your platform and use your BLAS and LAPACK libraries. 

In this repository, `fan455_arrf64` is the only rust crate that requires BLAS/LAPACK libraries, which default to [Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). Another commonly used BLAS/LAPACK library is [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS). For other BLAS/LAPACK libraries, you need to set their relevant file paths and names in [fan455_arrf64/build.rs](../src/fan455_arrf64/build.rs). On my computer, MKL runs notably faster than OpenBLAS. But since MKL does not have an open license, the compiled program in this repository uses OpenBLAS. Note that if you use a precompiled version of OpenBLAS, you may need to link to `libopenblas.dll` at runtime.
