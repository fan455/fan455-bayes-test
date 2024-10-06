rem set RUST_BACKTRACE=full
..\bin\bin_plr_bayes_unbalance.exe ^
y_aug_path="data/data_y_aug.npy" ^
X_aug_path="data/data_X_aug.npy" ^
pat_path="data/data_missing_patterns.npy" ^
num_path="data/data_units_each_group.npy" ^
seed=1000 ^
n_prog_adapt=10 ^
n_prog_draw=10 ^
random_seed=false ^
sample_data_path="data/data_beta_sample.npy" ^
run_freqs=true ^
n_warmup=10000 ^
n_sample=10000 ^
x_index=[0,1] ^
H0=[0,0]
pause
