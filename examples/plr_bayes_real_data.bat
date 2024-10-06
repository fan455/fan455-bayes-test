rem set RUST_BACKTRACE=full
D:\self_research\rust_target\fan455_bayes_test\release\bin_plr_bayes_unbalance.exe ^
y_aug_path="data/real_data/y_aug.npy" ^
X_aug_path="data/real_data/X_aug.npy" ^
pat_path="data/real_data/missing_patterns.npy" ^
num_path="data/real_data/units_each_group.npy" ^
seed=1000 ^
n_prog_adapt=20 ^
n_prog_draw=20 ^
random_seed=false ^
sample_data_path="data/data_beta_sample.npy" ^
run_freqs=true ^
n_warmup=10000 ^
n_sample=10000 ^
y_name="ROE" ^
x_name=["Dual","Size","Lev","LDebt","Top3","Cash","Age","TobinQ"] ^
x_index=[0,1,2,3,4,5,6,7] ^
H0=[0,0,0,0,0,0,0,0]
pause
