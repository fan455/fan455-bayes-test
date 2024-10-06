rem set RUST_BACKTRACE=full
..\bin\bin_plr_bayes_freqs_sim.exe ^
seed0=1000 ^
seed=10000 ^
random_seed=false ^
result_data_dir="data/sim" ^
n_warmup=10000 ^
n_sample=10000 ^
n_sim=1000 ^
n_prog_sim=10 ^
T=10 ^
N=300 ^
u_rho=0.75 ^
u_scale=1 ^
u_nonnormal=false ^
u_skew=1.5 ^
u_kurt=6 ^
beta=[0,0.25] ^
beta_other=1 ^
X_loc=[-1,1] ^
X_scale=[1.75,0.25] ^
X_rho=[0.75,0.75] ^
X_skew=[1.5,2] ^
X_kurt=[5,6.5] ^
X_nonnormal=false ^
H0=[0,0] ^
X_path="data/sim/data_sim_X.npy" ^
save_X=false ^
use_saved_X=false
pause

rem u_skew=1.5 u_kurt=6