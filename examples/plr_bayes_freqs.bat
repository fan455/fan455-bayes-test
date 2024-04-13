plr_bayes_freqs.exe seed0=1000 seed=10000 random_seed=false result_data_path="data/comparison_results.csv" n_warmup=10000 n_sample=10000 n_sim=10 n_prog_sim=10 T=15 N=100 u_rho=0.8 u_scale=10 u_nonnormal=true u_skew=1.5 u_kurt=6 beta=[0,-0.4] beta_other=1 X_loc=[-1,1] X_scale=[1,1] X_rho=[0,0] H0=[0,0] X_path="data/data_sim_X.npy" save_X=false use_saved_X=false
pause

rem u_skew=1.5 u_kurt=6