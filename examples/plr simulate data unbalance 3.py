# simulation test 01

import numpy as np
from sim import *
from plot import *

np.set_printoptions(precision=4, threshold=25, suppress=True)

folder = 'data'

# Set seed and generator.
seed = 1000
rng = np.random.default_rng(seed)

# Set parameters.
T = 5
N = 100
K = T + 2
b0 = 30 + rng.uniform(0.9, 1.1, size=T) # time fixed effects
b1 = np.array([2.0, 0.0]) # coefficients
b = np.append(b1, b0) # (K,)

# Missing settings
N0 = 50
N1 = N - N0
pat_1_idx = [3, 4]
T1 = len(pat_1_idx)
pat_0 = list(1 for _ in range(T))
pat_1 = list(int(i in pat_1_idx) for i in range(T))
pat = np.asfortranarray(np.array([pat_0, pat_1], dtype=np.uint8).swapaxes(0, 1))
pat_unit = np.asfortranarray(np.array([[True, False], [True, True]], dtype=np.uint8).swapaxes(0, 1))
num_units_each_group = np.array([N0, N1], dtype=np.uint64)
time_idx = np.arange(T)

mean_u = np.zeros(T)
cor_u = get_cor_ar1(T, r=0.5)
cov_u = cor2cov(cor_u, rng.uniform(0.9, 1.1, size=T))
cov_u = inv_wishart(cov_u, df=30).draw(rng=rng)

mean_x1 = 10 + rng.uniform(0, 1, size=T)
cor_x1 = get_cor_ar1(T, r=0.25)
cov_x1 = cor2cov(cor_x1, rng.uniform(0.9, 1.1, size=T))
cov_x1 = inv_wishart(cov_x1, df=30).draw(rng=rng)

mean_x2 = -10 + rng.uniform(0, 1, size=T)
cor_x2 = get_cor_ar1(T, r=0.75)
cov_x2 = cor2cov(cor_x2, rng.uniform(0.9, 1.1, size=T))
cov_x2 = inv_wishart(cov_x2, df=30).draw(rng=rng)

# Generate data.
time_mask = ~np.isin(time_idx, pat_1_idx)
u = rng.multivariate_normal(mean_u, cov_u, size=N) # (N,T)
u = np.asfortranarray(u) # (N,T)
u[N0:, time_mask ] = 0.
u = u.reshape((N*T,), order='F') # (N*T,)

x1 = np.expand_dims(rng.multivariate_normal(mean_x1, cov_x1, size=N), -1) # (N,T,1)
x2 = np.expand_dims(rng.multivariate_normal(mean_x2, cov_x2, size=N), -1) # (N,T,1)
x0 = np.broadcast_to(np.expand_dims(np.eye(T), 0), (N,T,T)) # (N,T,T)
X = np.concatenate((x1, x2, x0), axis=-1) # (N,T,K)
X = np.asfortranarray(X) # (N,T,K)
X[N0:, time_mask, :] = 0.
X = np.asfortranarray(X.reshape((N*T, K), order='F')) # (N*T,K)

y = X @ b + u


# Save data
np.save(f'{folder}/data_y_aug.npy', y)
np.save(f'{folder}/data_X_aug.npy', X)
np.save(f'{folder}/data_missing_patterns.npy', pat)
np.save(f'{folder}/data_units_each_group.npy', num_units_each_group)
