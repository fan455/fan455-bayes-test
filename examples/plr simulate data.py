# simulation test 01

import numpy as np
from sim import *
from plot import *

np.set_printoptions(precision=4, threshold=25, suppress=True)

folder = 'data'

# Set seed and generator.
seed = 500000
rng = np.random.default_rng(seed)

# Set parameters.
T = 10
N = 1000
K = T + 2
b0 = 30 + rng.uniform(0.9, 1.1, size=T) # time fixed effects
b1 = np.array([2.0, 0.0]) # coefficients
b = np.append(b1, b0) # (K,)

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
u = rng.multivariate_normal(mean_u, cov_u, size=N) # (N,T)
u = np.swapaxes(u, 0, 1) # (T,N)

x1 = np.expand_dims(rng.multivariate_normal(mean_x1, cov_x1, size=N), -1) # (N,T,1)
x2 = np.expand_dims(rng.multivariate_normal(mean_x2, cov_x2, size=N), -1) # (N,T,1)
x0 = np.broadcast_to(np.expand_dims(np.eye(T), 0), (N,T,T)) # (N,T,T)
X = np.concatenate((x1, x2, x0), axis=-1) # (N,T,K)
X = np.swapaxes(X, 0, 1) # (T,N,K)

y = X @ b + u # (T,N,K)@(K,) + (T,N) = (T,N)
X = X.reshape((T*N, K)) # (T*N, K), T*N is ordered N,..,N
X = np.require(X, np.float64, 'F')
y = y.reshape(T*N,) # T*N is ordered N,..,N

# Save data
np.save(f'{folder}/data_y.npy', y)
np.save(f'{folder}/data_X.npy', X)
