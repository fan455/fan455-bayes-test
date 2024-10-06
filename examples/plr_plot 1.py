#import timeit
import numpy as np
from plot import *

np.set_printoptions(precision=4, threshold=25, suppress=True)

# Data path.
folder = 'data'
x_path = f'{folder}/data_beta_sample.npy'

# 读取采样数据
x = np.load(x_path) # F order
plot_pdf(x[:, 0], title='probability density function of variable 0')
plot_pdf(x[:, 1], title='probability density function of variable 0')
