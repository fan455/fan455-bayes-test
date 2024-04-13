#import timeit
import numpy as np
from plot import *

# Data path.
folder = 'data'
x_path = f'{folder}/data_beta.npy'

# 读取采样数据
x = np.load(x_path)
plot_pdf(x[:, 0], title='pdf of x0') # F order
plot_pdf(x[:, 1], title='pdf of x1') # F order




