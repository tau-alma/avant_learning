import numpy as np

# Loader params:
loader_lr = 0.985777778
loader_lf = 1.1059
loader_r = 1.05
loader_d = 2*loader_r

# Loader "constraints"
loader_max_beta = 0.65
loader_max_dot_beta = np.deg2rad(33)
loader_max_dot_dot_beta = np.deg2rad(33)
loader_min_v = -1
loader_max_v = 1
loader_max_a = 1

loader_tr = min(loader_lf, loader_lr) / np.tan( loader_max_beta / 2 )