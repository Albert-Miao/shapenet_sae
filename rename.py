import os
import numpy as np


fnames = os.listdir('8192_npy/val')
test_inds = np.random.choice(range(len(fnames)), 180000, replace=False)


for ind in test_inds:
    os.rename('8192_npy/val/' + fnames[ind], '8192_npy/train/' + fnames[ind])

