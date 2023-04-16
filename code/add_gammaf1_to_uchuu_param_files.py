import h5py
import numpy as np
import os

check_mode = False
chain_param_dir = '../chains/param_files'
for fn in os.listdir(chain_param_dir):
    if "uchuu" in fn:
        chain_fn = f'{chain_param_dir}/{fn}'
        if check_mode:
            print("Will edit file", chain_fn)
        else:
            print("Editing file", chain_fn)
            fw = h5py.File(chain_fn, 'r+')
            param_names = fw.attrs['param_names_vary']
            true_values = fw.attrs['true_values']
            print(param_names)
            print(true_values)
            if 'f' in param_names:
                idx_gammaf = np.where(param_names=='f')[0][0]
                true_values[idx_gammaf] = 1.0
                print(true_values)
                fw.attrs['true_values'] = true_values
            fw.close()