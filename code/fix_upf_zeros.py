import glob
import numpy as np
#import shutil
import os


def main():

    #mock_names = ['aemulus_fmaxmocks_train', 'aemulus_fmaxmocks_test', 'aemulus_fmaxmocks_test_mean']
    mock_names = ['aemulus_fmaxmocks_train', 'aemulus_fmaxmocks_test']
    for mock_name in mock_names:
        edit_upf_zeros(mock_name)
    
def edit_upf_zeros(mock_name):
    result_dir_base = f'/mount/sirocco1/ksf293/clust/results_{mock_name}'
    result_dir = f'{result_dir_base}/results_upf'
    result_dir_raw = f'{result_dir_base}/results_upf_raw'
    
    # copy original dir so we don't mess it up
    if not os.path.exists(result_dir_raw):
        os.rename(result_dir, result_dir_raw)
    os.makedirs(result_dir, exist_ok=True)
    fns = glob.glob(f'{result_dir_raw}/*.dat') 

    for fn in fns:
        rs, ys = np.loadtxt(fn, delimiter=',', unpack=True)
        i_zero = ys==0
        ys[i_zero] = 5e-7 # half of smallest possible, 1e-6, bc avg of 0 or 1 points in sphere

        fn_new = os.path.join(result_dir, os.path.basename(fn))
        res = np.array([rs, ys]).T
        
        # must format as %e or else that last decimal gets lost and stays zero! (not totally sure why)
        np.savetxt(fn_new, res, delimiter=',', fmt=['%f', '%e'])
        if np.sum(i_zero)>0:
            print('fixed a zero in', fn)
            print('new:', fn_new)
            print(ys)

if __name__=='__main__':
    main()