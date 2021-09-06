import argparse
import h5py 
import numpy as np
import os
import time

import initialize_chain
import chain
import utils


def main(config_fn):
    chain_params_fn = initialize_chain.main(config_fn)
    run(chain_params_fn)


def run(chain_params_fn):
    
    f = h5py.File(chain_params_fn, 'r+')

    ### data params
    # required
    cosmo = f.attrs['cosmo']
    hod = f.attrs['hod']

    ### emu params
    # required
    statistics = f.attrs['statistic']
    train_tags = f.attrs['traintag']
    #testtags = f.attrs['testtag']
    #errtags = f.attrs['errtag']
    #tags = f.attrs['tag']
    #kernel_names = f.attrs['kernel_name']
    # optional
    logs = f.attrs['log']
    means = f.attrs['mean']
    nhods = f.attrs['nhod']

    ### chain params
    # required
    param_names_vary = f.attrs['param_names_vary']
    chain_results_fn = f.attrs['chain_results_fn']
    # optional
    n_threads = f.attrs['n_threads'] 
    dlogz = float(f.attrs['dlogz'])
    seed = f.attrs['seed']
    cov_fn = f.attrs['cov_fn']

    # Set actual calculated observable
    n_stats = len(statistics)
    ys_observed = []
    for i, statistic in enumerate(statistics):
        testing_dir = f'../../clust/results_aemulus_test_mean/results_{statistic}/'
        _, y_obs = np.loadtxt(f'{testing_dir}/{statistic}_cosmo_{cosmo}_HOD_{hod}_mean.dat', 
                                delimiter=',', unpack=True)
        ys_observed.extend(y_obs)
    f.attrs['ys_observed'] = ys_observed

    # Get true values
    cosmo_param_names, cosmo_params = utils.load_cosmo_params()
    hod_param_names, hod_params = utils.load_hod_params()
    all_param_names = np.concatenate((cosmo_param_names, hod_param_names))
    cosmo_truth = cosmo_params[cosmo]
    hod_truth = hod_params[hod]
    all_params_truth = np.concatenate((cosmo_truth, hod_truth))
    fixed_params = dict(zip(all_param_names, all_params_truth))

    # Remove params that we want to vary from fixed param dict and add true values
    truth = {}
    for pn in param_names_vary:
        truth[pn] = fixed_params[pn]
        fixed_params.pop(pn)
    #can't store dicts in h5py, so make sure truths (for variable params) are in same order as param names
    truths = [truth[pname] for pname in param_names_vary]
    fixed_param_names, fixed_param_values = [], []
    if len(fixed_params)>0:
        #fixed_param_names = list(fixed_params.keys())
        fixed_param_names = np.array(list(fixed_params.keys()), dtype=h5py.string_dtype())
        print(fixed_param_names)
        fixed_param_values = [fixed_params[fpn] for fpn in fixed_param_names]
    f.attrs['true_values'] = truths
    f.attrs['fixed_param_names'] = fixed_param_names
    f.attrs['fixed_param_values'] = fixed_param_values

    # Set up covariance matrix
    if os.path.exists(cov_fn):
        cov = np.loadtxt(cov_fn)
    else:
        raise ValueError(f"Path to covmat {cov_fn} doesn't exist!")
    n_bins_tot = 9 #the covmat should have been constructed with 9 bins per stat
    err_message = f"Cov bad shape! {cov.shape}, but n_bins_tot={n_bins_tot} and n_stats={n_stats}"
    assert cov.shape[0] == n_stats*n_bins_tot and cov.shape[1] == n_stats*n_bins_tot, err_message
    print("Condition number:", np.linalg.cond(cov))
    f.attrs['covariance_matrix'] = cov

    print("Building emulators")
    emus = [None]*n_stats
    for i, statistic in enumerate(statistics):
        Emu = utils.get_emu(train_tags[i])
        
        model_fn = f'../models/model_{statistic}{train_tags[i]}' #emu will add proper file ending
        scaler_x_fn = f'../models/scaler_x_{statistic}{train_tags[i]}.joblib'
        scaler_y_fn = f'../models/scaler_y_{statistic}{train_tags[i]}.joblib'
        err_fn = f"../../clust/covariances/error_aemulus_{statistic}_hod3_test0.dat"

        emu = Emu(statistic, model_fn, scaler_x_fn, scaler_y_fn, err_fn, train_mode=False, test_mode=True)
        emu.load_model()
        emus[i] = emu
        print(f"Emulator for {statistic} built with traintag {train_tags[i]}")

    f.close()

    start = time.time()
    res = chain.run_mcmc(emus, param_names_vary, ys_observed, cov, chain_params_fn, chain_results_fn, fixed_params=fixed_params,
                         n_threads=n_threads, dlogz=dlogz, seed=seed)
    end = time.time()
    print(f"Time: {(end-start)/60.0} min ({(end-start)/3600.} hrs) [{(end-start)/(3600.*24.)} days]")
                                

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_fn', type=str,
                        help='name of config file')
    args = parser.parse_args()
    main(args.config_fn)