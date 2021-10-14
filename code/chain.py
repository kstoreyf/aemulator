import gc
import h5py
import multiprocessing as mp
import numpy as np
import os
import pickle
import scipy
import time

import dynesty
from scipy.linalg import sqrtm
#from schwimmbad import MPIPool

import utils

os.environ["OMP_NUM_THREADS"] = "1"


_param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
_param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

def log_likelihood(theta, param_names, fixed_params, ys_observed, cov):
    s = time.time()
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason
    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    emu_preds = []
    for emu in _emus:
        pred = emu.predict(param_dict)
        emu_preds.append(pred)
    emu_pred = np.hstack(emu_preds)
    diff = (np.array(emu_pred) - np.array(ys_observed))/np.array(ys_observed) #fractional error
    diff = diff.flatten()
    # the solve is a better way to get the inverse
    like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    e = time.time()
    #print("like call: theta=", theta, "; time=", e-s, "s; like =", like)
    return like

def log_likelihood_const(theta, param_names, fixed_params, ys_observed, cov):
    return 1

def prior_transform_hypercube(u, param_names):
    v = np.array(u)
    # the indices of u / param_names that are cosmo
    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    if len(idxs_cosmo)>0:
        dist = scipy.stats.norm.ppf(u[idxs_cosmo])  # convert to standard normal
        v[idxs_cosmo] = np.dot(_hypercube_prior_cov_sqrt, dist) + _hypercube_prior_means

    idxs_hod = [i for i in range(len(param_names)) if param_names[i] in _param_names_hod]
    params_hod = param_names[idxs_hod]
    for i, pname in zip(idxs_hod, params_hod):
        low, high = _hod_bounds[pname]
        if pname=='M_cut':
            low = 11.5
        v[i] = u[i]*(high-low)+low
    return v

def get_cov_means_for_hypercube_prior(idxs_cosmo_vary):
    if len(idxs_cosmo_vary)==0:
        return None, None
    cosmo = np.loadtxt("../tables/cosmology_camb_full.dat")
    cosmo_good = np.delete(cosmo, 23, axis=0) #something bad about this one, zz did this too
    means = np.mean(cosmo_good, axis=0)
    cov = np.cov(cosmo_good.T)
    cov_vary = cov[idxs_cosmo_vary][:,idxs_cosmo_vary] #keep only rows/cols we are varying
    hypercube_prior_cov_sqrt = sqrtm(cov_vary)
    hypercube_prior_means = means[idxs_cosmo_vary]
    return hypercube_prior_cov_sqrt, hypercube_prior_means

#@profile
def run_mcmc(emus, param_names, ys_observed, cov, chain_params_fn, chain_results_fn, fixed_params={},
             n_threads=1, dlogz=0.01, seed=None):

    print("Dynesty sampling (static) - nongen")
    global _emus, _hod_bounds, _hypercube_prior_cov_sqrt, _hypercube_prior_means
    _emus = emus
    _hod_bounds = utils.get_hod_bounds()
    num_params = len(param_names)

    # Get indices of cosmological parameters that will vary; non-listed params are fixed
    idxs_cosmo_vary = [i for i in range(len(_param_names_cosmo)) if _param_names_cosmo[i] in param_names]
    _hypercube_prior_cov_sqrt, _hypercube_prior_means = get_cov_means_for_hypercube_prior(idxs_cosmo_vary)

    prior_args = [param_names]
    logl_args = [param_names, fixed_params, ys_observed, cov]

    # Set chain hyperparameters
    # "The rule of thumb I use is N^2 * a few" (https://github.com/joshspeagle/dynesty/issues/208)
    nlive = max(num_params**2 * 3, 10) #make sure the min is 10
    sample_method = 'rwalk'
    slices = 5 
    walks = 25 
    bound = 'multi' #default
    vol_dec = 0.5 #for multi; default = 0.5, smaller more conservative
    vol_check = 2.0 #for multi; default = 2.0, larger more conservative
    if np.isnan(seed):
        seed = np.random.randint(low=0, high=1000)
    rstate = np.random.RandomState(seed)
    
    # Add info to chain file
    f = h5py.File(chain_params_fn, 'r+')
    f.attrs['nlive'] = nlive
    f.attrs['sample_method'] = sample_method
    f.attrs['slices'] = slices
    f.attrs['walks'] = walks
    f.attrs['bound'] = bound
    f.attrs['vol_dec'] = vol_dec
    f.attrs['vol_check'] = vol_check
    if 'dlogz' not in f.attrs:
        f.attrs['dlogz'] = dlogz
    f.close()

    # Print info
    print("nlive:", nlive)
    print("sample_method:", sample_method)
    print("walks:", walks)
    print("bound:", bound)
    print("seed:", seed)
    print("vol_dec:", vol_dec, "vol_check:", vol_check)
    print("slices:", slices)
    print("dlogz: ", dlogz)
    print("n_threads:", n_threads)

    with mp.Pool(processes=n_threads) as pool:

        queue_size = n_threads
        if n_threads<=1:
            print("running in serial")
            pool, queue_size = None, 1

        print('mp cpu count:', mp.cpu_count())
        print("queue size:", queue_size)

        print("initialize sampler")
        sampler = dynesty.NestedSampler(
            #log_likelihood,
            log_likelihood_const,
            prior_transform_hypercube,
            num_params, logl_args=logl_args, nlive=nlive,
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size,
            sample=sample_method, walks=walks,
            bound=bound, vol_dec=vol_dec, vol_check=vol_check,
            slices=slices)

        # Run sampler
        sampler.run_nested(dlogz=dlogz)
        res = sampler.results
        print(res.summary())

        # save with pickle
        print("Saving results obejct with pickle")
        with open(chain_results_fn, 'wb') as pickle_file:
            pickle.dump(res, pickle_file)
