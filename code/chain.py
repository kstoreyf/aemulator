import gc
import h5py
import multiprocessing as mp
import numpy as np
import os
import pickle
import scipy
import scipy.stats
import time

import dynesty
from scipy.linalg import sqrtm
#from schwimmbad import MPIPool

import utils

os.environ["OMP_NUM_THREADS"] = "1"


# _param_names_cosmo = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
# _param_names_hod = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

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
    # print('theta:', theta)
    # print('ys_obs:', ys_observed)
    # print('ys_pred:', emu_pred)
    diff = (np.array(emu_pred) - np.array(ys_observed))/np.array(ys_observed) #fractional error
    diff = diff.flatten()
    # print('cov', cov)
    # print('diff:', diff)

    # the solve is a better way to get the inverse
    like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    e = time.time()
    #print("like call: theta=", theta, "; time=", e-s, "s; like =", like)
    return like


def log_likelihood_ellprior(theta, param_names, fixed_params, ys_observed, cov):
    s = time.time()
    theta = np.array(theta).flatten() #theta looks like [[[p]]] for some reason

    # return -inf for params outside of multivar ellipsoid
    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    delta_p = theta[idxs_cosmo] - _hypercube_prior_means
    t_forttest = np.dot(delta_p, np.linalg.solve(_hypercube_prior_cov, delta_p))
    t_thresh = 12
    print('t:', t_forttest)
    if t_forttest > t_thresh:
        return -1e100

    #idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    #if len(idxs_cosmo)>0:
        #dist = scipy.stats.norm.ppf(u[idxs_cosmo])  # convert to standard normal
        #v[idxs_cosmo] = np.dot(_hypercube_prior_cov_sqrt, dist) + _hypercube_prior_means
        #np.dot(_hypercube_prior_cov_sqrt, dist) + _hypercube_prior_means

    param_dict = dict(zip(param_names, theta)) #weirdly necessary for Powell minimization
    param_dict.update(fixed_params)
    emu_preds = []
    for emu in _emus:
        pred = emu.predict(param_dict)
        emu_preds.append(pred)
    emu_pred = np.hstack(emu_preds)
    #print('theta:', theta)
    #print('ys_obs:', ys_observed)
    #print('ys_pred:', emu_pred)
    diff = (np.array(emu_pred) - np.array(ys_observed))/np.array(ys_observed) #fractional error
    diff = diff.flatten()
    # the solve is a better way to get the inverse
    like = -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    e = time.time()
    print("like call: theta=", theta, "; time=", e-s, "s; like =", like)
    return like

def log_likelihood_const(theta, param_names, fixed_params, ys_observed, cov):
    return 1

def prior_transform_hypercube(u, param_names):
    v = np.array(u)
    # the indices of u / param_names that are cosmo

    # For cosmology, use a multi-dimensional Gaussian prior defined by 
    # the mean and covariance of the cosmology training set parameter space
    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    if len(idxs_cosmo)>0:
        dist = scipy.stats.norm.ppf(u[idxs_cosmo])  # convert to standard normal
        v[idxs_cosmo] = np.dot(_hypercube_prior_cov_sqrt, dist) + _hypercube_prior_means

    # For the HOD and assembly bias parameters, as well as gammaf, we use a uniform prior 
    # with a range given in Table 3 of Zhai+2022, with an additional bound on Mcut
    idxs_hod = [i for i in range(len(param_names)) if param_names[i] in _param_names_hod]
    params_hod = param_names[idxs_hod]
    for i, pname in zip(idxs_hod, params_hod):
        low, high = _hod_bounds[pname]
        if pname=='M_cut':
            low = 11.5
        v[i] = u[i]*(high-low)+low
    return v


def prior_transform_flat(u, param_names):
    v = np.array(u)
    # the indices of u / param_names that are cosmo

    # For cosmology, try a uniform prior given by training set parameter space
    idxs_cosmo = [i for i in range(len(param_names)) if param_names[i] in _param_names_cosmo]
    params_cosmo = param_names[idxs_cosmo]
    for i, pname in zip(idxs_cosmo, params_cosmo):
        low, high = _cosmo_bounds[pname]
        v[i] = u[i]*(high-low)+low

    # For the HOD and assembly bias parameters, as well as gammaf, we use a uniform prior 
    # with a range given in Table 3 of Zhai+2022, with an additional bound on Mcut
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
    hypercube_prior_cov = cov[idxs_cosmo_vary][:,idxs_cosmo_vary] #keep only rows/cols we are varying
    hypercube_prior_means = means[idxs_cosmo_vary]
    return hypercube_prior_cov, hypercube_prior_means


#@profile
def run_mcmc(emus, param_names, ys_observed, cov, chain_params_fn, chain_results_fn, mock_name_hod, fixed_params={},
             n_threads=1, dlogz=0.01, seed=None, data_name=''):

    print("Dynesty sampling (static) - nongen")
    global _emus, _hod_bounds, _cosmo_bounds
    global _param_names_cosmo, _param_names_hod
    global _hypercube_prior_cov, _hypercube_prior_cov_sqrt, _hypercube_prior_means
    
    _emus = emus
    _param_names_cosmo, _ = utils.load_cosmo_params(data_name)
    _param_names_hod, _ = utils.load_hod_params(mock_name_hod, data_name=data_name)
    _hod_bounds = utils.get_hod_bounds(mock_name_hod)
    _cosmo_bounds = utils.get_cosmo_bounds(mock_name_hod)
    num_params = len(param_names)

    # Get indices of cosmological parameters that will vary; non-listed params are fixed
    idxs_cosmo_vary = [i for i in range(len(_param_names_cosmo)) if _param_names_cosmo[i] in param_names]
    _hypercube_prior_cov, _hypercube_prior_means = get_cov_means_for_hypercube_prior(idxs_cosmo_vary)
    _hypercube_prior_cov_sqrt = sqrtm(_hypercube_prior_cov)

    prior_args = [param_names]
    logl_args = [param_names, fixed_params, ys_observed, cov]

    # Set chain hyperparameters
    # "The rule of thumb I use is N^2 * a few" (https://github.com/joshspeagle/dynesty/issues/208)
    nlive = max(num_params**2 * 3, 10) #make sure the min is 10
    sample_method = 'rwalk'
    slices = 5 
    walks = 25 
    bound = 'single' #TRYING THIS
    #bound = 'multi' #default
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

    if data_name=='prior':
        likelihood_func = log_likelihood_const
    else:
        likelihood_func = log_likelihood


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

        #print("USING FLAT COSMO PRIOR; ARE U SURE?")
        sampler = dynesty.NestedSampler(
            likelihood_func,
            prior_transform_hypercube,
            num_params, logl_args=logl_args, nlive=nlive,
            ptform_args=prior_args, rstate=rstate,
            pool=pool, queue_size=queue_size,
            sample=sample_method, walks=walks,
            bound=bound, vol_dec=vol_dec, vol_check=vol_check,
            slices=slices)

        # Run sampler
        sampler.run_nested(dlogz=dlogz,
                           print_progress=False)
        res = sampler.results
        print(res.summary())

        # save with pickle
        print("Saving results object to", chain_results_fn)
        with open(chain_results_fn, 'wb') as pickle_file:
            pickle.dump(res, pickle_file)
