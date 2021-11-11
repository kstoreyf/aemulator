import h5py
import numpy as np
import pickle
from collections import defaultdict

from dynesty import utils as dyfunc


# variables
nbins = 9
rbins = np.logspace(np.log10(0.1), np.log10(50), nbins + 1) # Note the + 1 to nbins
rlog = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
rlin = np.linspace(5, 45, 9)
r_dict = {'wp': rlog, 'xi': rlog, 'upf': rlin, 'mcf': rlog, 'xi2': rlog}
scale_dict = {'wp': ('log', 'log'), 'xi': ('log', 'log'), 'upf': ('linear', 'log'), 'mcf': ('log', 'linear'), 'xi2': ('log', 'linear')} #x, y
#stat_labels = {'wp':r'$w_p$($r_p$)', 'upf':r"P$_U$(r)", 'mcf':r"M($r$)", 'xi':r"$\xi_0$($r$)", 'xi2':r"$\xi_2$($r$)"}
#r_labels = {'wp':r'$r_p (h^{-1}$Mpc)', 'upf':r"$r (h^{-1}$Mpc)", 'mcf':r"$r (h^{-1}$Mpc)", 'xi':r"$r (h^{-1}$Mpc)", 'xi2':r"$r (h^{-1}$Mpc)"}
stat_labels = {'wp':r'$w_p$($r_p$)', 'upf':r"$P_U(s)$", 'mcf':r"$M(s)$", 'xi':r"$\xi_0(s)$", 'xi2':r"$\xi_2(s)$"}
r_labels = {'wp':r'$r_p$ ($h^{-1}$Mpc)', 'upf':"$s$ ($h^{-1}$Mpc)", 'mcf':r"$s$ ($h^{-1}$Mpc)", 'xi':r"$s$ ($h^{-1}$Mpc)", 'xi2':r"$s$ ($h^{-1}$Mpc)"}

cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
key_param_names = ['Omega_m', 'sigma_8', 'M_sat', 'v_bc', 'v_bs', 'f', 'f_env']
ab_param_names = ["f_env", "delta_env", "sigma_env"]
param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w", \
               "M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
param_labels = {'Omega_m': '\Omega_m',
                'Omega_b': '\Omega_b',
                'sigma_8': '\sigma_8',
                'h': 'h',
                'n_s': 'n_s',
                'N_eff': 'N_{eff}',
                'w': 'w',
                'M_sat': 'M_{sat}',
                'alpha': r'\alpha',
                'M_cut': 'M_{cut}',
                'sigma_logM': '\sigma_{logM}',
                'v_bc': 'v_{bc}',
                'v_bs': 'v_{bs}',
                'c_vir': 'c_{vir}',
                'f': '\gamma_f',
                'f_env': 'f_{env}',
                'delta_env': '\delta_{env}',
                'sigma_env': '\sigma_{env}',
                'fsigma8': '\gamma_f \, f \, \sigma_8'}

def get_emu(emu_name):
    import emulator
    emu_dict = {'MLP': emulator.EmulatorMLP,
                'GPFlow': emulator.EmulatorGPFlow, 
                'GPFlowVGP': emulator.EmulatorGPFlowVGP,
                'GPFlowBinned': emulator.EmulatorGPFlowBinned,
                'George': emulator.EmulatorGeorge,
                'GeorgeOrig': emulator.EmulatorGeorgeOrig,
                'PyTorch': emulator.EmulatorPyTorch}
    return emu_dict[emu_name]

def load_cosmo_params():
    # 7 cosmo params
    cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
    cosmo_params = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')
    return cosmo_param_names, cosmo_params

def load_hod_params():
    # 11 cosmo params
    hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
    hod_params = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
    hod_params[:, 0] = np.log10(hod_params[:, 0])
    hod_params[:, 2] = np.log10(hod_params[:, 2])
    return hod_param_names, hod_params

# Prior is the min and max of training set parameters, +/- 10% on either side
def get_hod_bounds():
    hod_bounds = {}
    hod_param_names, hod_params = load_hod_params()
    for pname in hod_param_names:
        pidx = hod_param_names.index(pname)
        vals = hod_params[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        # Add a 10% buffer on either side of training set
        buf = (pmax-pmin)*0.1
        hod_bounds[pname] = [pmin-buf, pmax+buf]
    return hod_bounds

def get_cosmo_bounds():
    cosmo_bounds = {}
    cosmo_param_names, cosmo_params = load_cosmo_params()
    for pname in cosmo_param_names:
        pidx = cosmo_param_names.index(pname)
        vals = cosmo_params[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        # Add a 10% buffer on either side of training set
        buf = (pmax-pmin)*0.1
        cosmo_bounds[pname] = [pmin-buf, pmax+buf]
    return cosmo_bounds

def get_bounds():
    bounds = get_hod_bounds()
    bounds.update(get_cosmo_bounds())
    return bounds

def make_label(statistics):
    if type(statistics) is str:
        return stat_labels[statistics]
    else:
        stats_nice = [stat_labels[s] for s in statistics]
        return ' + '.join(stats_nice)

def get_fiducial_emu_name(statistic):
    emu_name_dict = {'wp': 'George',
                     'xi': 'George',
                     'upf': 'George',
                     'mcf': 'George',
                     'xi2': 'George'}
    return emu_name_dict[statistic]

def get_fiducial_emu_scaling(statistic):
    emu_scaling_dict = {'wp': 'log',
                    'xi': 'log',
                    'upf': 'log',
                    'mcf': 'log',
                    'xi2': 'xrsqmean'}
    return emu_scaling_dict[statistic]

def get_nthreads(n_statistics):
    if n_statistics<=3:
        return 24
    elif n_statistics==4:
        return 18
    elif n_statistics==5:
        return 14
    else:
        print("Don't know how many threads should use for >5 emus, defaulting to 1")
        return 1

def get_bin_indices(scales):
    if scales=='smallest':
        return list(range(0,2))
    if scales=='small':
        return list(range(0,5))
    elif scales=='large':
        return list(range(5,9))
    elif scales=='largest':
        return list(range(7,9))
    elif scales=='all':
        return list(range(0,9))
    else:
        return ValueError(f"Scales mode '{scales}' not recognized! Choose from ['smallest', 'small', 'large', 'largest', 'all']")

def covariance(arrs, zeromean=False):
    arrs = np.array(arrs)
    N = arrs.shape[0]

    if zeromean:
        w = arrs
    else:
        w = arrs - arrs.mean(0)

    outers = np.array([np.outer(w[n], w[n]) for n in range(N)])
    covsum = np.sum(outers, axis=0)
    cov = 1.0/float(N-1.0) * covsum
    return cov

# aka Correlation Matrix
def reduced_covariance(cov):
    cov = np.array(cov)
    Nb = cov.shape[0]
    reduced = np.zeros_like(cov)
    for i in range(Nb):
        ci = cov[i][i]
        for j in range(Nb):
            cj = cov[j][j]
            reduced[i][j] = cov[i][j]/np.sqrt(ci*cj)
    return reduced

def correlation_to_covariance(corr, cov_orig):
    corr = np.array(corr)
    Nb = corr.shape[0]
    cov = np.zeros_like(corr)
    for i in range(Nb):
        ci = cov_orig[i][i]
        for j in range(Nb):
            cj = cov_orig[j][j]
            cov[i][j] = corr[i][j]*np.sqrt(ci*cj)
    return cov


### for statistical results plots

def get_medians(samples_equal):
    medians = np.median(samples_equal, axis=0)
    return medians

# now for testing
def get_means(samples_equal):
    # checked that same as dyfunc.mean_and_cov(samples, weights)
    # and np.std(x, axis=0) = np.sqrt(np.diag(cov))
    means = np.mean(samples_equal, axis=0)
    return means

def get_posterior_maxes(samples_equal, param_names):
    samps = MCSamples(names=param_names)
    samps.setSamples(samples_equal)
    maxes = []
    for i, pn in enumerate(param_names):
        xvals = np.linspace(min(samples_equal[:,i]), max(samples_equal[:,i]), 1000)
        dens = samps.get1DDensity(pn)   
        probs = dens(xvals)
        posterior_max = xvals[np.argmax(probs)]
        maxes.append(posterior_max)
    return maxes

def get_uncertainties(samples_equal):
    lowers = np.percentile(samples_equal, 16, axis=0)
    uppers = np.percentile(samples_equal, 84, axis=0)
    uncertainties = (uppers-lowers)/2.0
    return uncertainties

def bootstrap(data, function, n_resamples=1000):
    result_arr = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = np.random.choice(data, size=n_resamples)
        result_arr[i] = function(sample)
    return result_arr


def construct_results_dict(chaintag):
    
    chain_fn = f'../chains/param_files/chain_params_{chaintag}.h5'
    fw = h5py.File(chain_fn, 'r')
    param_names = fw.attrs['param_names_vary']
    truths = fw.attrs['true_values']
    fw.close()
    
    chain_results_fn = f'../chains/results/results_{chaintag}.pkl'
    with open(chain_results_fn, 'rb') as pf:
        res = pickle.load(pf)
        samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
        samples_equal = dyfunc.resample_equal(samples, weights)
        
    # add fsigma8
    idx_Omega_m = np.where(param_names=='Omega_m')[0][0]
    idx_gamma_f = np.where(param_names=='f')[0][0]
    idx_sigma_8 = np.where(param_names=='sigma_8')[0][0]
    f = samples_equal[:,idx_Omega_m]**0.55
    fsigma8 = f*samples_equal[:,idx_gamma_f]*samples_equal[:,idx_sigma_8]
    samples_equal = np.hstack((samples_equal, np.atleast_2d(fsigma8).T))
    param_names = np.append(param_names, 'fsigma8')
    fsigma8_true = truths[idx_Omega_m]**0.55 * truths[idx_gamma_f] * truths[idx_sigma_8]
    truths = np.append(truths, fsigma8_true)
        
    means = get_means(samples_equal)
    medians = get_medians(samples_equal)
    uncertainties = get_uncertainties(samples_equal)
    
    result_dict_single = {}
    for j, pn in enumerate(param_names):
        sub_dict = defaultdict(dict)
        sub_dict['mean'] = means[j]
        #sub_dict['max'] = maxes[j]
        sub_dict['median'] = medians[j]
        sub_dict['uncertainty'] = uncertainties[j]
        sub_dict['truth'] = truths[j]
        result_dict_single[pn] = sub_dict
        
    return result_dict_single
