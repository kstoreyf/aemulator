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
r_dict = {'wp': rlog, 'xi': rlog, 'upf': rlin, 'mcf': rlog, 'xi2': rlog, 'wp80': rlog}
scale_dict = {'wp': ('log', 'log'), 
              'xi': ('log', 'log'), 
              'upf': ('linear', 'log'), 
              'mcf': ('log', 'linear'), 
              'xi2': ('log', 'linear'),
              'wp80': ('log', 'log')} #x, y

#stat_labels = {'wp':r'$w_p$($r_p$)', 'upf':r"P$_U$(r)", 'mcf':r"M($r$)", 'xi':r"$\xi_0$($r$)", 'xi2':r"$\xi_2$($r$)"}
#r_labels = {'wp':r'$r_p (h^{-1}$Mpc)', 'upf':r"$r (h^{-1}$Mpc)", 'mcf':r"$r (h^{-1}$Mpc)", 'xi':r"$r (h^{-1}$Mpc)", 'xi2':r"$r (h^{-1}$Mpc)"}
stat_labels = {'wp':r'$w_\mathrm{p}$($r_\mathrm{p}$)', 
               'upf':r"$P_\mathrm{U}(s)$", 
               'mcf':r"$M(s)$", 
               'xi':r"$\xi_0(s)$", 
               'xi2':r"$\xi_2(s)$",
               'wp80':r'$w_\mathrm{p}$($r_\mathrm{p}$), $\pi_\mathrm{max}=80$'}
r_labels = {'wp':r'$r_\mathrm{p}$ ($h^{-1}\,\mathrm{Mpc}$)', 
            'upf':r"$s$ ($h^{-1}\,\mathrm{Mpc}$)", 
            'mcf':r"$s$ ($h^{-1}\,\mathrm{Mpc}$)", 
            'xi':r"$s$ ($h^{-1}\,\mathrm{Mpc}$)", 
            'xi2':r"$s$ ($h^{-1}\,\mathrm{Mpc}$)",
            'wp80':r'$r_\mathrm{p}$ ($h^{-1}\,\mathrm{Mpc}$)'}

cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
cosmo_withf_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w", "f"]
hod_nof_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f_env", "delta_env", "sigma_env"]
key_param_names = ['Omega_m', 'sigma_8', 'M_sat', 'v_bc', 'v_bs', 'f', 'f_env']
ab_param_names = ["f_env", "delta_env", "sigma_env"]
param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w", \
               "M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
param_names_freorder = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w", "f", \
                        "M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f_env", "delta_env", "sigma_env"]     
param_labels = {'Omega_m': '\Omega_\mathrm{m}',
                'Omega_b': '\Omega_\mathrm{b}',
                'sigma_8': '\sigma_\mathrm{8}',
                'h': 'h',
                'n_s': 'n_\mathrm{s}',
                'N_eff': 'N_\mathrm{eff}',
                'w': 'w',
                'M_sat': 'M_\mathrm{sat}',
                'alpha': r'\alpha',
                'M_cut': 'M_\mathrm{cut}',
                'sigma_logM': '\sigma_{\mathrm{log}M}',
                'v_bc': 'v_\mathrm{bc}',
                'v_bs': 'v_\mathrm{bs}',
                'c_vir': 'c_\mathrm{vir}',
                'f': '\gamma_\mathrm{f}',
                'f_env': 'f_\mathrm{env}',
                'delta_env': '\delta_\mathrm{env}',
                'sigma_env': '\sigma_\mathrm{env}',
                'fsigma8': '\gamma_\mathrm{f} \, f \, \sigma_\mathrm{8}'}

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
                     'xi2': 'George',
                     'wp80': 'George'}
    return emu_name_dict[statistic]

def get_fiducial_emu_scaling(statistic):
    emu_scaling_dict = {'wp': 'log',
                    'xi': 'log',
                    'upf': 'log',
                    'mcf': 'log',
                    'xi2': 'xrsqmean',
                    'wp80': 'log'}
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
    
    chain_results_dir = '/export/sirocco1/ksf293/aemulator/chains/results'
    chain_results_fn = f'{chain_results_dir}/results_{chaintag}.pkl'
    with open(chain_results_fn, 'rb') as pf:
        #print(chain_results_fn, pf)
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

# Intersection point for min and max scales plot

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return None

def find_intersection_point(r, a, b):
    sign = np.sign(a[0]-b[0])
    assert len(a)==len(b), "Lists a and b must be same length!"
    for i in range(1,len(a)):
        sign_prev = sign
        sign = np.sign(a[i]-b[i])
        if sign != sign_prev:
            line_a = line([r[i-1], a[i-1]], [r[i], a[i]])
            line_b = line([r[i-1], b[i-1]], [r[i], b[i]])
            intersect = intersection(line_a, line_b)
            return intersect
    return None


def compute_uncertainties_from_results(results_dict, stat_strs, params, id_pairs):
    uncertainties = np.empty((len(params), len(stat_strs)))
    for p, param in enumerate(params):
        for s, stat_str in enumerate(stat_strs):
            uncertainties_id_pairs = []
            for id_pair in id_pairs:
                uncertainties_id_pairs.append(results_dict[stat_str][tuple(id_pair)][param]['uncertainty'])
            uncertainties[p,s] = np.mean(uncertainties_id_pairs)
    return uncertainties


def print_uncertainty_results_abstract(results_dict, params, id_pairs, prior_dict):
    for j, pn in enumerate(params):    

        print(pn)
        uncertainty_prior = prior_dict[pn]['uncertainty']
        print(f"Prior: {uncertainty_prior:.4f}")
        
        stat_str_wp = 'wp'
        stat_str_standard = 'wp_xi_xi2'
        stat_str_incl_density = 'wp_xi_xi2_upf_mcf'
        stat_strs = [stat_str_wp, stat_str_standard, stat_str_incl_density]
        
        uncertainties_stat_strs = []
        for stat_str in stat_strs:
            uncertainties_id_pairs = []
            for i, id_pair in enumerate(id_pairs):
                uncertainties_id_pairs.append(results_dict[stat_str][tuple(id_pair)][pn]['uncertainty'])
            uncertainty = np.mean(uncertainties_id_pairs)
            uncertainties_stat_strs.append(uncertainty)
            print(f"{stat_str}: {uncertainty:.4f}")

        idx_standard = stat_strs.index(stat_str_standard)
        idx_incl_density = stat_strs.index(stat_str_incl_density)
        uncertainty_change = (uncertainties_stat_strs[idx_incl_density]-uncertainties_stat_strs[idx_standard])/uncertainties_stat_strs[idx_standard]
        increased_precision = -uncertainty_change
        print(f"Increased precision from standard to beyond by: {100*increased_precision:.1f}%")
        print()
