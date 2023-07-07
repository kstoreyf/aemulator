import h5py
import matplotlib
import numpy as np
import pickle
import os
from collections import defaultdict
from matplotlib import pyplot as plt

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

def load_cosmo_params_mock(mock_name):
    cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
    if mock_name=='uchuu':
        # from http://skiesanduniverses.org/Simulations/Uchuu/
        # not sure about N_eff! pulled from Planck2015 table 5, rightmost col (https://arxiv.org/pdf/1502.01589.pdf)
        # w??
        cosmo_params = 0.3089, 0.0486, 0.8159, 0.6774, 0.9667, 3.04, -1
    return cosmo_param_names, cosmo_params

def load_hod_params_mock(mock_name):
    hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
    if mock_name=='uchuu':
        # made using SHAM! but we do know gamma_f=1
        hod_params = [float("NaN")]*len(hod_param_names)
        #idx_f = hod_params.index("f")
        idx_f = hod_param_names.index("f")
        hod_params[idx_f] = 1.0
    return hod_param_names, hod_params

def load_cosmo_params(mock_name):
    # 7 cosmo params
    cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
    if mock_name=='uchuu':
        # from http://skiesanduniverses.org/Simulations/Uchuu/
        # not sure about N_eff! pulled from Planck2015 table 5, rightmost col (https://arxiv.org/pdf/1502.01589.pdf)
        # w??
        cosmo_params = [0.3089, 0.0486, 0.8159, 0.6774, 0.9667, 3.04, -1]
    elif 'test' in mock_name:
        cosmo_fn = '../tables/cosmology_camb_test_box_full.dat'
        cosmo_params = np.loadtxt(cosmo_fn)
    elif 'train' in mock_name:
        cosmo_fn = '../tables/cosmology_camb_full.dat'
        cosmo_params = np.loadtxt(cosmo_fn)
    else: 
        raise ValueError(f'Mock name {mock_name} not recognized!')
    
    return cosmo_param_names, cosmo_params

def load_hod_params(mock_name, data_name=None):
    # 11 hod params, + f_max
    
    hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
    if 'fmaxmocks' in mock_name:
        hod_param_names.append("f_max")

    if data_name=='uchuu':
        # made using SHAM! but we do know gamma_f=1
        hod_params = [float("NaN")]*len(hod_param_names)
        #idx_f = hod_params.index("f")
        idx_f = hod_param_names.index("f")
        hod_params[idx_f] = 1.0
        hod_params = np.array(hod_params)
        return hod_param_names, hod_params
    
    if mock_name=='aemulus_test':
        hod_fn = '../tables/HOD_test_np11_n1000_new_f_env.dat'
    elif mock_name=='aemulus_Msatmocks_test':
        hod_fn = '/mount/sirocco2/zz681/emulator/CMASSLOWZ_Msat/test_mocks/HOD_test_np11_n5000_new_f_env_Msat.dat'
    elif mock_name=='aemulus_Msatmocks_train':
        hod_fn = '/mount/sirocco2/zz681/emulator/CMASSLOWZ_Msat/training_mocks/HOD_design_np11_n5000_new_f_env_Msat.dat'
    elif mock_name=='aemulus_fmaxmocks_test':
        hod_fn = '/mount/sirocco1/zz681/emulator/CMASSLOWZ_Msat_fmax_new/test_mocks/HOD_test_np11_n5000_new_f_env_Msat_fmax_new.dat'
    elif mock_name=='aemulus_fmaxmocks_train':
        hod_fn = '/mount/sirocco1/zz681/emulator/CMASSLOWZ_Msat_fmax_new/training_mocks/HOD_design_np11_n5000_new_f_env_Msat_fmax_new.dat'
    else: 
        raise ValueError(f'Mock name {mock_name} not recognized!')
    hod_params = np.loadtxt(hod_fn)
    # # Convert these columns (0: M_sat, 2: M_cut) to log to reduce dynamic range
    hod_params[:, 0] = np.log10(hod_params[:, 0])
    hod_params[:, 2] = np.log10(hod_params[:, 2])
    assert len(hod_param_names)==hod_params.shape[1], "Lengths of HOD names and params don't line up!"
    return hod_param_names, hod_params

# Prior is the min and max of training set parameters, +/- 10% on either side
def get_hod_bounds(mock_name):
    hod_bounds = {}
    hod_param_names, hod_params = load_hod_params(mock_name)
    for pname in hod_param_names:
        pidx = hod_param_names.index(pname)
        vals = hod_params[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        # Add a 10% buffer on either side of training set
        #buf = (pmax-pmin)*0.1
        #hod_bounds[pname] = [pmin-buf, pmax+buf]
        hod_bounds[pname] = [pmin, pmax]
    return hod_bounds

def get_cosmo_bounds(mock_name):
    cosmo_bounds = {}
    cosmo_param_names, cosmo_params = load_cosmo_params(mock_name)
    for pname in cosmo_param_names:
        pidx = cosmo_param_names.index(pname)
        vals = cosmo_params[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        # Add a 10% buffer on either side of training set
        # buf = (pmax-pmin)*0.1
        # cosmo_bounds[pname] = [pmin-buf, pmax+buf]
        cosmo_bounds[pname] = [pmin, pmax]
    return cosmo_bounds

def get_bounds(mock_name):
    bounds = get_hod_bounds(mock_name)
    bounds.update(get_cosmo_bounds(mock_name))
    return bounds

def make_label(statistics):
    if type(statistics) is str:
        statistics = statistics.split('_')
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

    # TODO: doesn't this seem like opposite of zeromean?? uh...
    # maybe i meant we *assume* the data has zero mean already so we 
    # don't correct for it - like with fractional errors!
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
    
    #chain_results_dir = '/export/sirocco1/ksf293/aemulator/chains/results'
    chain_results_dir = '/mount/sirocco1/ksf293/aemulator/chains/results'
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
    # print(get_uncertainties(f))
    # print(get_uncertainties(f*samples_equal[:,idx_sigma_8]))
    fsigma8 = f*samples_equal[:,idx_gamma_f]*samples_equal[:,idx_sigma_8]
    # print(get_uncertainties(f*samples_equal[:,idx_gamma_f]))
    # print(get_uncertainties(samples_equal[:,idx_gamma_f]*samples_equal[:,idx_sigma_8]))
    # print(get_uncertainties(f*samples_equal[:,idx_gamma_f]*samples_equal[:,idx_sigma_8]))
    samples_equal = np.hstack((samples_equal, np.atleast_2d(fsigma8).T))
    param_names = np.append(param_names, 'fsigma8')
    if truths.size>0: # is not empty (e.g. in prior case)
        fsigma8_true = truths[idx_Omega_m]**0.55 * truths[idx_gamma_f] * truths[idx_sigma_8]
        truths = np.append(truths, fsigma8_true)
    # print()
    # print(get_uncertainties(samples_equal[:,idx_gamma_f]))
    # print(get_uncertainties(samples_equal[:,idx_sigma_8]))
    # print(get_uncertainties(samples_equal[:,idx_gamma_f]*samples_equal[:,idx_sigma_8]))

    # trying this 
    samples_f = samples[:,idx_Omega_m]**0.55
    samples_fsigma8 = samples_f*samples[:,idx_gamma_f]*samples[:,idx_sigma_8]
    # weights are not by parameter, just one for each point/sample, so 
    # not sure if need to do anything to them?
    fsigma8_equal = dyfunc.resample_equal(samples_fsigma8, weights)
    # print()
    # print(get_uncertainties(fsigma8_equal))

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
        if truths.size>0: # is not empty (e.g. in prior case)
            sub_dict['truth'] = truths[j]
        result_dict_single[pn] = sub_dict
        
    #return result_dict_single, param_names, samples_equal
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


def load_statistics(statistic, mock_name, id_pairs, verbose=False):
    result_dir_base = f'/mount/sirocco1/ksf293/clust/results_{mock_name}'
    result_dir = f'{result_dir_base}/results_{statistic}'
    
    if 'test' in mock_name and 'mean' not in mock_name:
        raise KeyError("Loading statistics for individual test boxes (rather than mean) not supported")
    if 'mean' in mock_name:
        clust_tag = '_mean'
    else:
        clust_tag = '_test_0'
    
    r_arr = []
    y_train_arr = []
    for id_pair in id_pairs:
        cosmo_id, hod_id = id_pair
        fn_y_train = f'{statistic}_cosmo_{cosmo_id}_HOD_{hod_id}{clust_tag}.dat'
        r_vals, y_train = np.loadtxt(os.path.join(result_dir, fn_y_train), delimiter=',', unpack=True)
        r_arr.append(r_vals)
        y_train_arr.append(y_train)
        if verbose and statistic!='xi2' and np.any(y_train <= 0):
            print(id_pair, " has 0s or negatives in data vector!")
    r_arr = np.array(r_arr)
    y_train_arr = np.array(y_train_arr)
    return r_arr, y_train_arr


def load_id_pairs_train(mock_name_train, train_tag=''):
    ## ID values (cosmo and hod numbers)
    if 'nclosest' in train_tag:
        for tag in train_tag.split('_'):
            if 'nclosest' in tag:
                fn_train = f'../tables/id_pairs_train_{tag}.txt'
    else:
        fn_train = '../tables/id_pairs_train.txt'
    id_pairs_train = np.loadtxt(fn_train, delimiter=',', dtype=int)
    print("original number of training ID pairs:", len(id_pairs_train))
    # Remove models that give zero or negative clustering statistic values
    # for all of the statistics (even the ones that are ok)
    if mock_name_train=='aemulus_Msatmocks_train' and 'nclosest' not in train_tag:
        bad_id_indices = [1296, 1335] #effectively the HOD IDs
        id_pairs_train = np.delete(id_pairs_train, bad_id_indices, axis=0)
        print("Deleted bad ID pairs with indices", bad_id_indices)
    if mock_name_train=='aemulus_fmaxmocks_train':
        bad_id_indices = [2228, 3208] #effectively the HOD IDs 
        id_pairs_train = np.delete(id_pairs_train, bad_id_indices, axis=0)
        print("Deleted bad ID pairs with indices", bad_id_indices)
    n_train = len(id_pairs_train)
    print("N train:", n_train)
    return id_pairs_train 


def load_id_pairs_test(train_tag=''):
    ### ID values (cosmo and hod numbers)
    if 'nclosest' in train_tag:
        for tag in train_tag.split('_'):
            if 'nclosest' in tag:
                fn_test = f'../tables/id_pairs_test_{tag}.txt'
    else:
        fn_test = '../tables/id_pairs_test.txt'
    id_pairs_test = np.loadtxt(fn_test, delimiter=',', dtype=int)
    return id_pairs_test 
    

def get_chisqs(ys_to_compare, y_arr, variances):
    # this works, checked - see https://stackoverflow.com/a/72299599
    ys_to_compare_flat = np.hstack(ys_to_compare)
    y_arr_flat = np.hstack(y_arr) 
    chisqs = []
    for j in range(len(y_arr_flat)):
        chisqs.append( chi2(ys_to_compare_flat, y_arr_flat[j], variances) )
    # i_min = np.argmin(chisqs)
    # print(ys_to_compare_flat)
    # print(y_arr_flat[i_min])
    # print(variances)   
    # print(chisqs[i_min])
    return np.array(chisqs)


def get_closest_models(ys_to_compare, y_arr, variances, n_closest=2000):
    chisqs = get_chisqs(ys_to_compare, y_arr, variances)
    # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
    idx = np.argpartition(chisqs, n_closest)
    idxs_closest = idx[:n_closest]
    chisq_max_of_chosen = np.max(chisqs[idxs_closest])
    print(np.min(chisqs), np.max(chisqs), chisq_max_of_chosen)
    return idxs_closest, chisq_max_of_chosen


def get_models_within_chi2(ys_to_compare, y_arr, variance_arr, chisq_thresh):
    chisqs_mean = get_chisqs(ys_to_compare, y_arr, variance_arr)
    idxs_within_err = np.where(chisqs_mean < chisq_thresh)[0]
    return idxs_within_err


def load_emus(chaintag):

    base_dir = '/export/sirocco1/ksf293/aemulator'
    
    chain_fn = f'../chains/param_files/chain_params_{chaintag}.h5'
    f = h5py.File(chain_fn, 'r')

    statistics = f.attrs['statistics']
    emu_names = f.attrs['emu_names']
    scalings = f.attrs['scalings']
    
    n_stats = len(statistics)

    # optional
    if 'train_tags_extra' in f.attrs:
        train_tags_extra = f.attrs['train_tags_extra']
    else:
        train_tags_extra = ['']*len(statistics)
    
    if 'mock_name_train' in f.attrs:
        mock_name_train = f.attrs['mock_name_train']
    else:
        mock_name_train = 'aemulus_train'

    if 'mock_name_test' in f.attrs:
        mock_name_test = f.attrs['mock_name_test']
    else:
        mock_name_test = 'aemulus_test'
        
    if 'bins' in f.keys():
        bins = [list(b_arr) for b_arr in f['bins']]
    #elif not np.isnan(f.attrs['bins']):
    #    bins = f.attrs['bins']
    else:
        bins = [np.arange(9) for _ in range(n_stats)]
    print(bins)
    f.close()
        
    emus = [None]*n_stats
    mock_tag_train = '_'+mock_name_train
    for i, statistic in enumerate(statistics):
    
        # load emu
        Emu = get_emu(emu_names[i])

        train_tag = f'_{emu_names[i]}_{scalings[i]}{train_tags_extra[i]}'
        model_fn = f'{base_dir}/models/model_{statistic}{train_tag}' #emu will add proper file ending
        scaler_x_fn = f'{base_dir}/models/scaler_x_{statistic}{train_tag}.joblib'
        scaler_y_fn = f'{base_dir}/models/scaler_y_{statistic}{train_tag}.joblib'

        err_fn = f"../covariances/stdev_{mock_name_test}_{statistic}_hod3_test0.dat"

        emu = Emu(statistic, scalings[i], model_fn, scaler_x_fn, scaler_y_fn, err_fn,
                  bins=bins[i], predict_mode=True, mock_tag_train=mock_tag_train)
        emu.load_model()
        emus[i] = emu
    
    emu_dict = dict(zip(statistics, emus))
    return emu_dict


def get_best_fit(chaintag, emu_dict, data_tag_aem=None, return_pred_on_true_params=False):
    chain_results_dir = '/export/sirocco1/ksf293/aemulator/chains/results'
    chain_results_fn = f'{chain_results_dir}/results_{chaintag}.pkl'
    with open(chain_results_fn, 'rb') as pf:
        res = pickle.load(pf)
        samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
        samples_equal = dyfunc.resample_equal(samples, weights)

    # getting really bad results with mode! idk why ??
    #params_best = scipy.stats.mode(samples_equal, axis=0)[0][0] #0 is mode, 1 is counts. 2nd zero is array
    params_best = get_medians(samples_equal)
    
    chain_fn = f'../chains/param_files/chain_params_{chaintag}.h5'
    f = h5py.File(chain_fn, 'r')
    param_names = f.attrs['param_names_vary']
    params_true = f.attrs['true_values']
    
    param_dict = dict(zip(param_names, params_best)) 
    print(param_names)
    print(params_best)

    statistics = f.attrs['statistics']
    if 'data_name' in f.attrs:
        data_name = f.attrs['data_name']
    else:
        data_name = 'aemulus_test'
    f.close()
    rs = []
    ys_true = []
    ys_pred_best = []
    ys_pred_on_truth = []
    for i, statistic in enumerate(statistics):
    
        emu = emu_dict[statistic]
        y_pred = emu.predict(param_dict)
        ys_pred_best.append(y_pred)

        if 'aemulus' in data_name:
            result_dir=f"/home/users/ksf293/clust/results_{data_name}_mean/results_{statistic}"
            fn_stat=f"{result_dir}/{statistic}_{data_tag_aem}_mean.dat"
        else: 
            result_dir=f"/home/users/ksf293/clust/results_{data_name}/results_{statistic}"
            fn_stat=f"{result_dir}/{statistic}_{data_name}.dat"
            
        r, y_true = np.loadtxt(fn_stat, delimiter=',', unpack=True)
        rs.append(r)
        ys_true.append(y_true)
        # can only do this for aem bc uses same hod params as emu
        if 'aemulus' in data_name and return_pred_on_true_params:
            param_dict_true = dict(zip(param_names, params_true)) 
            y_pred_on_truth = emu.predict(param_dict_true)
            ys_pred_on_truth.append(y_pred_on_truth)

    if return_pred_on_true_params:
        return rs, ys_true, ys_pred_best, ys_pred_on_truth
    else:
        return rs, ys_true, ys_pred_best


def chi2(y_true, y_pred, variances):
    assert len(y_true)==len(y_pred), 'y_true and y_pred must be same length!'
    assert len(y_true)==len(variances), 'y_true and variances must be same length!'
    chisq = np.sum((y_pred-y_true)**2/variances)
    return chisq


def reduced_chi2(y_true, y_pred, variances, deg_freedom):
    chisq = chi2(y_true, y_pred, variances)
    return chisq/deg_freedom


def get_deg_of_freedom(chaintag):
    chain_fn = f'../chains/param_files/chain_params_{chaintag}.h5'
    f = h5py.File(chain_fn, 'r')
    # The degree of freedom, nu = n-m,
    # equals the number of observations n minus the number of fitted parameters m.
    m_params = len(f.attrs['param_names_vary']) - len(f.attrs['fixed_param_names'])
    bin_arr = np.vstack(f['bins'])
    n_obs = len(bin_arr.flatten())
    return n_obs - m_params


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return 
