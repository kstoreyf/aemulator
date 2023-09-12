import numpy as np
import os

import utils
from utils import rbins, rlin

def main():
    # example contours
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str)

    # covariance tests in appendix
    stat_strs = ['wp_xi_xi2_upf_mcf']
    generate_single_mock(stat_strs, (1,19), config_tag='_minscale0_covemuperf')
    generate_single_mock(stat_strs, (6,66), config_tag='_minscale0_covemuperf')
    #generate_recovery_set(stat_strs)
    #config_prior()

    #stat_strs = np.loadtxt('../tables/statistic_sets_scale_analysis.txt', dtype=str)
    #stat_strs = ['wp_xi_xi2_mcf']
    #generate_scale_analysis_set(stat_strs, mode='minscales')
    #generate_scale_analysis_set(stat_strs, mode='maxscales')

    # stat_strs = np.loadtxt('../tables/statistic_sets_addin.txt', dtype=str)
    # stat_strs = np.concatenate((np.array(['wp']), stat_strs))
    # generate_recovery_set(stat_strs, config_tag='_minscale0_wpximaxscale6')

    # stat_strs = np.array(['xi_xi2', 'wp_xi_xi2'])
    # generate_recovery_set(stat_strs, config_tag='_minscale0',
    #                       param_tag='_fixgammaf')
 
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_upf_mcf']
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf', 'upf']
    #stat_strs = ['wp_xi', 'wp_upf', 'wp_mcf']
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_upf', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']
    #stat_strs = ['wp_xi_xi2_upf_mcf', 'wp_xi_xi2_mcf']
    #stat_strs = ['wp_xi_xi2_upf_mcf']
    #stat_strs = ['wp_xi_xi2']

    # for stat_str in stat_strs:
    #   config_uchuu(stat_str)


def generate_single_mock(stat_strs, id_pair, config_tag='_minscale0',
                         param_tag=''):
    cosmo, hod = id_pair
    for stat_str in stat_strs:  
        config_aemulus(stat_str, cosmo, hod, config_tag=config_tag,
                       param_tag=param_tag)


def generate_recovery_set(stat_strs, config_tag='_minscale0',
                          param_tag=''):
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    #id_pairs = [(0,0)] # for testing
    for id_pair in id_pairs:
        cosmo, hod = id_pair
        for stat_str in stat_strs:  
            config_aemulus(stat_str, cosmo, hod, config_tag=config_tag,
                           param_tag=param_tag)


def generate_scale_analysis_set(stat_strs, mode='maxscales',
                                config_tag_extra=''):
    n_bins_tot = 9
    scales = np.arange(0, n_bins_tot)
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    #id_pairs = [(0,0)]
    for id_pair in id_pairs:
        cosmo, hod = id_pair
        for stat_str in stat_strs:  
            statistics = stat_str.split('_')
            cosmo, hod = id_pair

            for scale in scales:
                if mode=='minscales':
                    bins = [list(range(scale, n_bins_tot))]*len(statistics) # for n_bins_tot = 9 
                    config_tag = f'_minscale{scale}'
                    # if we need to line up upf with other bins, match those bins
                    # if just upf, regular binning and we will make sure to 
                    # plot it on proper scales
                    if 'upf' in statistics and len(statistics)>1:
                        idx_upf = statistics.index('upf')
                        bins[idx_upf] = match_bins(rbins, rlin, minscale_idx=scale, maxscale_idx=n_bins_tot-1) 
                elif mode=='maxscales':
                    bins = [list(range(0, scale+1))]*len(statistics) # for n_bins_tot = 9; +1 bc should be inclusive
                    config_tag = f'_maxscale{scale}'
                    if 'upf' in statistics and len(statistics)>1:
                        idx_upf = statistics.index('upf')
                        bins[idx_upf] = match_bins(rbins, rlin, minscale_idx=0, maxscale_idx=scale)  
                else:
                    raise ValueError("Mode note recognized!")
                
                config_tag += config_tag_extra
                config_aemulus(stat_str, cosmo, hod, config_tag=config_tag, bins=bins)


def config_uchuu(stat_str):

    statistics = stat_str.split('_')

    data_name = 'uchuu'
    data_tag = '_'+data_name

    mock_tag = '_aemulus_fmaxmocks'
    mock_name_train = 'aemulus_fmaxmocks_train'
    mock_name_test = 'aemulus_fmaxmocks_test'

    infl_tag = ''
    #infl_tag = '_inflateupferr2nox'
    #comb_tag = '_smooth'+infl_tag
    comb_tag = '_smoothemuboth'+infl_tag
    #comb_tag = '_smoothboth'+infl_tag
    #comb_tag = '_smooth'+infl_tag
    #comb_tag = '_smooth_covnegfix'+infl_tag
    cov_tag_extra = '_uchuuchi2nclosest2000'
    #cov_tag_extra = ''
    config_tag = f'{mock_tag}{cov_tag_extra}_smoothemuboth{infl_tag}_wpxiupfmaxscale6'
    #config_tag = f'{mock_tag}{cov_tag_extra}_smoothemu{infl_tag}_wpximaxscale6'
    #config_tag = f'{mock_tag}{cov_tag_extra}_smoothboth{infl_tag}_allmaxscale6'
    #config_tag = f'{mock_tag}{cov_tag_extra}_covnegfix{infl_tag}_wpximaxscale6'
    #config_tag = f'_Msatmocks_upfmaxscale6_covglamsmooth_boundsingle{cov_tag_extra}{infl_tag}'
    #config_tag = '_Msatmocks_wpmaxscale6'

    #param_tag = ''
    param_tag = '_fixgammaf'
    #param_tag = '_all'
    #param_tag = '_hodparams'
    save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}{data_tag}{param_tag}{config_tag}.h5'

    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]

    train_tags_extra = [f'_errstdev_fmaxmocks{cov_tag_extra}']*len(statistics)

    chain_results_fn = f'/mount/sirocco1/ksf293/aemulator/chains/results/results_{stat_str}{data_tag}{param_tag}{config_tag}.pkl'
    n_threads = utils.get_nthreads(len(statistics))
    dlogz_str = '1e-2'
    # use aemulus covariance for uchuu
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_smoothgauss_emuperf_{mock_name_test}_{stat_str}_hod3_test0.dat'
    # try w combined glam4 cov
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_combined_{mock_name_test}_uchuuglam4_{stat_str}.dat'
    # combined glam cov
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_combined_{mock_name_test}_uchuuglam_{stat_str}.dat'
    cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_combined_{mock_name_test}{cov_tag_extra}_uchuuglam{comb_tag}_{stat_str}.dat'
    # aemulus for uchuu cov
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_combined_{mock_name_test}_uchuu_{stat_str}.dat'
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_combined_{mock_name_test}_uchuu_smooth_{stat_str}.dat'

    param_names_vary = get_param_names(param_tag, mock_name_train)
    seed = np.random.randint(1000)

    if 'wpmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if 'wp' in statistics[i]: #writing this way to include "wp80"
                bins.append(list(range(0, 7)))
            else:
                bins.append(list(range(0, 9)))
    elif 'wpximaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            #writing this way for wp to include "wp80", but not for xi bc of xi2
            if 'wp' in statistics[i] or statistics[i]=='xi': 
                bins.append(list(range(0, 7)))
            else:
                bins.append(list(range(0, 9)))
    elif 'wpxiupfmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            #writing this way for wp to include "wp80", but not for xi bc of xi2
            if 'wp' in statistics[i] or statistics[i]=='xi' or statistics[i]=='upf': 
                bins.append(list(range(0, 7)))
            else:
                bins.append(list(range(0, 9)))
    elif 'upfmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if statistics[i]=='upf':
                bins.append(list(range(0, 7)))
            else:
                bins.append(list(range(0, 9)))
    elif 'upfmaxscale0' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if statistics[i]=='upf':
                bins.append([0])
            else:
                bins.append(list(range(0, 9)))
    elif 'allmaxscale6' in config_tag:
        bins = [list(range(0, 7))]*len(statistics)
    else:
        bins = [list(range(0, 9))]*len(statistics)
    
    contents = populate_config_blank(save_fn, statistics, emu_names, scalings, train_tags_extra,
                        mock_name_train, mock_name_test,
                        chain_results_fn, n_threads, dlogz_str, 
                        cov_fn, param_names_vary, seed,
                        data_name, bins)

    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{data_tag}{param_tag}{config_tag}.cfg'
    
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)


def config_aemulus(stat_str, cosmo, hod, config_tag='', param_tag='', bins=None):
    statistics = stat_str.split('_')

    data_name = 'aemulus_fmaxmocks_test'
    data_tag = '_'+data_name

    # mock names used for building emus
    mock_name_train = 'aemulus_fmaxmocks_train'
    mock_name_test = 'aemulus_fmaxmocks_test'

    #param_tag = '_omegam_sigma8'
    save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.h5'

    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    train_tags_extra = ['_errstdev_fmaxmocks']*len(statistics)

    #chain_results_fn = f'/home/users/ksf293/aemulator/chains/results/results_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.pkl'
    chain_results_fn = f'/mount/sirocco1/ksf293/aemulator/chains/results/results_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.pkl'
    n_threads = utils.get_nthreads(len(statistics))
    dlogz_str = '1e-2'
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_smoothgauss_emuperf_{mock_name_test}_{stat_str}_hod3_test0.dat'
    cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_emuperf_{mock_name_test}_{stat_str}_hod3_test0.dat'
    print("COV EMUPERF, NOT SMOOTH")

    # param names, fmaxmocks: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env', 'f_max']
    param_names_vary = get_param_names(param_tag, mock_name_train)
    #print('param names vary:', param_names_vary)

    seed = np.random.randint(1000)

    # if it isn't None, we've passed it in and take that!
    if bins is None:
        if 'wpmaxscale6' in config_tag:
            bins = []
            for i in range(len(statistics)):
                if 'wp' in statistics[i]:
                    bins.append(list(range(0, 7)))
                else:
                    bins.append(list(range(0, 9)))
        elif 'wpximaxscale6' in config_tag:
            bins = []
            for i in range(len(statistics)):
                if 'wp' in statistics[i] or statistics[i]=='xi':
                    bins.append(list(range(0, 7)))
                else:
                    bins.append(list(range(0, 9)))
        elif 'allmaxscale6' in config_tag:
            bins = [list(range(0, 7))]*len(statistics)
        else:
            bins = [list(range(0, 9))]*len(statistics)
    
    contents = populate_config_blank(save_fn, statistics, emu_names, scalings, train_tags_extra,
                        mock_name_train, mock_name_test,
                        chain_results_fn, n_threads, dlogz_str, 
                        cov_fn, param_names_vary, seed,
                        data_name, bins, cosmo=cosmo, hod=hod)
    
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)


def config_prior(config_tag=''):

    statistics = []

    # don't actually need for prior but need to run_chain smoothly
    data_name = 'prior'
    data_tag = '_'+data_name

    # mock names used for building emus
    mock_name_train = 'aemulus_fmaxmocks_train'
    mock_name_test = 'aemulus_fmaxmocks_test'

    param_tag = '' # means all
    #save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.h5'
    save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params{data_tag}{param_tag}{config_tag}.h5'

    emu_names = []
    scalings = []
    train_tags_extra = []

    #chain_results_fn = f'/home/users/ksf293/aemulator/chains/results/results_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.pkl'
    chain_results_fn = f'/mount/sirocco1/ksf293/aemulator/chains/results/results{data_tag}{param_tag}{config_tag}.pkl'
    n_threads = 24
    dlogz_str = '1e-2'
    # won't be used but need a dummy covmat to make run_chain smoothly
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_smoothgauss_emuperf_{mock_name_test}_wp_hod3_test0.dat'
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_empty.dat'
    cov_fn = None
    #if not os.path.exists(cov_fn):
    # empty = np.empty((0,0))
    # print(empty.shape)
    # np.savetxt(cov_fn, empty)

    param_names_vary = get_param_names(param_tag, mock_name_train)

    seed = np.random.randint(1000)
    bins = []
    
    # won't be used but need a dummy to make run_chain smoothly
    cosmo, hod = 0, 0

    contents = populate_config_blank(save_fn, statistics, emu_names, scalings, train_tags_extra,
                        mock_name_train, mock_name_test,
                        chain_results_fn, n_threads, dlogz_str, 
                        cov_fn, param_names_vary, seed,
                        data_name, bins, cosmo=cosmo, hod=hod)
    
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains{data_tag}{param_tag}{config_tag}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)


def get_param_names(param_tag, mock_name_train):
    if param_tag=='':
        cosmo_param_names, _ = utils.load_cosmo_params(mock_name_train)
        hod_param_names, _ = utils.load_hod_params(mock_name_train)
        param_names_vary = cosmo_param_names + hod_param_names
    elif param_tag=='_omegam':
        # usually for testing purposes
        param_names_vary = ['Omega_m']
    elif param_tag=='_omegam_sigma8':
        # usually for testing purposes
        param_names_vary = ['Omega_m', 'sigma_8']
    elif param_tag=='_fixgammaf':
        cosmo_param_names, _ = utils.load_cosmo_params(mock_name_train)
        hod_param_names, _ = utils.load_hod_params(mock_name_train)
        param_names_vary = cosmo_param_names + hod_param_names
        param_names_vary.remove('f')
    else:
        raise ValueError("What to use for param_names_vary??")
    return param_names_vary


def single():
    #cosmo, hod = 3, 3
    cosmo, hod = 1, 12
    #statistics = ['wp', 'xi', 'xi2', 'upf', 'mcf']
    statistics = ['wp']
    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    scales_to_include = ['all']*len(statistics)
    bins = [utils.get_bin_indices(scales) for scales in scales_to_include]
    n_threads = utils.get_nthreads(len(statistics))

    config_tag = '_test'
    #config_tag = '_smallscales'
    stat_str = '_'.join(statistics)
    contents = populate_config(config_tag, statistics, emu_names, scalings, n_threads, cosmo, hod, bins)
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}{config_tag}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)


def recovery_set():
    #id_pairs = np.loadtxt('../tables/id_pairs_test.txt', delimiter=',', dtype=np.int)
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str) 
    #stat_strs = np.loadtxt('../tables/statistic_sets_addin.txt', dtype=str)
    #stat_strs = np.concatenate((stat_strs, ['wp_xi_xi2_mcf']))
    #stat_strs = np.concatenate((stat_strs, ['wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    stat_strs = np.array(['wp80'])
    config_tag = '_minscale0'
    #config_tag = '_wpmaxscale6'
    #config_tag = '_largescales'
    #config_tag = '_smallscales'
    for stat_str in stat_strs:
        statistics = stat_str.split('_')
        emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
        scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
        #scales_to_include = ['large']*len(statistics)
        bins = [list(range(0, 9))]*len(statistics) # for n_bins_tot = 9
        #bins = [utils.get_bin_indices(scales) for scales in scales_to_include]
        #bins = []
        # exclude 2 largest-scale wp bins (causing bias)
        #for stat in statistics:
            #if stat=='wp':
            #    bins.append(list(range(0, 7)))
            #else:
            #    bins.append(list(range(0, 9)))
        n_threads = utils.get_nthreads(len(statistics))
        for id_pair in id_pairs:
            cosmo, hod = id_pair
            contents = populate_config(config_tag, statistics, emu_names, scalings, n_threads, cosmo, hod, bins)
            config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}{config_tag}.cfg'
            if os.path.exists(config_fn):
                print(f"Config {config_fn} already exists, skipping")
                continue
            with open(config_fn, 'w') as f:
                f.write(contents)
            print(f"Wrote config {config_fn}!")


def scale_analysis_set():
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    overwrite = False
    #id_pairs = [(1,12)]
    #stat_strs = ['wp']
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str) 
    #stat_strs = np.concatenate((stat_strs, ['wp_xi_xi2_upf_mcf', 'wp_xi_xi2_mcf']))
    stat_strs = np.array(['wp_xi_xi2_upf_mcf'])
    #stat_strs = ['wp_xi_xi2_mcf']
    #min_scales = np.array([0])
    #min_scales = np.arange(0, 9)
    max_scales = np.arange(0,9)
    for stat_str in stat_strs:
        statistics = stat_str.split('_')
        emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
        scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
        n_threads = utils.get_nthreads(len(statistics))
        for id_pair in id_pairs:
            cosmo, hod = id_pair
            for max_scale in max_scales:
            #for min_scale in min_scales:
                # bins = [list(range(min_scale, 9))]*len(statistics) # for n_bins_tot = 9 
                # config_tag = f'_minscale{min_scale}'
                bins = [list(range(0, max_scale+1))]*len(statistics) # for n_bins_tot = 9; +1 bc should be inclusive
                config_tag = f'_maxscale{max_scale}'

                # if UPF in list, get it's special bins to align it with the others
                print(rbins)
                print(rlin)
                if 'upf' in statistics:
                    idx_upf = statistics.index('upf')
                    bins[idx_upf] = match_bins(rbins, rlin, minscale_idx=0, maxscale_idx=max_scale)  
                    #bins[idx_upf] = match_bins(rbins, rlin, minscale_idx=min_scale, maxscale_idx=8) 
                    
                    print('max scale idx:', max_scale)
                    print('max scale:', rbins[max_scale], rbins[max_scale+1])
                    print('upf bins:', rlin[bins[idx_upf]])
                    #print(bins)     
                    config_tag += '_upfmatch'    
                
                continue 

                contents = populate_config(config_tag, statistics, emu_names, scalings, n_threads, cosmo, hod, bins)
                config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}{config_tag}.cfg'
                if os.path.exists(config_fn) and not overwrite:
                    print(f"Config {config_fn} already exists, skipping")
                    continue
                with open(config_fn, 'w') as f:
                    f.write(contents)
                print(f"Wrote config {config_fn}!")


# minscaleN refers to min N bins from bins1, which should be 
# the log-scaled bins for our purposes. 
# bins are bin averages, or radii for UPF
# e.g. minscale0  
def match_bins(bins1, bins2, minscale_idx, maxscale_idx):
    minscale_val = bins1[minscale_idx]
    maxscale_val = bins1[maxscale_idx+1] # because want top bin edge
    bins2_idxs = np.where((minscale_val < bins2) & (bins2 < maxscale_val))[0]
    return list(bins2_idxs)


def populate_config(config_tag, statistics, emu_names, scalings, n_threads, cosmo, hod, bins):
    stat_str = '_'.join(statistics)
    n_stats = len(statistics)
    contents = \
f"""---
save_fn: '/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}_c{cosmo}h{hod}_all{config_tag}.h5'

emu:
    statistics: {statistics}
    emu_names: {emu_names}
    scalings: {scalings}
    
chain:
    chain_results_fn: '/home/users/ksf293/aemulator/chains/results/results_{stat_str}_c{cosmo}h{hod}_all{config_tag}.pkl'
    n_threads: {n_threads}
    dlogz: 1e-2
    cov_fn: '/home/users/ksf293/aemulator/covariances/cov_smoothgauss_emuperf_{stat_str}_hod3_test0.dat'
    param_names_vary: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env'] #all
    #seed: 12 # If not defined, code chooses random int in [0, 1000)

data:
    cosmo: {cosmo}
    hod: {hod}
    bins: {bins}
"""
    return contents


def populate_config_blank(save_fn, statistics, emu_names, scalings, train_tags_extra,
                          mock_name_train, mock_name_test,
                          chain_results_fn, n_threads, dlogz_str, 
                          cov_fn, param_names_vary, seed,
                          data_name, bins, cosmo=None, hod=None):
    contents = \
f"""---
save_fn: '{save_fn}'

emu:
    statistics: {statistics}
    emu_names: {emu_names}
    scalings: {scalings}
    train_tags_extra: {train_tags_extra}
    mock_name_train: '{mock_name_train}'
    mock_name_test: '{mock_name_test}'
    
chain:
    chain_results_fn: '{chain_results_fn}'
    n_threads: {n_threads}
    dlogz: {dlogz_str}
    cov_fn: '{cov_fn}'
    param_names_vary: {param_names_vary}
    seed: {seed}

data:
    data_name: {data_name}
    cosmo: {cosmo}
    hod: {hod}
    bins: {bins}
"""
    return contents


if __name__=='__main__':
    main()
