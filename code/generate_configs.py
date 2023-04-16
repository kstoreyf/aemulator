import numpy as np
import os

import utils
from utils import rbins, rlin

def main():
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    stat_strs = ['wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf']
    #stat_strs = [ 'wp_xi_xi2_mcf']
    #stat_strs = ['wp_xi', 'wp_xi_upf', 'wp_xi_mcf']
    #stat_strs = ['wp', 'wp_xi_upf', 'wp_xi_mcf']
    #stat_strs = ['wp_xi_upf', 'wp_xi_mcf']
    #stat_strs = ['wp', 'wp_xi', 'wp_xi_upf_mcf']
    #single()
    #recovery_set()
    #scale_analysis_set()
    for stat_str in stat_strs:
      uchuu(stat_str)

    # id_pairs = [(3,3)]
    # for stat_str in stat_strs:  
    #     for id_pair in id_pairs:    
    #         cosmo, hod = id_pair
    #         aemulus(stat_str, cosmo, hod)


def uchuu(stat_str):

    statistics = stat_str.split('_')

    data_name = 'uchuu'
    data_tag = '_'+data_name

    mock_name_train = 'aemulus_Msatmocks_train'
    mock_name_test = 'aemulus_Msatmocks_test'
    #config_tag = '_Msatmocks_covglam4'
    #config_tag = '_Msatmocks_covglam4_allmaxscale6'
    #config_tag = '_Msatmocks_covaemsmooth'
    #config_tag = '_Msatmocks_covglamsmooth_ellcosmoprior'
    #infl_tag = '_inflateupferr3nox'
    infl_tag = ''
    comb_tag = '_smooth'+infl_tag
    cov_tag_extra = '_uchuuchi2nclosest2000'
    #cov_tag_extra = ''
    config_tag = f'_Msatmocks_upfmaxscale6_covglamsmooth_boundsingle{cov_tag_extra}{infl_tag}'
    #config_tag = '_Msatmocks_wpmaxscale6'

    param_tag = '_all'
    #param_tag = '_hodparams'
    save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}{data_tag}{param_tag}{config_tag}.h5'

    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]

    train_tags_extra = [f'_errstdev_Msatmocks{cov_tag_extra}']*len(statistics)

    chain_results_fn = f'/home/users/ksf293/aemulator/chains/results/results_{stat_str}{data_tag}{param_tag}{config_tag}.pkl'
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

    param_names_vary = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    #param_names_vary = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    seed = np.random.randint(1000)

    if 'wpmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if 'wp' in statistics[i]: #writing this way to include "wp80"
                bins.append(list(range(0, 7)))
            else:
                bins.append(list(range(0, 9)))
    if 'upfmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if statistics[i]=='upf':
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
                        data_name, bins)
    if param_tag=='_all':
        config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{data_tag}{config_tag}.cfg'
    else:
        config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{data_tag}{param_tag}{config_tag}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)


def aemulus(stat_str, cosmo, hod):

    statistics = stat_str.split('_')

    data_name = 'aemulus_Msatmocks_test'
    data_tag = '_'+data_name

    # mock names used for building emus
    mock_name_train = 'aemulus_Msatmocks_train'
    mock_name_test = 'aemulus_Msatmocks_test'
    config_tag = ''

    param_tag = '_all'
    save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.h5'

    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    train_tags_extra = ['_errstdev_Msatmocks']*len(statistics)

    chain_results_fn = f'/home/users/ksf293/aemulator/chains/results/results_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.pkl'
    n_threads = utils.get_nthreads(len(statistics))
    dlogz_str = '1e-2'
    cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_smoothgauss_emuperf_{mock_name_test}_{stat_str}_hod3_test0.dat'

    param_names_vary = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    #param_names_vary = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    seed = np.random.randint(1000)

    if 'wpmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if 'wp' in statistics[i]:
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
    if param_tag=='_all':
        config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{data_tag}_c{cosmo}h{hod}{config_tag}.cfg'
    else:
        config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)


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
