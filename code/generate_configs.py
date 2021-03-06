import numpy as np
import os

import utils
from utils import rbins, rlin

def main():
    #single()
    #recovery_set()
    #scale_analysis_set()
    uchuu()


def uchuu():

    statistics = ['wp', 'upf', 'mcf']
    #statistics = ['wp', 'xi', 'xi2', 'upf']
    #statistics = ['wp', 'xi', 'xi2']
    #statistics = ['wp', 'xi', 'xi2']
    #statistics = ['wp']
    #statistics = ['wp80']
    #statistics = ['mcf']
    #statistics = ['wp80', 'xi', 'xi2', 'upf', 'mcf']
    stat_str = '_'.join(statistics)

    mock_name = 'uchuu'
    mock_tag = '_'+mock_name
    config_tag = '_covglam4'
    #config_tag = '_covglam4_wpmaxscale6'
    #config_tag = '_wpmaxscale6'

    param_tag = '_all'
    save_fn = f'/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}{mock_tag}{param_tag}{config_tag}.h5'

    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]

    chain_results_fn = f'/home/users/ksf293/aemulator/chains/results/results_{stat_str}{mock_tag}{param_tag}{config_tag}.pkl'
    n_threads = utils.get_nthreads(len(statistics))
    dlogz_str = '1e-2'
    # use aemulus covariance for uchuu
    #cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_smoothgauss_emuperf_{stat_str}_hod3_test0.dat'
    # try w combined glam cov
    cov_fn = f'/home/users/ksf293/aemulator/covariances/cov_combined_uchuuglam4_{stat_str}.dat'
    param_names_vary = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']
    seed = np.random.randint(1000)

    if 'wpmaxscale6' in config_tag:
        bins = []
        for i in range(len(statistics)):
            if 'wp' in statistics[i]:
                bins.append(list(range(0, 7)))
            else:
                bins.append(list(range(0, 9)))
    else:
        bins = [list(range(0, 9))]*len(statistics)
    
    contents = populate_config_blank(save_fn, statistics, emu_names, scalings, 
                        chain_results_fn, n_threads, dlogz_str, 
                        cov_fn, param_names_vary, seed,
                        mock_name, bins)
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}{mock_tag}{config_tag}.cfg'
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


def populate_config_blank(save_fn, statistics, emu_names, scalings, 
                          chain_results_fn, n_threads, dlogz_str, 
                          cov_fn, param_names_vary, seed,
                          mock_name, bins):
    contents = \
f"""---
save_fn: '{save_fn}'

emu:
    statistics: {statistics}
    emu_names: {emu_names}
    scalings: {scalings}
    
chain:
    chain_results_fn: '{chain_results_fn}'
    n_threads: {n_threads}
    dlogz: {dlogz_str}
    cov_fn: '{cov_fn}'
    param_names_vary: {param_names_vary}
    seed: {seed}

data:
    data_name: {mock_name}
    bins: {bins}
"""
    return contents


if __name__=='__main__':
    main()
