import numpy as np
import os

import utils


def main():
    #single()
    #recovery_set()
    scale_analysis_set()

def single():
    #cosmo, hod = 3, 3
    cosmo, hod = 1, 12
    #statistics = ['wp', 'xi', 'xi2', 'upf', 'mcf']
    statistics = ['wp', 'xi']
    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    scales_to_include = ['large']*len(statistics)
    bins = [utils.get_bin_indices(scales) for scales in scales_to_include]
    n_threads = utils.get_nthreads(len(statistics))

    config_tag = '_largescales'
    #config_tag = '_smallscales'
    stat_str = '_'.join(statistics)
    contents = populate_config(config_tag, statistics, emu_names, scalings, n_threads, cosmo, hod, bins)
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}{config_tag}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)

def recovery_set():
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    stat_strs = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str) 
    stat_strs = np.concatenate((stat_strs, ['wp_xi_xi2_upf_mcf']))
    #config_tag = ''
    #config_tag = '_largescales'
    config_tag = '_smallscales'
    for stat_str in stat_strs:
        statistics = stat_str.split('_')
        emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
        scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
        scales_to_include = ['large']*len(statistics)
        bins = [utils.get_bin_indices(scales) for scales in scales_to_include]
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
    #id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    id_pairs = [(1,12)]
    #stat_strs = ['wp']
    stat_strs = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str) 
    stat_strs = np.concatenate((stat_strs, ['wp_xi_xi2_upf_mcf']))
    min_scales = np.arange(0, 9)
    for stat_str in stat_strs:
        statistics = stat_str.split('_')
        emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
        scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
        n_threads = utils.get_nthreads(len(statistics))
        for id_pair in id_pairs:
            cosmo, hod = id_pair
            for min_scale in min_scales:
                bins = [list(range(min_scale, 9))]*len(statistics) # for n_bins_tot = 9 
                config_tag = f'_minscale{min_scale}'
                contents = populate_config(config_tag, statistics, emu_names, scalings, n_threads, cosmo, hod, bins)
                config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}{config_tag}.cfg'
                if os.path.exists(config_fn):
                    print(f"Config {config_fn} already exists, skipping")
                    continue
                with open(config_fn, 'w') as f:
                    f.write(contents)
                print(f"Wrote config {config_fn}!")

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


if __name__=='__main__':
    main()
