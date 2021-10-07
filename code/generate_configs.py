import numpy as np
import os

import utils


def main():
    #single()
    recovery_set()

def single():
    cosmo, hod = 3, 3
    #cosmo, hod = 1, 12
    statistics = ['wp', 'xi', 'upf', 'mcf', 'xi2']
    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    n_threads = utils.get_nthreads(len(statistics))

    stat_str = '_'.join(statistics)
    contents = populate_config(statistics, emu_names, scalings, n_threads, cosmo, hod)
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)

def recovery_set():
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    #statistics = ['wp', 'xi', 'upf', 'mcf', 'xi2']
    statistics = ['wp', 'mcf']
    stat_str = '_'.join(statistics)
    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    n_threads = utils.get_nthreads(len(statistics))
    for id_pair in id_pairs:
        cosmo, hod = id_pair
        contents = populate_config(statistics, emu_names, scalings, n_threads, cosmo, hod)
        config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}.cfg'
        if os.path.exists(config_fn):
            print(f"Config {config_fn} already exists, skipping")
            continue
        with open(config_fn, 'w') as f:
            f.write(contents)
        print(f"Wrote config {config_fn}!")

def populate_config(statistics, emu_names, scalings, n_threads, cosmo, hod):
    stat_str = '_'.join(statistics)
    n_stats = len(statistics)
    contents = \
f"""---
save_fn: '/home/users/ksf293/aemulator/chains/param_files/chain_params_{stat_str}_c{cosmo}h{hod}_all.h5'

emu:
    statistics: {statistics}
    emu_names: {emu_names}
    scalings: {scalings}
    
chain:
    chain_results_fn: '/home/users/ksf293/aemulator/chains/results/results_{stat_str}_c{cosmo}h{hod}_all.pkl'
    n_threads: {n_threads}
    dlogz: 1e-2
    cov_fn: '/home/users/ksf293/clust/covariances/cov_smoothgauss1_emuperf_{stat_str}_nonolap_hod3_test0_mean_test0.dat'
    param_names_vary: ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w', 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env'] #all
    #seed: 12 # If not defined, code chooses random int in [0, 1000)

data:
    cosmo: {cosmo}
    hod: {hod}
"""
    return contents


if __name__=='__main__':
    main()
