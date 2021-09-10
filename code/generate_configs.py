import numpy as np



def main():
    single()

def single():
    #cosmo, hod = 3, 3
    cosmo, hod = 1, 12
    statistics = ['wp', 'xi', 'upf', 'mcf', 'xi2']
    emu_names = [utils.get_fiducial_emu_name[statistic] for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling[statistic] for statistic in statistics]
    n_threads = utils.get_nthreads(len(statistics))

    stat_str = '_'.join(statistics)
    contents = populate_config(statistics, emu_names, scalings, n_threads, cosmo, hod)
    config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}.cfg'
    print(config_fn)
    with open(config_fn, 'w') as f:
        f.write(contents)

def recovery_set():
    ntotal = 21
    ncosmos = 7
    statistics = ['wp']
    #statistics = ['wp', 'xi', 'upf', 'mcf', 'xi2']
    emu_names = [utils.get_fiducial_emu_name[statistic] for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling[statistic] for statistic in statistics]
    n_threads = utils.get_nthreads(len(statistics))
    stat_str = '_'.join(statistics)
    for n in range(ntotal):
        hoddigit = int(n/ncosmos)
        cosmo = n%ncosmos
        hod = cosmo*10 + hoddigit
        contents = populate_config(statistics, train_tags, cosmo, hod)
        config_fn = f'/home/users/ksf293/aemulator/chains/configs/chains_{stat_str}_c{cosmo}h{hod}.cfg'
        print(config_fn)
        with open(config_fn, 'w') as f:
            f.write(contents)

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
