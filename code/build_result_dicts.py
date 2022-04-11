import numpy as np
import os
from collections import defaultdict

import utils


def main():
    run_scale_dicts()

def run():

    # for all dicts
    results_dict_dir = '../data_products/results_dicts'
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)

    # single and addin
    stat_strs_single = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str)
    fn_results_dict_single = f'{results_dict_dir}/results_dict_single.npy'
    build_dict(stat_strs_single, id_pairs, '_minscale0', fn_results_dict_single)

    stat_strs_addin = np.loadtxt('../tables/statistic_sets_addin.txt', dtype=str)
    stat_strs_addin_full = np.concatenate((['wp'], stat_strs_addin))
    fn_results_dict_addin_full = f'{results_dict_dir}/results_dict_addin_full.npy'
    build_dict(stat_strs_addin_full, id_pairs, '_minscale0', fn_results_dict_addin_full)

    fn_results_dict_wpmaxscale = f'{results_dict_dir}/results_dict_wpmaxscale6.npy'
    build_dict(stat_strs_addin_full, id_pairs, '_wpmaxscale6', fn_results_dict_wpmaxscale)


def run_scale_dicts():

    results_dict_dir = '../data_products/results_dicts'
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)

    # scale dicts
    stat_strs_single = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str)
    #stat_strs_scale = np.concatenate((stat_strs_single, ['wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    stat_strs_scale = np.concatenate((stat_strs_single, ['xi_xi2', 'wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    #stat_strs_scale = np.concatenate((stat_strs_single, ['wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    extra_tag = '_upfmatch'
    #extra_tag = ''
    min_scales = np.arange(0,9)
    fn_results_dict_minscales = f'{results_dict_dir}/results_dict_minscales{extra_tag}.npy'
    build_dict_scales(stat_strs_scale, id_pairs, '_minscale', fn_results_dict_minscales, min_scales)

    max_scales = np.arange(0,9)
    fn_results_dict_maxscales = f'{results_dict_dir}/results_dict_maxscales{extra_tag}.npy'
    build_dict_scales(stat_strs_scale, id_pairs, '_maxscale', fn_results_dict_maxscales, max_scales)


def build_dict(stat_strs, id_pairs, config_tag, fn_results_dict, overwrite=False):
    if os.path.exists(fn_results_dict) and not overwrite:
        print(f"Results dict {fn_results_dict} already exists, not overwriting!")
        return
    print(f"Building results dict {fn_results_dict}...")
    results_dict = defaultdict(dict)
    for stat_str in stat_strs:
        print(stat_str)
        for id_pair in id_pairs:
            id_cosmo, id_hod = id_pair
            chaintag = f'{stat_str}_c{id_cosmo}h{id_hod}_all{config_tag}'
            results_dict[stat_str][tuple(id_pair)] = utils.construct_results_dict(chaintag)
    np.save(fn_results_dict, results_dict)
    print("Built and saved!")


def build_dict_scales(stat_strs, id_pairs, config_subtag, fn_results_dict, scales, overwrite=False):
    if os.path.exists(fn_results_dict) and not overwrite:
        print(f"Results dict {fn_results_dict} already exists, not overwriting!")
        return
    print(f"Building results dict {fn_results_dict}...")
    results_dict = defaultdict(dict)
    for scale in scales:
        print(f"{config_subtag} {scale}")
        config_tag = f'{config_subtag}{scale}'
        results_dict_m = defaultdict(dict)
        for stat_str in stat_strs:
            print(stat_str)
            for id_pair in id_pairs:
                id_cosmo, id_hod = id_pair
                if 'upf' in stat_str and len(stat_str.split('_'))>1:
                    chaintag = f'{stat_str}_c{id_cosmo}h{id_hod}_all{config_tag}_upfmatch'
                else:
                    chaintag = f'{stat_str}_c{id_cosmo}h{id_hod}_all{config_tag}'
                results_dict_m[stat_str][tuple(id_pair)] = utils.construct_results_dict(chaintag)
        results_dict[scale] = results_dict_m

    np.save(fn_results_dict, results_dict)
    print("Built and saved!")


if __name__=='__main__':
    main()
