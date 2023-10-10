import numpy as np
import os
from collections import defaultdict

import utils


def main():
    run()
    #run_scale_dicts()
    #run_single()
    #run_scale_dicts_addin()

def run_single():
    results_dict_dir = '../data_products/results_dicts'
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)

    fn_results_dict_wp80 = f'{results_dict_dir}/results_dict_wp_wp80.npy'
    stat_strs = ['wp', 'wp80']
    build_dict(stat_strs, id_pairs, '_minscale0', fn_results_dict_wp80)


def run():

    # for all dicts
    results_dict_dir = '../data_products/results_dicts'
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)
    data_tag = '_aemulus_fmaxmocks_test'
    #config_tag = '_minscale0'

    stat_strs_single = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str)
    stat_strs_addin = np.loadtxt('../tables/statistic_sets_addin.txt', dtype=str)
    stat_strs_addin_full = np.concatenate((['wp'], stat_strs_addin))

    # single and addin
    # fn_results_dict_single = f'{results_dict_dir}/results_dict_single.npy'
    # build_dict(stat_strs_single, id_pairs, data_tag, config_tag, fn_results_dict_single)

    #fn_results_dict_addin_full = f'{results_dict_dir}/results_dict_addin_full.npy'
    #build_dict(stat_strs_addin_full, id_pairs, data_tag, config_tag, fn_results_dict_addin_full)

    fn_results_dict_single_wpximaxscale = f'{results_dict_dir}/results_dict_single_wpximaxscale6.npy'
    config_tag = '_minscale0'
    print(config_tag)
    build_dict(stat_strs_single, id_pairs, data_tag, config_tag, fn_results_dict_single_wpximaxscale)

    # fn_results_dict_wpximaxscale = f'{results_dict_dir}/results_dict_wpximaxscale6.npy'
    # config_tag = '_minscale0'
    # print(config_tag)
    # build_dict(stat_strs_addin_full, id_pairs, data_tag, config_tag, fn_results_dict_wpximaxscale)


def run_scale_dicts():

    data_tag = '_aemulus_fmaxmocks_test'

    results_dict_dir = '../data_products/results_dicts'
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)

    # scale dicts
   # stat_strs_single = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str)
    stat_strs_scale = np.loadtxt('../tables/statistic_sets_scale_analysis.txt', dtype=str)
    print(stat_strs_scale)
    #stat_strs_scale = np.delete(stat_strs_scale, np.where(stat_strs_scale=='upf'))
    #print(stat_strs_scale)
    #stat_strs_scale = np.concatenate((stat_strs_single, ['wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    #stat_strs_scale = np.concatenate((stat_strs_single, ['xi_xi2', 'wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    #stat_strs_scale = np.concatenate((stat_strs_single, ['xi_xi2', 'wp_xi_xi2']))
    #stat_strs_scale = np.concatenate((stat_strs_single, ['wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']))
    #extra_tag = '_upfmatch'
    extra_tag = ''
    #extra_tag = '_mini'
    # min_scales = np.arange(0,9)
    # fn_results_dict_minscales = f'{results_dict_dir}/results_dict_minscales{extra_tag}.npy'
    # build_dict_scales(stat_strs_scale, id_pairs, data_tag, '_minscale', fn_results_dict_minscales, min_scales)

    max_scales = np.arange(0,9)
    fn_results_dict_maxscales = f'{results_dict_dir}/results_dict_maxscales{extra_tag}.npy'
    build_dict_scales(stat_strs_scale, id_pairs, data_tag, '_maxscale', fn_results_dict_maxscales, max_scales)


def run_scale_dicts_addin():

    data_tag = '_aemulus_fmaxmocks_test'

    results_dict_dir = '../data_products/results_dicts'
    id_pairs = np.loadtxt('../tables/id_pairs_recovery_test_70.txt', delimiter=',', dtype=np.int)

    stat_strs_scale = ['wp', 'wp_xi', 'wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']
    extra_tag = '_upfmatch_addin'
    min_scales = np.arange(0,9)
    fn_results_dict_minscales = f'{results_dict_dir}/results_dict_minscales{extra_tag}.npy'
    build_dict_scales(stat_strs_scale, id_pairs, data_tag, '_minscale', fn_results_dict_minscales, min_scales)

    max_scales = np.arange(0,9)
    fn_results_dict_maxscales = f'{results_dict_dir}/results_dict_maxscales{extra_tag}.npy'
    build_dict_scales(stat_strs_scale, id_pairs, data_tag, '_maxscale', fn_results_dict_maxscales, max_scales)


def build_dict(stat_strs, id_pairs, data_tag, config_tag, fn_results_dict, 
               param_tag='', overwrite=False):
    if os.path.exists(fn_results_dict) and not overwrite:
        print(f"Results dict {fn_results_dict} already exists, not overwriting!")
        return
    print(f"Building results dict {fn_results_dict}...")
    results_dict = defaultdict(dict)
    print('config tag:', config_tag)
    for stat_str in stat_strs:
        print(stat_str)
        for id_pair in id_pairs:
            id_cosmo, id_hod = id_pair
            # print('config tag2:', config_tag)
            # print('wpximaxscale6' in config_tag)
            # print('wpximaxscale6' in config_tag)
            # print(stat_str=='wp')
            # print('xi' in stat_str.split('_'))
            if 'wpximaxscale6' in fn_results_dict and stat_str=='wp':
                chaintag = f'{stat_str}{data_tag}_c{id_cosmo}h{id_hod}{param_tag}{config_tag}_wpmaxscale6'
            elif 'wpximaxscale6' in fn_results_dict and stat_str=='xi':
                chaintag = f'{stat_str}{data_tag}_c{id_cosmo}h{id_hod}{param_tag}{config_tag}_ximaxscale6'
            elif 'wpximaxscale6' in fn_results_dict and 'xi' in stat_str.split('_'):
                chaintag = f'{stat_str}{data_tag}_c{id_cosmo}h{id_hod}{param_tag}{config_tag}_wpximaxscale6'
            else:
                chaintag = f'{stat_str}{data_tag}_c{id_cosmo}h{id_hod}{param_tag}{config_tag}'
            print(chaintag)
            results_dict[stat_str][tuple(id_pair)] = utils.construct_results_dict(chaintag)
    np.save(fn_results_dict, results_dict)
    print("Built and saved!")


def build_dict_scales(stat_strs, id_pairs, data_tag, config_subtag, fn_results_dict, 
                      scales, param_tag='', overwrite=False):
    if os.path.exists(fn_results_dict) and not overwrite:
        print(f"Results dict {fn_results_dict} already exists, not overwriting!")
        return
    print(f"Building results dict {fn_results_dict}...", flush=True)
    results_dict = defaultdict(dict)
    for scale in scales:
        print(f"{config_subtag} {scale}", flush=True)
        config_tag = f'{config_subtag}{scale}'
        results_dict_m = defaultdict(dict)
        for stat_str in stat_strs:
            print(stat_str, flush=True)
            for id_pair in id_pairs:
                id_cosmo, id_hod = id_pair
                chaintag = f'{stat_str}{data_tag}_c{id_cosmo}h{id_hod}{param_tag}{config_tag}'
                # if 'upf' in stat_str and len(stat_str.split('_'))>1:
                #     chaintag = f'{stat_str}_c{id_cosmo}h{id_hod}_all{config_tag}_upfmatch'
                # else:
                #     chaintag = f'{stat_str}_c{id_cosmo}h{id_hod}_all{config_tag}'
                # chain_results_dir = '/mount/sirocco1/ksf293/aemulator/chains/results'
                # chain_results_fn = f'{chain_results_dir}/results_{chaintag}.pkl'
                # if not os.path.exists(chain_results_fn):
                #     print("doesn't exist!", chaintag)
                #     continue

                results_dict_m[stat_str][tuple(id_pair)] = utils.construct_results_dict(chaintag)

        results_dict[scale] = results_dict_m

    np.save(fn_results_dict, results_dict)
    print("Built and saved!", flush=True)


if __name__=='__main__':
    main()
