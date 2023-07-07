import argparse
import numpy as np

import initialize_chain
import run_chain


def main(cosmo, hod, mode, config_tag='', param_tag=''):
    
    print(f"Running chain set for cosmo {cosmo}, hod {hod}")
    print("config_tag:", config_tag)
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = np.loadtxt('../tables/statistic_sets_single.txt', dtype=str)
    #stat_strs = np.array(['xi', 'xi2', 'upf', 'mcf'])
    stat_strs = np.loadtxt('../tables/statistic_sets_addin.txt', dtype=str)
    #stat_strs = np.concatenate((stat_strs, ['wp_xi_xi2_upf_mcf', 'wp_xi_xi2_mcf']))
    #stat_strs = ['wp_xi_xi2_upf_mcf', 'wp_xi_xi2_mcf']
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf']
    #stat_strs = np.array(['wp'])
    data_name = 'aemulus_fmaxmocks_test'
    data_tag = '_'+data_name
    
    for stat_str in stat_strs:
        print(f'Running chain for stat_str={stat_str} (cosmo {cosmo}, hod {hod})')

        if mode=='single':
            config_tags = [config_tag]
        elif mode=='minscales':
            min_scales = np.arange(0,9)
            config_tags = [f'_minscale{min_scale}' for min_scale in min_scales]
        elif mode=='maxscales':
            max_scales = np.arange(0,9)
            config_tags = [f'_maxscale{max_scale}' for max_scale in max_scales]

        for config_tag in config_tags:
            config_fn = f'../chains/configs/chains_{stat_str}{data_tag}_c{cosmo}h{hod}{param_tag}{config_tag}.cfg'
            chain_params_fn = initialize_chain.main(config_fn, overwrite_param_file=False)
            if chain_params_fn==-1:
                continue # means exists already
            run_chain.run(chain_params_fn)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cosmo', type=int,
                        help='id of cosmology')
    parser.add_argument('hod', type=int,
                        help='id of HOD')
    parser.add_argument('mode', type=str,
                        help='single,minscales,maxscales')
    parser.add_argument('config_tag', type=str, nargs='?', default='_minscale0')
    parser.add_argument('param_tag', type=str, nargs='?', default='')
    args = parser.parse_args()
    main(args.cosmo, args.hod, args.mode, config_tag=args.config_tag,
         param_tag=args.param_tag)