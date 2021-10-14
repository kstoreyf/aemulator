import argparse
import numpy as np

import initialize_chain
import run_chain


def main(cosmo, hod):

    print(f"Running chain set for cosmo {cosmo}, hod {hod}")
    stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    for stat_str in stat_strs:
        print(f'Running chain for stat_str={stat_str} (cosmo {cosmo}, hod {hod})')

        config_fn = f'../chains/configs/chains_{stat_str}_c{cosmo}h{hod}.cfg'
        chain_params_fn = initialize_chain.main(config_fn)
        if chain_params_fn==-1:
            continue # means exists already
    
        run_chain.run(chain_params_fn)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cosmo', type=int,
                        help='id of cosmology')
    parser.add_argument('hod', type=int,
                        help='id of HOD')
    args = parser.parse_args()
    main(args.cosmo, args.hod)