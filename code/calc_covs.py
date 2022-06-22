import numpy as np

import calc_aemulus_error
import calc_cov_emuperf
import calc_smoothgauss_cov

def main():
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    stat_strs = ['wp80_xi_xi2_upf_mcf']
    for stat_str in stat_strs:
        print(f'Calculating covariance matrices for {stat_str}')
        calc_aemulus_error.run(stat_str)
        calc_cov_emuperf.run(stat_str)
        calc_smoothgauss_cov.run(stat_str)

if __name__=='__main__':
    main()
