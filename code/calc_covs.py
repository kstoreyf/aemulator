import numpy as np

import calc_aemulus_error
import calc_cov_emuperf
import calc_smoothgauss_cov
import combine_covariances

def main():
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = ['wp_xi', 'wp_xi_upf']
    stat_strs = ['mcf', 'wp_mcf']
    mock_tag = '_aemulus_Msatmocks_test'
    train_tag_extra = '_errstdev_Msatmocks'
    #mode = 'glam_for_uchuu'
    mode = 'aemulus_for_uchuu'
    for stat_str in stat_strs:
        print(f'Calculating covariance matrices for {stat_str}')
        calc_aemulus_error.run(mock_tag, stat_str)
        calc_cov_emuperf.run(mock_tag, stat_str, train_tag_extra)
        calc_smoothgauss_cov.run(mock_tag, stat_str)
        combine_covariances.run(mock_tag, stat_str, mode)

if __name__=='__main__':
    main()
