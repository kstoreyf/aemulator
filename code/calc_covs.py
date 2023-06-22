import numpy as np

import calc_aemulus_error
import calc_cov_emuperf
import calc_smoothgauss_cov
import combine_covariances

def main():
    stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = ['wp']
    #stat_strs = ['wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf']
    #stat_strs = ['wp_xi_xi2_mcf']
    #stat_strs = ['mcf', 'wp_mcf']
    # mock_tag = '_aemulus_Msatmocks_test'
    # cov_tag_extra = '_uchuuchi2nclosest2000'
    #train_tag_extra = f'_errstdev_Msatmocks{cov_tag_extra}'
    mock_tag = '_aemulus_fmaxmocks_test'
    cov_tag_extra = ''
    train_tag_extra = f'_errstdev_fmaxmocks{cov_tag_extra}'
    mode = None # for aemulus
    #mode = 'glam_for_uchuu'
    #mode = 'aemulus_for_uchuu'
    inflate_upf_err = True
    for stat_str in stat_strs:
        print(f'Calculating covariance matrices for {stat_str}')
        calc_aemulus_error.run(mock_tag, stat_str)
        calc_cov_emuperf.run(mock_tag, stat_str, train_tag_extra=train_tag_extra, cov_tag_extra=cov_tag_extra)
        calc_smoothgauss_cov.run(mock_tag, stat_str, cov_tag_extra=cov_tag_extra)
        if 'uchuu' in cov_tag_extra:
            # only do this for uchuu / other mock; smoothgauss is what we want for aemulus
            combine_covariances.run(mock_tag, stat_str, mode, 
                                    cov_tag_extra=cov_tag_extra, inflate_upf_err=inflate_upf_err)

if __name__=='__main__':
    main()
