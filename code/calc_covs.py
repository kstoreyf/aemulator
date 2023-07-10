import numpy as np

import calc_aemulus_error
import calc_cov_emuperf
import calc_smoothgauss_cov
import combine_covariances

def main():
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    #stat_strs = np.concatenate((stat_strs, ['xi_xi2']))
    #stat_strs = ['xi_xi2']
    #stat_strs = ['wp']
    #stat_strs = ['wp_xi_xi2_mcf']
    #stat_strs = ['mcf', 'wp_mcf']
    # mock_tag = '_aemulus_Msatmocks_test'
    #train_tag_extra = f'_errstdev_Msatmocks{cov_tag_extra}'
    mock_tag = '_aemulus_fmaxmocks_test'
    #cov_tag_extra = ''
    #cov_tag_extra = '_uchuu'
    #mode = None # for aemulus
    #inflate_upf_err = False

    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf']
    stat_strs = ['wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']
    #stat_strs = ['mcf']
    cov_tag_extra = '_uchuuchi2nclosest2000'
    mode = 'glam_for_uchuu'
    #mode = 'aemulus_for_uchuu'
    inflate_upf_err = False
    id_tag = '_aemulus_fmaxmocks_uchuuchi2nclosest2000'

    train_tag_extra = f'_errstdev_fmaxmocks{cov_tag_extra}'

    for stat_str in stat_strs:
        print(f'Calculating covariance matrices for {stat_str}')
        # for now not changing this for nclosest bc devs from mean
        print("Aemulus error")
        calc_aemulus_error.run(mock_tag, stat_str)
        print("Emuperf error")
        calc_cov_emuperf.run(mock_tag, stat_str, train_tag_extra=train_tag_extra, cov_tag_extra=cov_tag_extra, id_tag=id_tag)
        print("Smoothing")
        calc_smoothgauss_cov.run(mock_tag, stat_str, cov_tag_extra=cov_tag_extra)
        if 'uchuu' in mode:
            # only do this for uchuu / other mock; smoothgauss is what we want for aemulus
            print("Combining covariances")
            combine_covariances.run(mock_tag, stat_str, mode, 
                                    cov_tag_extra=cov_tag_extra, inflate_upf_err=inflate_upf_err)

if __name__=='__main__':
    main()
