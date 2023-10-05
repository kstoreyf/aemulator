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

    # AEUMULUS
    # stat_strs = ['wp_xi_xi2_mcf', 'wp_xi_xi2_upf_mcf']
    # mock_tag = '_aemulus_fmaxmocks_test'
    # cov_tag_extra = ''
    # mode = '' # for aemulus
    # inflate_upf_err = False
    # id_tag = ''

    # UCHUU
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf']
    #stat_strs = ['wp_xi_xi2_upf_mcf', 'wp_xi_xi2_mcf']
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_upf_mcf']    
    #stat_strs = ['wp_xi_xi2', 'wp_xi_xi2_mcf', 'wp_xi_xi2_upf', 'wp_xi_xi2_upf_mcf', 'upf']
    #stat_strs = ['wp_upf', 'wp_mcf']
    #stat_strs = ['upf_mcf', 'wp_upf_mcf']

    # UNIT
    stat_strs = ['wp', 'xi', 'xi2', 'upf', 'mcf', 'wp_xi_xi2', 'wp_upf_mcf', 'wp_xi_xi2_upf_mcf']
    mock_tag = '_aemulus_fmaxmocks_test'
    cov_tag_extra = ''
    mode = 'glam_for_unit'
    inflate_upf_err = False
    id_tag = ''
    train_tag_extra = f'_errstdev_fmaxmocks{cov_tag_extra}'

    # UCHUU
    # stat_strs = ['mcf']
    # mock_tag = '_aemulus_fmaxmocks_test'
    # #stat_strs = ['upf']
    # cov_tag_extra = '_uchuuchi2nclosest2000'
    # #cov_tag_extra = ''
    # mode = 'glam_for_uchuu'
    # inflate_upf_err = False
    # if 'uchuuchi2nclosest2000' in cov_tag_extra:
    #     id_tag = '_aemulus_fmaxmocks_uchuuchi2nclosest2000'
    # else:
    #     id_tag = ''
    # train_tag_extra = f'_errstdev_fmaxmocks{cov_tag_extra}'

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
