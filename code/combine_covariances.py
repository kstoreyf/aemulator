import numpy as np
import os

import calc_smoothgauss_cov as csc


def main():
    # The tag that we made the covariance matrices with! 
    mock_tag_cov = '_aemulus_Msatmocks_test'
    statistics = ['wp', 'xi', 'upf']
    #statistics = ['mcf']
    stat_str = '_'.join(statistics)
    mode = 'glam_for_uchuu'
    #mode = 'glam_for_aemulus'
    #mode = 'aemulus_for_uchuu'
    run(mock_tag_cov, stat_str, mode)


def run(mock_tag_cov, stat_str, mode, cov_tag_extra=''):

    mock_name_glam = 'glam'
    cov_dir = '../covariances'
    if mode=='glam_for_uchuu':
        cov_combined_fn = f"{cov_dir}/cov_combined{mock_tag_cov}{cov_tag_extra}_uchuu{mock_name_glam}_smooth_{stat_str}.dat"
    elif mode=='glam_for_aemulus':
        cov_glam_fn = f'{cov_dir}/cov_{mock_name_glam}_{stat_str}.dat'
        cov_combined_fn = f"{cov_dir}/cov_combined{mock_tag_cov}{cov_tag_extra}_{mock_name_glam}_{stat_str}.dat"
    elif mode=='aemulus_for_uchuu':
        cov_combined_fn = f"{cov_dir}/cov_combined{mock_tag_cov}{cov_tag_extra}_uchuu_smooth_{stat_str}.dat"

    if 'glam' in mode:
        cov_glam_fn = f'{cov_dir}/cov_{mock_name_glam}_{stat_str}.dat'
        cov_glam = np.loadtxt(cov_glam_fn)

    cov_aemulus_fn = f'{cov_dir}/cov{mock_tag_cov}_{stat_str}_hod3_test0.dat'
    #cov_emuperf_fn = f'{cov_dir}/cov_emuperf_{stat_str}_nonolap_hod3_test0_mean_test0.dat'
    cov_emuperf_fn = f'{cov_dir}/cov_emuperf{mock_tag_cov}{cov_tag_extra}_{stat_str}_hod3_test0.dat'

    # TODO: check if i want to be using smoothgauss here! wasnt before, bc only smoothed after glam failed
    cov_smooth_emuperf_fn = f'{cov_dir}/cov_smoothgauss_emuperf{mock_tag_cov}{cov_tag_extra}_{stat_str}_hod3_test0.dat'
    
    # load covs
    cov_aemulus = np.loadtxt(cov_aemulus_fn)
    cov_emuperf = np.loadtxt(cov_emuperf_fn)
    cov_smooth_emuperf = np.loadtxt(cov_smooth_emuperf_fn)
    
    # rescale aemulus cov bc tested emu on mean of 5 boxes
    cov_aemulus_5box = cov_aemulus*(1/5)
    
    # rescale covariances 
    L_uchuu = 2000.
    L_glam = 1000.
    L_aemulus = 1050.
    if mode=='glam_for_uchuu':
        # scale glam to uchuu, for recovery on uchuu
        cov_glam_scaled = cov_glam*(L_glam/L_uchuu)**3
        #cov_emu = cov_smooth_emuperf - cov_aemulus_5box
        cov_emu = cov_emuperf - cov_aemulus_5box
        cov_combined = cov_emu + cov_glam_scaled
        # now we smooth
        statistics = stat_str.split('_')
        cov_combined = csc.smooth_cov_gaussian(cov_combined, statistics, nbins=9, width=1)

    elif mode=='glam_for_aemulus': 
        # below is for using the glam covmat for recovery on aemulus
        cov_glam_scaled = cov_glam*(1/5)*(L_glam/L_aemulus)**3
        cov_aemulus_5box = cov_aemulus*(1/5)
    
        # combine
        # note: im not sure why i considered glam to be the covmat to subtract off, vs add on;
        # i think it makes sense, bc we were trying to use it for getting a better emu covmat 
        # that we could then apply to any data. but in uchuu case, it's opposite i think
        cov_emu = cov_smooth_emuperf - cov_glam_scaled
        cov_combined = cov_emu + cov_aemulus_5box

    elif mode=='aemulus_for_uchuu':
        # use raw emuperf here, then we'll smooth later
        cov_emu = cov_emuperf - cov_aemulus_5box

        # L_aemulus/L_uchuu because uchuu is larger volume so should have smaller cov
        cov_uchuu = cov_aemulus_5box*(L_aemulus/L_uchuu)**3
        cov_combined = cov_emu + cov_uchuu 
        # now we smooth
        statistics = stat_str.split('_')
        cov_combined = csc.smooth_cov_gaussian(cov_combined, statistics, nbins=9, width=1)

    else:
        print("Mode not recognized!")
    
    # save
    np.savetxt(cov_combined_fn, cov_combined)
    print("Saved to", cov_combined_fn)
    
    
if __name__=='__main__':
    main()
