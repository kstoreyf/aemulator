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


def run(mock_tag_cov, stat_str, mode, cov_tag_extra='', inflate_upf_err=False):

    mock_name_glam = 'glam'
    cov_dir = '../covariances'
    #comb_tag = '_smooth_covnegfix'
    comb_tag = '_smoothemuboth'
    #comb_tag = '_smoothemuperf'
    #comb_tag = '_smoothboth'
    #comb_tag = '_smooth'
    if inflate_upf_err and 'upf' in stat_str:
        inflate_factor = 2
        comb_tag += f'_inflateupferr{inflate_factor}nox'

    if mode=='glam_for_uchuu':
        cov_combined_fn = f"{cov_dir}/cov_combined{mock_tag_cov}{cov_tag_extra}_uchuu{mock_name_glam}{comb_tag}_{stat_str}.dat"
    elif mode=='glam_for_aemulus':
        cov_glam_fn = f'{cov_dir}/cov_{mock_name_glam}_{stat_str}.dat'
        cov_combined_fn = f"{cov_dir}/cov_combined{mock_tag_cov}{cov_tag_extra}_{mock_name_glam}{comb_tag}_{stat_str}.dat"
    elif mode=='aemulus_for_uchuu':
        cov_combined_fn = f"{cov_dir}/cov_combined{mock_tag_cov}{cov_tag_extra}_uchuu{comb_tag}_{stat_str}.dat"

    if 'glam' in mode:
        cov_glam_fn = f'{cov_dir}/cov_{mock_name_glam}_{stat_str}.dat'
        cov_glam = np.loadtxt(cov_glam_fn)

    cov_aemulus_fn = f'{cov_dir}/cov{mock_tag_cov}_{stat_str}_hod3_test0.dat'
    # TODO: check if i want to be using smoothgauss here! wasnt before, bc only smoothed after glam failed
    if 'smoothboth' in comb_tag or 'smoothemuperf' in comb_tag:
        cov_emuperf_fn = f'{cov_dir}/cov_smoothgauss_emuperf{mock_tag_cov}{cov_tag_extra}_{stat_str}_hod3_test0.dat'
    else:
        cov_emuperf_fn = f'{cov_dir}/cov_emuperf{mock_tag_cov}{cov_tag_extra}_{stat_str}_hod3_test0.dat'

    # load covs
    cov_aemulus = np.loadtxt(cov_aemulus_fn)
    cov_emuperf = np.loadtxt(cov_emuperf_fn)
    #cov_smooth_emuperf = np.loadtxt(cov_smooth_emuperf_fn)
    
    # rescale aemulus cov bc tested emu on mean of 5 boxes
    cov_aemulus_5box = cov_aemulus*(1/5)
    
    # rescale covariances 
    L_uchuu = 2000.
    L_glam = 1000.
    L_aemulus = 1050.
    if mode=='glam_for_uchuu':
        # scale glam to uchuu, for recovery on uchuu
        cov_data = cov_glam*(L_glam/L_uchuu)**3
        #cov_emu = cov_smooth_emuperf - cov_aemulus_5box
        cov_emu = cov_emuperf - cov_aemulus_5box

    elif mode=='glam_for_aemulus': 
        # below is for using the glam covmat for recovery on aemulus
        cov_glam_scaled = cov_glam*(1/5)*(L_glam/L_aemulus)**3
        cov_data = cov_aemulus_5box
    
        # combine
        # note: im not sure why i considered glam to be the covmat to subtract off, vs add on;
        # i think it makes sense, bc we were trying to use it for getting a better emu covmat 
        # that we could then apply to any data. but in uchuu case, it's opposite i think
        cov_emu = cov_emuperf - cov_glam_scaled

    elif mode=='aemulus_for_uchuu':
        # use raw emuperf here, then we'll smooth later
        cov_emu = cov_emuperf - cov_aemulus_5box

        # L_aemulus/L_uchuu because uchuu is larger volume so should have smaller cov
        cov_data = cov_aemulus_5box*(L_aemulus/L_uchuu)**3

    else:
        print("Mode not recognized!")
    
    statistics = stat_str.split('_')
    if inflate_upf_err and 'upf' in stat_str:
        print("Inflating upf error")
        nbins = 9
        i_upf = statistics.index('upf')
        i_start = i_upf*nbins 
        i_end = i_start + nbins
        print(statistics)
        print(i_start, i_end)
        print(cov_data[i_start:i_end, i_start:i_end])
        upf_mask = np.full(cov_data.shape, False)
        # these two lines also inflate all cross terms
        #upf_mask[i_start:i_end,:] = True
        #upf_mask[:,i_start:i_end] = True 
        # this line does just the upf-upf square ("nox")
        upf_mask[i_start:i_end,i_start:i_end] = True
        # this does just diag 
        # for i in range(i_start, i_end):
        #     upf_mask[i,i] = True
        cov_data[upf_mask] *= inflate_factor**2 #because we're scaling the variances
        print(cov_data[i_start:i_end, i_start:i_end])

    # cov emu could be (a bit) negative if performance better than 
    # sample variance estimate; set to val of one above/below it #hack for now
    i_neg = np.diag(cov_emu) < 0
    if np.sum(i_neg)>0:
        print("Fixing negative cov_emu values!")
        idx_neg = np.arange(len(i_neg))[i_neg]
        print("idx_neg:", idx_neg)
        for ii in idx_neg:
            if ii<cov_emu.shape[0]-1:
                val_fix = cov_emu[ii+1,ii+1]
            else:
                val_fix = cov_emu[ii-1,ii-1]
            cov_emu[ii,ii] = val_fix
    print('emu fixed:', cov_emu)

    # must be equal to not get emuperf
    if comb_tag=='_smoothemu' or comb_tag=='_smoothemuboth':
        cov_emu = csc.smooth_cov_gaussian(cov_emu, statistics, nbins=9, width=1)
    cov_combined = cov_emu + cov_data

    print("nans?")
    print(np.sum(np.isnan(cov_emu)))
    print(np.sum(np.isnan(cov_data)))
    print(np.sum(np.isnan(cov_combined)))


    # not smoothemu or smoothemuperf
    if comb_tag=='_smooth' or comb_tag=='_smoothboth' or comb_tag=='_smoothemuboth':
        print("Smoothing final cov")
        cov_combined = csc.smooth_cov_gaussian(cov_combined, statistics, nbins=9, width=1)
    print(np.sum(np.isnan(cov_combined)))
    print('combined:', cov_combined)

    # save
    np.savetxt(cov_combined_fn, cov_combined)
    print("Saved to", cov_combined_fn)
    
    
if __name__=='__main__':
    main()
