import numpy as np
import os

cov_dir = '../covariances'

#statistics = ['wp', 'xi', 'xi2', 'upf', 'mcf']
statistics = ['xi2']
stat_str = '_'.join(statistics)
mode = 'glam_for_uchuu'
#mode = 'glam_for_aemulus'
if mode=='glam_for_uchuu':
    cov_combined_fn = f"{cov_dir}/cov_combined_uchuuglam4_{stat_str}.dat"
elif mode=='glam_for_aemulus':
    # note: before i was using just glam! maybe should have been glam4?
    cov_combined_fn = f"{cov_dir}/cov_combined_glam4_{stat_str}.dat"

cov_glam_fn = f'{cov_dir}/cov_glam4_{stat_str}.dat'
cov_aemulus_fn = f'{cov_dir}/cov_aemulus_{stat_str}_hod3_test0.dat'
#cov_emuperf_fn = f'{cov_dir}/cov_emuperf_{stat_str}_nonolap_hod3_test0_mean_test0.dat'
# TODO: check if i want to be using smoothgauss here! wasnt before, bc only smoothed after glam failed
cov_emuperf_fn = f'{cov_dir}/cov_smoothgauss_emuperf_{stat_str}_hod3_test0.dat'

# load covs
cov_glam = np.loadtxt(cov_glam_fn)
cov_aemulus = np.loadtxt(cov_aemulus_fn)
cov_emuperf = np.loadtxt(cov_emuperf_fn)

# rescale aemulus cov bc tested emu on mean of 5 boxes
cov_aemulus_5box = cov_aemulus*(1/5)

# rescale covariances 
L_uchuu = 2000.
L_glam = 1000.
L_aemulus = 1050.
if mode=='glam_for_uchuu':
    # scale glam to uchuu, for recovery on uchuu
    cov_glam_scaled = cov_glam*(L_glam/L_uchuu)**3
    cov_emu = cov_emuperf - cov_aemulus_5box
    cov_combined = cov_emu + cov_glam_scaled

elif mode=='glam_for_aemulus': 
    # below is for using the glam covmat for recovery on aemulus
    cov_glam_scaled = cov_glam*(1/5)*(L_glam/L_aemulus)**3
    cov_aemulus_5box = cov_aemulus*(1/5)

    # combine
    # note: im not sure why i considered glam to be the covmat to subtract off, vs add on;
    # i think it makes sense, bc we were trying to use it for getting a better emu covmat 
    # that we could then apply to any data. but in uchuu case, it's opposite i think
    cov_emu = cov_emuperf - cov_glam_scaled
    cov_combined = cov_emu + cov_aemulus_5box
else:
    print("Mode not recognized!")

# save
np.savetxt(cov_combined_fn, cov_combined)
print("Saved to", cov_combined_fn)



