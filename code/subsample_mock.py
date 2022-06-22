from email.policy import default
import numpy as np
from numpy.random import default_rng

mock_dir = '/mount/sirocco1/ksf293/aemulator/mocks_nseries'

def load_random():
    res = np.loadtxt(f'{mock_dir}/Nseries_cutsky_randoms_50x_redshifts.dat', unpack=True)
    ra, dec, redshift, _ = res
    return ra, dec, redshift

print('loading')
ra_rand, dec_rand, z_rand = load_random()
N_all = len(ra_rand)
print('loaded, N =', N_all)

factor = 20 #compared to data catalog! not random; 50 should give exact same as already
frac_subsample = factor/50
N_sub = int(frac_subsample*N_all)
idx_all = np.arange(N_all)

rng = default_rng()
idx_keep = rng.choice(idx_all, size=N_sub, replace=False)
print('subsampling to N =', N_sub)

ra_keep = ra_rand[idx_keep]
dec_keep = dec_rand[idx_keep]
z_keep = z_rand[idx_keep]

fn_save = f'{mock_dir}/Nseries_cutsky_randoms_{factor}x_redshifts.dat'

results = np.array([ra_keep, dec_keep, z_keep]).T
np.savetxt(fn_save, results)

print("Saved to", fn_save)