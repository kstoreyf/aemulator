import numpy as np

# variables
nbins = 9
rbins = np.logspace(np.log10(0.1), np.log10(50), nbins + 1) # Note the + 1 to nbins
rlog = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
rlin = np.linspace(5, 45, 9)
r_dict = {'wp': rlog, 'xi': rlog, 'upf': rlin, 'mcf': rlog}
scale_dict = {'wp': ('log', 'log'), 'xi': ('log', 'log'), 'upf': ('linear', 'log'), 'mcf': ('log', 'linear')} #x, y
stat_labels = {'upf':r"P$_U$(r)", 'wp':r'$w_p$($r_p$)', 'mcf':"M($r$)", 'xi':r"$\xi_0$($r$)", 'xi2':r"$\xi_2$($r$)"}
r_labels = {'upf':r"$r$", 'wp':r'$r_p$', 'mcf':r"$r$", 'xi':r"$r$", 'xi2':r"$r$"}