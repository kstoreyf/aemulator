import numpy as np


# variables
nbins = 9
rbins = np.logspace(np.log10(0.1), np.log10(50), nbins + 1) # Note the + 1 to nbins
rlog = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
rlin = np.linspace(5, 45, 9)
r_dict = {'wp': rlog, 'xi': rlog, 'upf': rlin, 'mcf': rlog, 'xi2': rlog}
scale_dict = {'wp': ('log', 'log'), 'xi': ('log', 'log'), 'upf': ('linear', 'log'), 'mcf': ('log', 'linear'), 'xi2': ('log', 'linear')} #x, y
stat_labels = {'wp':r'$w_p$($r_p$)', 'upf':r"P$_U$(r)", 'mcf':r"M($r$)", 'xi':r"$\xi_0$($r$)", 'xi2':r"$\xi_2$($r$)"}
r_labels = {'wp':r'$r_p (h^{-1}$Mpc)', 'upf':r"$r (h^{-1}$Mpc)", 'mcf':r"$r (h^{-1}$Mpc)", 'xi':r"$r (h^{-1}$Mpc)", 'xi2':r"$r (h^{-1}$Mpc)"}

cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
key_param_names = ['Omega_m', 'sigma_8', 'M_sat', 'v_bc', 'v_bs', 'f', 'f_env']
ab_param_names = ["f_env", "delta_env", "sigma_env"]
param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w", \
               "M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
param_labels = {'Omega_m': '\Omega_m',
                'Omega_b': '\Omega_b',
                'sigma_8': '\sigma_8',
                'h': 'h',
                'n_s': 'n_s',
                'N_eff': 'N_{eff}',
                'w': 'w',
                'M_sat': 'M_{sat}',
                'alpha': r'\alpha',
                'M_cut': 'M_{cut}',
                'sigma_logM': '\sigma_{logM}',
                'v_bc': 'v_{bc}',
                'v_bs': 'v_{bs}',
                'c_vir': 'c_{vir}',
                'f': '\gamma_f',
                'f_env': 'f_{env}',
                'delta_env': '\delta_{env}',
                'sigma_env': '\sigma_{env}'}

def get_emu(emu_name):
    import emulator
    emu_dict = {'MLP': emulator.EmulatorMLP,
                'GPFlow': emulator.EmulatorGPFlow, 
                'GPFlowVGP': emulator.EmulatorGPFlowVGP,
                'GPFlowBinned': emulator.EmulatorGPFlowBinned,
                'George': emulator.EmulatorGeorge,
                'GeorgeOrig': emulator.EmulatorGeorgeOrig,
                'PyTorch': emulator.EmulatorPyTorch}
    return emu_dict[emu_name]

def load_cosmo_params():
    # 7 cosmo params
    cosmo_param_names = ["Omega_m", "Omega_b", "sigma_8", "h", "n_s", "N_eff", "w"]
    cosmo_params = np.loadtxt('../tables/cosmology_camb_test_box_full.dat')
    return cosmo_param_names, cosmo_params

def load_hod_params():
    # 11 cosmo params
    hod_param_names = ["M_sat", "alpha", "M_cut", "sigma_logM", "v_bc", "v_bs", "c_vir", "f", "f_env", "delta_env", "sigma_env"]
    hod_params = np.loadtxt('../tables/HOD_test_np11_n1000_new_f_env.dat')
    hod_params[:, 0] = np.log10(hod_params[:, 0])
    hod_params[:, 2] = np.log10(hod_params[:, 2])
    return hod_param_names, hod_params

# Prior is the min and max of training set parameters, +/- 10% on either side
def get_hod_bounds():
    hod_bounds = {}
    hod_param_names, hod_params = load_hod_params()
    for pname in hod_param_names:
        pidx = hod_param_names.index(pname)
        vals = hod_params[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        # Add a 10% buffer on either side of training set
        buf = (pmax-pmin)*0.1
        hod_bounds[pname] = [pmin-buf, pmax+buf]
    return hod_bounds

def get_cosmo_bounds():
    cosmo_bounds = {}
    cosmo_param_names, cosmo_params = load_cosmo_params()
    for pname in cosmo_param_names:
        pidx = cosmo_param_names.index(pname)
        vals = cosmo_params[:,pidx]
        pmin = np.min(vals)
        pmax = np.max(vals)
        # Add a 10% buffer on either side of training set
        buf = (pmax-pmin)*0.1
        cosmo_bounds[pname] = [pmin-buf, pmax+buf]
    return cosmo_bounds

def get_bounds():
    bounds = get_hod_bounds()
    bounds.update(get_cosmo_bounds())
    return bounds

def make_label(statistics):
    if type(statistics) is str:
        return stat_labels[statistics]
    else:
        stats_nice = [stat_labels[s] for s in statistics]
        return ' + '.join(stats_nice)

def get_fiducial_emu_name(statistic):
    emu_name_dict = {'wp': 'George',
                     'xi': 'George',
                     'upf': 'George',
                     'mcf': 'George',
                     'xi2': 'George'}
    return emu_name_dict[statistic]

def get_fiducial_emu_scaling(statistic):
    emu_scaling_dict = {'wp': 'log',
                    'xi': 'log',
                    'upf': 'log',
                    'mcf': 'log',
                    'xi2': 'xrsqmean'}
    return emu_scaling_dict[statistic]

def get_nthreads(n_statistics):
    if n_statistics<=3:
        return 24
    elif n_statistics==4:
        return 18
    elif n_statistics==5:
        return 14
    else:
        print("Don't know how many threads should use for >5 emus, defaulting to 1")
        return 1
