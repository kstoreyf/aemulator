import numpy as np


# variables
nbins = 9
rbins = np.logspace(np.log10(0.1), np.log10(50), nbins + 1) # Note the + 1 to nbins
rlog = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
rlin = np.linspace(5, 45, 9)
r_dict = {'wp': rlog, 'xi': rlog, 'upf': rlin, 'mcf': rlog, 'xi2': rlog}
scale_dict = {'wp': ('log', 'log'), 'xi': ('log', 'log'), 'upf': ('linear', 'log'), 'mcf': ('log', 'linear'), 'xi2': ('log', 'linear')} #x, y
stat_labels = {'upf':r"P$_U$(r)", 'wp':r'$w_p$($r_p$)', 'mcf':"M($r$)", 'xi':r"$\xi_0$($r$)", 'xi2':r"$r^2 \xi_2$($r$)"}
r_labels = {'upf':r"$r (h^{-1}$Mpc)", 'wp':r'$r_p (h^{-1}$Mpc)', 'mcf':r"$r (h^{-1}$Mpc)", 'xi':r"$r (h^{-1}$Mpc)", 'xi2':r"$r (h^{-1}$Mpc)$"}

def get_emu(traintag):
    import emulator
    emu_name = traintag.split('_')[1]
    print("name:", emu_name)
    emu_dict = {'MLP': emulator.EmulatorMLP,
                'GPFlow': emulator.EmulatorGPFlow, 
                'GPFlowBinned': emulator.EmulatorGPFlowBinned,
                'George': emulator.EmulatorGeorge}
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