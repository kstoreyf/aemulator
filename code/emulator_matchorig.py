import time
import numpy as np
import george
from george import kernels
import multiprocessing as mp
import json

import gp_trainer as trainer


class Emulator:

    def __init__(self, statistic, training_dir, testing_dir=None, hyperparams=None, fixed_params={}, nbins=9, gperr=None, testmean=True, log=False, mean=False, meansub=False, xrsq=False, nhod=100, kernel_name=None, nhod_test=100):
        
        #print("george version:", george.__version__) 
        # set parameters
        self.statistic = statistic
        self.fixedparams = fixed_params
        self.nbins = nbins
        self.gps = [None]*nbins
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.testmean = testmean # use the mean of the test boxes (recommended)
        self.log = log
        self.mean = mean
        self.meansub = meansub
        self.xrsq = xrsq
        self.nhod = nhod
        self.nhod_test = nhod_test
        assert not (mean and meansub), "can't have both mean and meansub true!"
        assert kernel_name is not None, "Must specify kernel_name!"
        self.kernel_name = kernel_name

        # load data
        self.load_training_data()
        if self.testing_dir:
            self.load_testing_data()

        # initialize emulator
        self.param_bounds = self.set_param_bounds()
        assert gperr is not None, "Must specify gperr, the error for the GPs!"
        gperr_frac = self.load_file_or_obj(gperr)
        self.gperr = self.scale_error(gperr_frac) 
        if hyperparams:
            self.hyperparams = self.load_file_or_obj(hyperparams) #may still be None
        else:
            kernel = self.get_kernel(np.full(self.nparams, 0.1))
            self.hyperparams = np.empty((nbins, len(kernel)))

    def scale_error(self, fractional_error):
        # absolute error
        error = fractional_error * self.training_mean
        # for log of y, errors are 1/ln(10) * dy/y. dy is error, for y we use the mean.
        # source: https://faculty.washington.edu/stuve/log_error.pdf, https://web.ma.utexas.edu/users/m408n/m408c/CurrentWeb/LM3-6-2.php
        if self.log:
            error = 1/np.log(10) * error / self.training_mean
        return error


    def load_file_or_obj(self, name):
        if type(name)==str:
            return np.loadtxt(name)
        else:
            return name

    def process_data(self, data_orig, bb):
        data = data_orig.copy()
        if self.xrsq:
            data = data * self.rs[bb]**2
        if self.mean:
            data /= self.training_mean[bb]
        if self.meansub:
            data -= self.training_mean[bb]
        if self.log:
            data = np.log10(data)
        return data

    # Make sure consistent with unprocess! 
    # [opposite order and operations]
    def unprocess_data(self, data_orig, bb):
        data = data_orig.copy()
        if self.log:
            data = 10**data
        if self.meansub:
            data += self.training_mean[bb]
        if self.mean:
            data *= self.training_mean[bb]
        if self.xrsq:
            data = data / (self.rs[bb]**2)
        return data


    def set_training_data_mean(self):
        data = self.training_data
        if self.xrsq:
            data = data * self.rs**2
        if self.log:
            data = np.log10(data)
        self.training_mean = np.mean(data, axis=0)


    def predict(self, params_pred):
        #print(params_pred)
        if type(params_pred)==dict:
            params_arr = []
            param_names_ordered = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w',
                                    'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f',
                                   'f_env', 'delta_env', 'sigma_env']
            for pn in param_names_ordered:
                params_arr.append(params_pred[pn])
        elif type(params_pred)==list or type(params_pred)==np.ndarray:
            params_arr = params_pred
        else:
            raise ValueError("Params to predict at must be dict or array")

        params_arr = np.atleast_2d(params_arr)
        y_pred = np.zeros(self.nbins)
        for bb in range(self.nbins):
            # predict on all the training data in the bin
            training_data_pred = self.process_data(self.training_data[:,bb], bb)
            val_pred, cov_pred = self.gps[bb].predict(training_data_pred, params_arr)
            val_pred = self.unprocess_data(val_pred, bb)
            y_pred[bb] = val_pred

        return y_pred


    def build(self):
        #print("Rebuilding emulators")
        for bb in range(self.nbins):
            training_data = self.process_data(self.training_data[:,bb], bb)
            # here i think it makes sense to just take the mean directly, bc have already done the other operations
            # (e.g. may already have divided by the mean, now there's a new mean)
            mean = np.mean(training_data)
            kernel = self.get_kernel(np.full(self.nparams, 0.1))
            gp = self.init_gp(self.training_params, self.gperr[bb], kernel, mean=mean)
            self.gps[bb] = self.set_hyperparams(gp, self.training_params, 
                            self.gperr[bb], self.hyperparams[bb])


    def train_serial(self, save_hyperparams_fn):
        start = time.time()
        print("Training commences!")
        for bb in range(self.nbins):
            print(f"Training bin {bb}")
            training_data = self.process_data(self.training_data[:,bb], bb)            
            mean = np.mean(training_data)
            kernel = self.get_kernel(np.full(self.nparams, 0.1))
            gp = self.init_gp(self.training_params, self.gperr[bb], kernel, mean=mean)
            if george.__version__=='0.3.1':
                tr = trainer.gp_tr(self.training_params, 
                        training_data, self.gperr[bb], 
                        gp, optimize=True)
                hyps = tr.gp.kernel.get_parameter_vector()
            elif george.__version__=='0.2.1':
                hyps = trainer.gp_tr(self.training_params, 
                        training_data, self.gperr[bb], 
                        gp, optimize=True).p_op
            self.hyperparams[bb, :] = hyps

        print("Done training!")
        np.savetxt(save_hyperparams_fn, self.hyperparams, fmt='%.7f')
        print(f"Saved hyperparameters to {save_hyperparams_fn}")
        end = time.time()
        print(f"Time: {(end-start)/60.0} min")

    def train(self, save_hyperparams_fn, nthreads=None):
        start = time.time()
        print("Training commences!")
        if not nthreads:
            nthreads = self.nbins
        print("Constructing pool")
        pool = mp.Pool(processes=nthreads)
        print("Mapping bins")
        res = pool.map(self.train_bin, range(self.nbins))
        print("Done training!")
        print(np.array(res).shape)
        for bb in range(self.nbins):
            self.hyperparams[bb, :] = res[bb]
        np.savetxt(save_hyperparams_fn, self.hyperparams, fmt='%.7f')
        print(f"Saved hyperparameters to {save_hyperparams_fn}")
        end = time.time()
        print(f"Time: {(end-start)/60.0} min")

    def train_bin(self, bb):
        print(f"Training bin {bb}")
        training_data = self.process_data(self.training_data[:,bb], bb)
        mean = np.mean(training_data)
        kernel = self.get_kernel(np.full(self.nparams, 0.1))
        gp = self.init_gp(self.training_params, self.gperr[bb], kernel, mean=mean)
        if george.__version__=='0.3.1':
            tr = trainer.gp_tr(self.training_params, 
                    training_data, self.gperr[bb], 
                    gp, optimize=True)
            hyps = tr.gp.kernel.get_parameter_vector()
        elif george.__version__=='0.2.1':
            hyps = trainer.gp_tr(self.training_params, 
                    training_data, self.gperr[bb], 
                    gp, optimize=True).p_op
        return hyps

    def test(self, predict_savedir):
        if not self.testing_dir:
            raise ValueError('Must provide testing directory in emulator constructor!')
        for pid, tparams in self.testing_params.items():
            vals_pred = self.predict(tparams)
            if self.testmean:
                idtag = "cosmo_{}_HOD_{}_mean".format(pid[0], pid[1])
            else:
                idtag = "cosmo_{}_Box_{}_HOD_{}_test_{}".format(pid[0], boxid, pid[1], testid)

            pred_fn = f"{predict_savedir}/{self.statistic}_{idtag}.dat"

            results = np.array([self.testing_radii, vals_pred])
            np.savetxt(pred_fn, results.T, delimiter=',', fmt=['%f', '%e']) 

    def test_glam(self, predict_savedir):
        param_file = 'glam_params.json'
        testing_dir = f'/home/users/ksf293/clust/results_glam/results_glam_{self.statistic}'
        with open(param_file, 'r') as jfile:
            glam_params = json.load(jfile)
            glam_params['M_sat'] = np.log10(glam_params['M_sat'])
            glam_params['M_cut'] = np.log10(glam_params['M_cut'])
        vals_pred = self.predict(glam_params)

        # load in first mock to get radii
        fnt = f"{testing_dir}/{self.statistic}_glam_n0.dat"
        radii, _ = np.loadtxt(fnt, delimiter=',', unpack=True)
        results = np.array([radii, vals_pred])

        pred_fn = f"{predict_savedir}/{self.statistic}_glam.dat"
        np.savetxt(pred_fn, results.T, delimiter=',', fmt=['%f', '%e']) 


    def test_glam4(self, predict_savedir):
        param_file = 'glam_params.json'
        testing_dir = f'/home/users/ksf293/clust/results_glam4/results_glam4_{self.statistic}'
        # glam3
        #glam_hod_fn = '/mount/sirocco2/zz681/emulator/CMASSLOWZ_Msat/test_mocks/HOD_test_np11_n5000_new_f_env_Msat.dat'
        # glam4
        glam_hod_fn = '/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat'
        Nmocks = 986
         
        # get cosmology params
        cosmo_param_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
        with open(param_file, 'r') as jfile:
            glam_params = json.load(jfile)
            cosmos_glam = np.array([glam_params[cpn] for cpn in cosmo_param_names])
            ncosmoparams_glam = len(cosmos_glam)

        # get hod params
        hods_glam = np.loadtxt(glam_hod_fn, usecols=range(8)) #only first 8 bc last 3 assembly bias fixed
        nhodparams_glam = hods_glam.shape[1]
        hods_glam[:,0] = np.log10(hods_glam[:,0])
        hods_glam[:,2] = np.log10(hods_glam[:,2])
        abs_glam = [0, 1, 0.5]
        # load in first mock to get radii (all same)
        fnt = f"{testing_dir}/{self.statistic}_glam4_n0.dat"
        radii, _ = np.loadtxt(fnt, delimiter=',', unpack=True)
        
        # make and save emu predictions for all mocks
        for n in range(Nmocks):
            print(f"Testing stat {self.statistic} GLAM mock {n}")
            hod_glam = hods_glam[n]
            ch_params = np.concatenate((cosmos_glam, hod_glam))
            params = np.concatenate((ch_params, abs_glam))
            vals_pred = self.predict(params)
            results = np.array([radii, vals_pred])

            pred_fn = f"{predict_savedir}/{self.statistic}_glam4_n{n}.dat"
            np.savetxt(pred_fn, results.T, delimiter=',', fmt=['%f', '%e'])
        

    def load_training_data(self):
        #print("Loading training data")
        # hod parameters (5000 rows, 8 cols)
        hods = np.loadtxt("../tables/HOD_design_np11_n5000_new_f_env.dat")

        hods[:, 0] = np.log10(hods[:, 0])
        hods[:, 2] = np.log10(hods[:, 2])
        nhodparams = hods.shape[1]
        nhodnonolap = self.nhod
        # cosmology params (40 rows, 7 cols)
        cosmos = np.loadtxt("../tables/cosmology_camb_full.dat")
        ncosmoparams = cosmos.shape[1]

        CC = range(0, cosmos.shape[0])
        nhodpercosmo = 100

        HH = np.array(range(0, len(CC) * nhodpercosmo))
        HH = HH.reshape(len(CC), nhodpercosmo)
        HH = HH[:, 0:nhodnonolap]

        self.nparams = nhodparams + ncosmoparams
        #print(f"Nparams: {self.nparams}")
        self.ndata = HH.shape[1] * cosmos.shape[0]

        self.training_params = np.empty((self.ndata, self.nparams))
        self.training_data = np.empty((self.ndata, self.nbins))

        idata = 0
        for CID in CC:
            HH_set = HH[CID]
            for HID in HH_set:
                HID = int(HID)
                # training data is all test0 (always)
                rs, vals = np.loadtxt(self.training_dir + "{}_cosmo_{}_HOD_{}_test_0.dat".format(self.statistic, CID, HID),
                                       delimiter=',', unpack=True)
                vals = vals[:self.nbins]
                self.training_data[idata,:] = vals

                param_arr = np.concatenate((cosmos[CID,:], hods[HID,:]))
                self.training_params[idata, :] = param_arr 
                idata += 1

        self.rs = rs #rs are all the same so just take the last one
        # set mean of values in each bin (training_mean has length nbins)
        self.set_training_data_mean()



    def load_testing_data(self):
        #print("Loading testing data")
        
        hods_test = np.loadtxt("/mount/sirocco2/zz681/emulator/CMASSLOWZ/test_galaxy_mocks_wp_RSD/test_galaxy_mocks_new_f_env/HOD_test_np11_n1000_new_f_env.dat")
        nhodparams_test = hods_test.shape[1]
        hods_test[:,0] = np.log10(hods_test[:,0])
        hods_test[:,2] = np.log10(hods_test[:,2])
        cosmos_test = np.loadtxt("../CMASS/Gaussian_Process/hod_file/cosmology_camb_test_box_full.dat")
        ncosmoparams_test = cosmos_test.shape[1]

        CC_test = range(0, 7)
        # TODO: add more tests, for now just did first 10 hod
        HH_test = range(0, self.nhod_test)

        self.nparams_test = nhodparams_test + ncosmoparams_test
        #print(f"Nparams: {self.nparams_test}")

        self.testing_params = {}
        self.testing_data = {} # this is never used! just params

        for CID_test in CC_test:
            for HID_test in HH_test:

                if self.testmean:
                    idtag = "cosmo_{}_HOD_{}_mean".format(CID_test, HID_test)
                    rads, vals_test = np.loadtxt(self.testing_dir + "{}_{}.dat".format(self.statistic, idtag), 
                                                  delimiter=',', unpack=True)
                else:
                    idtag = "cosmo_{}_Box_{}_HOD_{}_test_{}".format(CID_test, boxid, HID_test, testid)
                    rads, vals_test = np.loadtxt(self.testing_dir + "{}_{}.dat".format(self.statistic, idtag),
                                                  delimiter=',', unpack=True)

                pid = (CID_test, HID_test)
                self.testing_data[pid] = vals_test
                param_arr = np.concatenate((cosmos_test[CID_test,:], hods_test[HID_test,:]))
                self.testing_params[pid] = param_arr
                #TODO: really only need to set this once
                self.testing_radii = rads


    def init_gp(self, training_params, err, kernel, mean=None):
        gp = george.GP(kernel, mean=mean, solver=george.BasicSolver)
        gp.compute(training_params, err)
        return gp


    def set_hyperparams(self, gp, training_params, err, hyperparams):
        if george.__version__=='0.3.1':
            gp.set_parameter_vector(hyperparams)
        elif george.__version__=='0.2.1':
            gp.kernel.vector = hyperparams
        gp.compute(training_params, err)
        return gp


    # 15 initial values for the 7 hod and 8 cosmo params
    def get_kernel(self, p0):
        if george.__version__=='0.3.1':
            p0 = np.exp(p0) 
        k1 = kernels.ExpSquaredKernel(p0, ndim=len(p0))
        k2 = kernels.Matern32Kernel(p0, ndim=len(p0))
        k3 = kernels.ConstantKernel(0.1, ndim=len(p0))
        #k4 = kernels.WhiteKernel(0.1, ndim=len(p0))
        k5 = kernels.ConstantKernel(0.1, ndim=len(p0))
        
        kernel_dict = {'M32ExpConst': k1*k5 + k2,
                   'M32ExpConst2': k1*k5 + k2 + k3,
                   'M32Const': k2 + k5}
        assert self.kernel_name in kernel_dict, f"{self.kernel_name} not in dict!"
        kernel = kernel_dict[self.kernel_name]
        return kernel


    def get_param_bounds(self, pname):
        return self.param_bounds[pname]


    def set_param_bounds(self):
        bounds = {}
        cosmo_names = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w']
        hod_names = ['M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f', 'f_env', 'delta_env', 'sigma_env']

        cosmos_train = np.loadtxt('../tables/cosmology_camb_full.dat') # 40
        hods_train = np.loadtxt('../tables/HOD_design_np11_n5000_new_f_env.dat') # 5000
        hods_train[:, 0] = np.log10(hods_train[:, 0])
        hods_train[:, 2] = np.log10(hods_train[:, 2])

        for pname in cosmo_names:
            pidx = cosmo_names.index(pname)
            vals = cosmos_train[:,pidx]
            pmin = np.min(vals)
            pmax = np.max(vals)
            buf = (pmax-pmin)*0.1
            bounds[pname] = [pmin-buf, pmax+buf]

        for pname in hod_names:
            pidx = hod_names.index(pname)
            vals = hods_train[:,pidx]
            pmin = np.min(vals)
            pmax = np.max(vals)
            buf = (pmax-pmin)*0.1
            bounds[pname] = [pmin-buf, pmax+buf]
        
        return bounds
