import gc
import numpy as np
import os
import pickle
import scipy
import multiprocessing as mp

import george
import joblib
from sklearn.preprocessing import StandardScaler, FunctionTransformer  

import utils

# from sklearn.neural_network import MLPRegressor

# import gpflow
# import tensorflow as tf
# from gpflow.utilities import print_summary
# from gpflow.ci_utils import ci_niter
# from gpflow.optimizers import NaturalGradient
# from gpflow import set_trainable

# import torch
# from torch import nn
# from torch.autograd import Variable


class Emulator(object):

    def __init__(self, statistic, scaling, model_fn, scaler_x_fn, scaler_y_fn,
                 err_fn, bins=list(range(9)), 
                 train_mode=False, test_mode=False, predict_mode=False,
                 mock_tag_train='_aemulus_Msatmocks_train', mock_tag_test='_aemulus_Msatmocks_test'):
        assert np.any(np.array([train_mode, test_mode, predict_mode])), "At least one mode must be True!"
        self.statistic = statistic
        self.model_fn = model_fn
        self.scaler_x_fn = scaler_x_fn
        self.scaler_y_fn = scaler_y_fn
        # TODO: i think only need err_fn when in train mode; check when need other inputs
        self.err_fn = err_fn
        self.bins = bins
        self.n_bins = len(bins)
        self.n_bins_tot = 9
        if self.n_bins < self.n_bins_tot:
            assert "George" in model_fn, "Using fewer than all the bins only implemented for George emu!"
        self.mock_tag_train = mock_tag_train
        print(self.mock_tag_train)
        self.mock_tag_test = mock_tag_test
        assert self.mock_tag_train in ['_aemulus_train', '_aemulus_Msatmocks_train'], 'Training mock tag not recognized!'
        assert self.mock_tag_test in ['_aemulus_test', '_aemulus_Msatmocks_test'], 'Testing mock tag not recognized!'

        self.set_training_data() # always need training data, for error (and GP conditioning)
        self.load_y_error()
        self.scale_y_error(scaling)

        self.mock_tag_test = mock_tag_test
        # for training emulator initially
        if train_mode:
            self.construct_scalers(scaling)
            self.save_scalers()
        # for running accuracy test on full test suite
        if test_mode:
            self.load_scalers()
            self.set_testing_data()
            self.scale_testing_data()
        # for making one-off predictions (i.e. when running MCMC chains)
        if predict_mode: 
            self.load_scalers()
        self.scale_training_data() # need to scale training data for George emu, so might as well always
        self.param_names_ordered = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w',
                                    'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f',
                                   'f_env', 'delta_env', 'sigma_env']

    def load_y_error(self):
        # TODO: check what this error is / should be - std or variance?
        y_error_frac = np.loadtxt(self.err_fn)
        # our y error is fractional, so we first multiply by the mean to make it absolute:
        self.y_error = y_error_frac * np.mean(self.y_train, axis=0)

    def set_training_data(self):
        
        ### ID values (cosmo and hod numbers)
        # ok to pass model_fn instead of train tag bc the former contains the latter
        self.id_pairs_train = utils.load_id_pairs_train(self.mock_tag_train, self.model_fn)
        self.n_train = len(self.id_pairs_train)
        print("N train:", self.n_train)
        ### x values (data, cosmo and hod values)

        cosmos_train_fn = '../tables/cosmology_camb_full.dat'
        cosmos_train = np.loadtxt(cosmos_train_fn)
        n_cosmo_params = cosmos_train.shape[1]

        # original emus
        if self.mock_tag_train=='_aemulus_train':
            hods_train_fn = '../tables/HOD_design_np11_n5000_new_f_env.dat'
        elif self.mock_tag_train=='_aemulus_Msatmocks_train':
            # updated Msat 
            hods_train_fn = '/mount/sirocco2/zz681/emulator/CMASSLOWZ_Msat/training_mocks/HOD_design_np11_n5000_new_f_env_Msat.dat'
        elif self.mock_tag_train=='_aemulus_fmaxmocks_train':
            hods_train_fn = '/mount/sirocco1/zz681/emulator/CMASSLOWZ_Msat_fmax_new/training_mocks/HOD_design_np11_n5000_new_f_env_Msat_fmax_new.dat'
        
        hods_train = np.loadtxt(hods_train_fn)
        # Convert these columns (0: M_sat, 2: M_cut) to log to reduce range
        hods_train[:, 0] = np.log10(hods_train[:, 0])
        hods_train[:, 2] = np.log10(hods_train[:, 2])
        n_hod_params = hods_train.shape[1]

        self.n_params = n_cosmo_params + n_hod_params

        self.x_train = np.empty((self.n_train, self.n_params))
        for i in range(self.n_train):
            id_cosmo, id_hod = self.id_pairs_train[i]
            self.x_train[i,:] = np.concatenate((cosmos_train[id_cosmo], hods_train[id_hod]))

        ### y values (labels, value of statistics in each bin)

        self.y_train = np.empty((self.n_train, self.n_bins_tot))
        y_train_dir = f'/home/users/ksf293/clust/results{self.mock_tag_train}'
        for i in range(self.n_train):
            id_cosmo, id_hod = self.id_pairs_train[i]
            y_train_fn = f'{y_train_dir}/results_{self.statistic}/{self.statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_test_0.dat'
            r_vals, y = np.loadtxt(y_train_fn, delimiter=',', unpack=True)
            if id_cosmo==12 and id_hod==1296:
                print("Bad ID:", y)
            self.y_train[i,:] = y
        # all r_vals are the same so just save the last one
        self.r_vals = r_vals

        #FOR TESTING PURPOSES ONLY
        # print("TINY TRAINING SET")
        # print(self.x_train.shape)
        # self.n_train = 10
        # self.x_train = self.x_train[:self.n_train,:]
        # self.y_train = self.y_train[:self.n_train,:]
        # print(self.x_train.shape)


    def set_testing_data(self):

        ### ID values (cosmo and hod numbers)

        self.id_pairs_test = utils.load_id_pairs_test(self.model_fn)
        self.n_test = self.id_pairs_test.shape[0]
        print("N test ids:", self.n_test)
        ### x values (data, cosmo and hod values)

        cosmos_test_fn = '../tables/cosmology_camb_test_box_full.dat'
        cosmos_test = np.loadtxt(cosmos_test_fn)
        n_cosmo_params = cosmos_test.shape[1]

        # original
        if self.mock_tag_test=='_aemulus_test':
            hods_test_fn = '../tables/HOD_test_np11_n1000_new_f_env.dat'
        elif self.mock_tag_test=='_aemulus_Msatmocks_test':
            # updated msat
            hods_test_fn = '/mount/sirocco2/zz681/emulator/CMASSLOWZ_Msat/test_mocks/HOD_test_np11_n5000_new_f_env_Msat.dat'
        hods_test = np.loadtxt(hods_test_fn)
        # Convert these columns (0: M_sat, 2: M_cut) to log to reduce range
        hods_test[:, 0] = np.log10(hods_test[:, 0])
        hods_test[:, 2] = np.log10(hods_test[:, 2])
        n_hod_params = hods_test.shape[1]

        self.n_params = n_cosmo_params + n_hod_params

        self.x_test = np.empty((self.n_test, self.n_params))
        for i in range(self.n_test):
            id_cosmo, id_hod = self.id_pairs_test[i]
            self.x_test[i,:] = np.concatenate((cosmos_test[id_cosmo], hods_test[id_hod]))

        ### y values (labels, value of statistics in each bin)
        # Note: here we are using the mean of 5 boxes with the same parameters

        self.n_bins_tot = 9
        self.y_test = np.empty((self.n_test, self.n_bins_tot))
        y_test_dir = f'/home/users/ksf293/clust/results{self.mock_tag_test}_mean'
        for i in range(self.n_test):
            id_cosmo, id_hod = self.id_pairs_test[i]
            y_test_fn = f'{y_test_dir}/results_{self.statistic}/{self.statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_mean.dat'
            _, y = np.loadtxt(y_test_fn, delimiter=',', unpack=True)
            self.y_test[i,:] = y

    def pow10(self, x):
        return np.power(10,x)
    
    def times_rsq(self, x):
        return x * self.r_vals**2

    def div_rsq(self, x_rsq):
        return x_rsq / self.r_vals**2

    def times_rsq_const(self, x):
        x_xrsq =  x * self.r_vals**2
        return (x_xrsq + 300)/300

    def div_rsq_const(self, x_rsq):
        x = x_rsq / self.r_vals**2
        return x*300 - 300

    def times_rsq_mean(self, x):
        x_xrsq = x * self.r_vals**2
        x_xrsq_mean = (x_xrsq - self.y_train_xrsq_mean)/self.y_train_xrsq_std
        return x_xrsq_mean

    def div_rsq_mean(self, x_xrsq_mean):
        x_xrsq = x_xrsq_mean * self.y_train_xrsq_std + self.y_train_xrsq_mean
        x = x_xrsq / self.r_vals**2
        return x

    def times_rsq_mean_const(self, x):
        x_xrsq = x * self.r_vals**2
        x_xrsq_mean = (x_xrsq - self.y_train_xrsq_mean)/self.y_train_xrsq_std
        x_xrsq_mean_const = x_xrsq_mean + 4 #magic, to ensure no negatives
        return x_xrsq_mean_const

    def div_rsq_mean_const(self, x_xrsq_mean_const):
        x_xrsq_mean = x_xrsq_mean_const - 4 #magic
        x_xrsq = x_xrsq_mean * self.y_train_xrsq_std + self.y_train_xrsq_mean
        x = x_xrsq / self.r_vals**2
        return x

    def times_rsq_minmax(self, x):
        x_xrsq = x * self.r_vals**2
        x_xrsq_minmax = (x_xrsq - self.y_train_xrsq_min)/(self.y_train_xrsq_max - self.y_train_xrsq_min)
        return x_xrsq_minmax

    def div_rsq_minmax(self, x_xrsq_minmax):
        x_xrsq = x_xrsq_minmax*(self.y_train_xrsq_max - self.y_train_xrsq_min) + self.y_train_xrsq_min
        x = x_xrsq / self.r_vals**2
        return x

    def construct_scalers(self, scaling):
        
        self.scaler_x = StandardScaler(with_mean=False, with_std=False) # NO XSCALE 
        self.scaler_x.fit(self.x_train)

        if scaling=='log':
            self.scaler_y = FunctionTransformer(func=np.log10, inverse_func=self.pow10) 
            self.scaler_y.fit(self.y_train)  

        elif scaling=='mean':
            self.scaler_y = StandardScaler() 
            self.scaler_y.fit(self.y_train)  

        elif scaling=='xrsq':
            self.scaler_y = FunctionTransformer(func=self.times_rsq, inverse_func=self.div_rsq) 
            self.scaler_y.fit(self.y_train)  

        elif scaling=='xrsqconst':
            self.scaler_y = FunctionTransformer(func=self.times_rsq_const, inverse_func=self.div_rsq_const)
            self.scaler_y.fit(self.y_train)

        elif scaling=='xrsqmean':
            self.y_train_xrsq_mean = np.mean(self.times_rsq(self.y_train), axis=0)
            self.y_train_xrsq_std = np.std(self.times_rsq(self.y_train), axis=0)
            self.scaler_y = FunctionTransformer(func=self.times_rsq_mean, inverse_func=self.div_rsq_mean) 
            self.scaler_y.fit(self.y_train)  

        elif scaling=='xrsqmeanconst':
            self.y_train_xrsq_mean = np.mean(self.times_rsq(self.y_train), axis=0)
            self.y_train_xrsq_std = np.std(self.times_rsq(self.y_train), axis=0)
            self.scaler_y = FunctionTransformer(func=self.times_rsq_mean_const, inverse_func=self.div_rsq_mean_const) 
            self.scaler_y.fit(self.y_train)  

        elif scaling=='xrsqminmax':
            self.y_train_xrsq_min = np.min(self.times_rsq(self.y_train), axis=0)
            self.y_train_xrsq_max = np.max(self.times_rsq(self.y_train), axis=0)
            self.scaler_y = FunctionTransformer(func=self.times_rsq_minmax, inverse_func=self.div_rsq_minmax)
            self.scaler_y.fit(self.y_train)

        else:
            raise ValueError(f"Scaling method {scaling} not recognized! Choose from: ['log', 'mean', 'xrsq', 'xrsqconst', 'xrsqmean', 'xrsqminmax']")


    def scale_y_error(self, scaling):
        
        if scaling=='log':
            #for log of y, errors are 1/ln(10) * dy/y. dy is error, for y we use the mean.
            #source: https://faculty.washington.edu/stuve/log_error.pdf, https://web.ma.utexas.edu/users/m408n/m408c/CurrentWeb/LM3-6-2.php
            self.y_error_scaled = 1/np.log(10) * self.y_error / np.mean(self.y_train, axis=0) #logscaler

        elif scaling=='mean':
            self.scaler_y_err = StandardScaler(with_mean=False) #mean false bc don't want to shift it, just rescale by std of training
            #the args to fit are the mean and std used; we want to scale by the same std as we scaled the y_train data by
            self.scaler_y_err.fit(self.y_train)
            self.y_error_scaled = self.scaler_y_err.transform(self.y_error.reshape(1, -1))
            self.y_error_scaled = self.y_error_scaled.flatten()     
            # ends up same as: self.y_error/np.std(self.y_train, axis=0)

        elif scaling=='xrsq':
            self.y_error_scaled = self.y_error * self.r_vals**2

        elif scaling=='xrsqconst':
            self.y_error_scaled = self.y_error * self.r_vals**2
            self.y_error_scaled /= 300 #magic - chosen constant

        elif scaling=='xrsqmean' or scaling=='xrsqmeanconst':
            self.y_train_xrsq_std = np.std(self.times_rsq(self.y_train), axis=0)
            self.y_error_scaled = self.y_error * self.r_vals**2 # xsqr part
            self.y_error_scaled /= self.y_train_xrsq_std # mean part; only std bc don't want to shift it, just rescale by std of training
            #print("ARTIFICIAL ERRx1.5")
            #self.y_error_scaled *= 1.5

        elif scaling=='xrsqminmax':
            self.y_train_xrsq_min = np.min(self.times_rsq(self.y_train), axis=0)
            self.y_train_xrsq_max = np.max(self.times_rsq(self.y_train), axis=0)
            self.y_error_scaled = self.y_error * self.r_vals**2 # xsqr part
            self.y_error_scaled /= (self.y_train_xrsq_max - self.y_train_xrsq_min) # only do the multiplication part bc don't want to shift it, just rescale

        else:
            raise ValueError(f"Scaling method {scaling} not recognized! Choose from: ['log', 'mean', 'xrsq', 'xrsqconst', 'xrsqmean', 'xrsqminmax']")

    def scale_training_data(self):
        self.x_train_scaled = self.scaler_x.transform(self.x_train)  
        self.y_train_scaled = self.scaler_y.transform(self.y_train)  

    def scale_testing_data(self):
        self.x_test_scaled = self.scaler_x.transform(self.x_test)  
        self.y_test_scaled = self.scaler_y.transform(self.y_test)  

    def save_scalers(self):
        joblib.dump(self.scaler_x, self.scaler_x_fn)
        joblib.dump(self.scaler_y, self.scaler_y_fn)

    def load_scalers(self):
        self.scaler_x = joblib.load(self.scaler_x_fn)
        self.scaler_y = joblib.load(self.scaler_y_fn)

    #@profile
    def predict(self, x_to_predict):
        # make sure parameters in correct order
        if type(x_to_predict)==dict:
            x_to_predict_arr = []
            for pn in self.param_names_ordered:
                x_to_predict_arr.append(x_to_predict[pn])
        elif type(x_to_predict)==list or type(x_to_predict)==np.ndarray:
            x_to_predict_arr = x_to_predict
        else:
            raise ValueError("Params to predict at must be dict or array")

        # scale x
        x_to_predict_arr = np.atleast_2d(x_to_predict_arr)
        x_to_predict_scaled = self.scaler_x.transform(x_to_predict_arr)

        # every emu should implement its own predict_scaled
        y_predict_scaled = self.predict_scaled(x_to_predict_scaled)

        # transform back
        # pad y_predict_scaled before transforming bc scaler expects 9-vector
        y_predict_scaled_padded = np.zeros(self.n_bins_tot)
        y_predict_scaled_padded[self.bins] = y_predict_scaled
        y_predict_padded = self.scaler_y.inverse_transform(y_predict_scaled_padded)
        y_predict = y_predict_padded[self.bins]
        return y_predict

    def save_predictions(self, predictions_dir):
        os.makedirs(f'{predictions_dir}', exist_ok=True)
        
        for i in range(self.n_test):
            id_cosmo, id_hod = self.id_pairs_test[i]
            y_pred_fn = f'{predictions_dir}/{self.statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
            y_pred = self.y_predict[i]
            results = np.array([self.r_vals, y_pred])
            np.savetxt(y_pred_fn, results.T, delimiter=',', fmt=['%f', '%e'])

    def train(self):
        pass

    def test(self):
        pass

    def predict_scaled(self, x_to_predict_scaled):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass



class EmulatorMLP(Emulator):

    # def __init__(self, statistic):
    #     super().__init__(statistic)

    def train(self, max_iter=10000):
        self.model = MLPRegressor(max_iter=max_iter).fit(self.x_train_scaled, self.y_train_scaled)

    def test(self):
        self.y_predict_scaled = self.model.predict(self.x_test_scaled)
        self.y_predict = self.scaler_y.inverse_transform(self.y_predict_scaled)
 
    def save_model(self):
        joblib.dump(self.model, f'{self.model_fn}.joblib')

    def load_model(self):
        self.model = joblib.load(f'{self.model_fn}.joblib')


class EmulatorGPFlow(Emulator):

    def train(self, max_iter=1000):

        # print(self.x_train_scaled.shape)
        # self.x_train_scaled = self.x_train_scaled[:500,:]
        # self.y_train_scaled = self.y_train_scaled[:500,:]
        # self.n_train = 500
        # print(self.x_train_scaled.shape)

        lengthscales = np.full(self.n_params, 1.0) # defines amount of wiggliness
        k_expsq = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=lengthscales)
        k_m32 = gpflow.kernels.Matern32(variance=0.1, lengthscales=lengthscales)
        k_const = gpflow.kernels.Constant(0.1)
        # k_expsq = gpflow.kernels.SquaredExponential(variance=0.1)#, lengthscales=lengthscales)
        # k_m32 = gpflow.kernels.Matern32(variance=0.1)#, lengthscales=lengthscales)
        # k_const = gpflow.kernels.Constant(0.1)
        kernel = k_expsq*k_const + k_m32
        print_summary(kernel)

        data = (self.x_train_scaled, self.y_train_scaled)
        #mean_function = None
        mean_function = gpflow.mean_functions.Constant(np.mean(self.y_train_scaled, axis=0))
        self.model = gpflow.models.GPR(data=data, kernel=kernel, mean_function=mean_function)
        print_summary(self.model)
        print("Optimizing")
        opt = gpflow.optimizers.Scipy()
        #print(self.model.trainable_variables)
        #print()
        # there has GOT to be a better way to do this
        n_vars = np.sum([len(tv.numpy()) if type(tv.numpy()) is np.ndarray else 1 for tv in self.model.trainable_variables])
        bounds = [np.log((1e-6, 1e+6)) for i in range(n_vars)]
        opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables, 
                                method='L-BFGS-B', bounds=bounds,
                                options=dict(maxiter=max_iter))
        #print_summary(self.model)
        #print(opt_logs)

    def test(self):
        # predict_f outputs mean and variance.
        # TODO: check if should use predict_y, which includes noise
        self.y_predict_scaled, variance = self.model.predict_f_compiled(self.x_test_scaled)
        self.y_predict = self.scaler_y.inverse_transform(self.y_predict_scaled)
 
    def save_model(self):
        self.model.predict_f_compiled = tf.function(self.model.predict_f, 
                                   input_signature=[tf.TensorSpec(shape=[None, self.n_params], dtype=tf.float64)])
        tf.saved_model.save(self.model, self.model_fn)

    def load_model(self):
        self.model = tf.saved_model.load(self.model_fn)


class EmulatorGPFlowVGP(Emulator):

    def train(self, max_iter=1000):

        #(self.x_train_scaled.shape)
        self.x_train_scaled = self.x_train_scaled[:50,:]
        self.y_train_scaled = self.y_train_scaled[:50,:]
        self.n_train = 50
        #print(self.x_train_scaled.shape)

        lengthscales = np.full(self.n_params, 1.0) # defines amount of wiggliness
        k_expsq = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=lengthscales)
        k_m32 = gpflow.kernels.Matern32(variance=0.1, lengthscales=lengthscales)
        k_const = gpflow.kernels.Constant(0.1)
        # k_expsq = gpflow.kernels.SquaredExponential(variance=0.1)#, lengthscales=lengthscales)
        # k_m32 = gpflow.kernels.Matern32(variance=0.1)#, lengthscales=lengthscales)
        # k_const = gpflow.kernels.Constant(0.1)
        kernel = k_expsq*k_const + k_m32
        #print_summary(kernel)

        likelihood = HeteroskedasticGaussian()

        #Y_data = np.hstack([Y, NoiseVar])
        print("Data shapes")
        #print(self.y_train_scaled.shape, self.y_error_scaled.shape)
        y_error_matched = np.empty(self.y_train_scaled.shape)
        for i in range(self.n_train):
            y_error_matched[i,:] = self.y_error_scaled
        #print(y_error_matched.shape)
        #print(y_error_matched)
        y_data = np.vstack([self.y_train_scaled, self.y_error_scaled])
        #y_data = np.hstack([self.y_train_scaled, y_error_matched])
        #y_data = np.array([self.y_train_scaled, y_error_matched]).reshape(self.n_train, self.n_bins_tot, 2)
        #print(y_data.shape)
        #y_data = [self.y_train_scaled, self.y_error_scaled]
        #print(y_data.shape)
        data = (self.x_train_scaled, y_data)
        #data = (self.x_train_scaled, self.y_train_scaled)
        #mean_function = None
        #mean_function = gpflow.mean_functions.Constant(np.mean(self.y_train_scaled, axis=0))
        #self.model = gpflow.models.GPR(data=data, kernel=kernel, mean_function=mean_function)
        self.model = gpflow.models.VGP(data, kernel=kernel, likelihood=likelihood, num_latent_gps=1)
        #print_summary(self.model)

        print("Optimizing")
        natgrad = NaturalGradient(gamma=1.0)
        adam = tf.optimizers.Adam()

        set_trainable(self.model.q_mu, False)
        set_trainable(self.model.q_sqrt, False)

        for _ in range(ci_niter(max_iter)):
            natgrad.minimize(self.model.training_loss, [(self.model.q_mu, self.model.q_sqrt)])
            adam.minimize(self.model.training_loss, self.model.trainable_variables)
        # opt = gpflow.optimizers.Scipy()
        # print(self.model.trainable_variables)
        # #print()
        # # there has GOT to be a better way to do this
        # n_vars = np.sum([len(tv.numpy()) if type(tv.numpy()) is np.ndarray else 1 for tv in self.model.trainable_variables])
        # bounds = [np.log((1e-6, 1e+6)) for i in range(n_vars)]
        # opt_logs = opt.minimize(self.model.training_loss, self.model.trainable_variables, 
        #                         method='L-BFGS-B', bounds=bounds,
        #                         options=dict(maxiter=max_iter))
        print_summary(self.model)
        print(opt_logs)

    def test(self):
        # predict_f outputs mean and variance.
        # TODO: check if should use predict_y, which includes noise
        self.y_predict_scaled, variance = self.model.predict_f_compiled(self.x_test_scaled)
        self.y_predict = self.scaler_y.inverse_transform(self.y_predict_scaled)
 
    def save_model(self):
        self.model.predict_f_compiled = tf.function(self.model.predict_f, 
                                   input_signature=[tf.TensorSpec(shape=[None, self.n_params], dtype=tf.float64)])
        tf.saved_model.save(self.model, self.model_fn)

    def load_model(self):
        self.model = tf.saved_model.load(self.model_fn)


class EmulatorGPFlowBinned(Emulator):

    def train(self, max_iter=1000):

        #FOR TESTING, CAREFUL
        # print(self.x_train_scaled.shape)
        # self.x_train_scaled = self.x_train_scaled[:50,:]
        # self.y_train_scaled = self.y_train_scaled[:50,:]
        # self.n_train = 50
        # print(self.x_train_scaled.shape)

        self.models = np.empty((self.n_bins_tot), dtype=object)
        for n in range(self.n_bins_tot):
            print(f"Training bin {n}")
            lengthscales = np.full(self.n_params, 1.0) # 1.0 is wiggliness
            k_expsq = gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=lengthscales)
            k_m32 = gpflow.kernels.Matern32(variance=0.1, lengthscales=lengthscales)
            k_const = gpflow.kernels.Constant(0.1)
            kernel = k_expsq*k_const + k_m32
            print_summary(kernel)

            y_data = self.y_train_scaled[:,n].reshape(-1, 1)
            data = (self.x_train_scaled, y_data)
            mean_function = None
            #mean_function = gpflow.mean_functions.Constant(np.mean(self.y_train_scaled[:,n]))
            self.models[n] = gpflow.models.GPR(data=data, kernel=kernel, mean_function=mean_function,
                                               noise_variance=self.y_error_scaled[n])
            print_summary(self.models[n])
            print("Optimizing")
            opt = gpflow.optimizers.Scipy()
            n_vars = np.sum([len(tv.numpy()) if type(tv.numpy()) is np.ndarray else 1 for tv in self.models[n].trainable_variables])
            bounds = [np.log((1e-6, 1e+6)) for i in range(n_vars)]
            opt_logs = opt.minimize(self.models[n].training_loss, self.models[n].trainable_variables, 
                                    method='L-BFGS-B', bounds=bounds,
                                    options=dict(maxiter=max_iter))
            print_summary(self.models[n])
            print(opt_logs)

    def test(self):
        self.y_predict = np.empty(self.y_test.shape)
        self.y_predict_scaled = np.empty(self.y_test.shape)
        for n in range(self.n_bins_tot):
            y_pred, variance = self.models[n].predict_f_compiled(self.x_test_scaled)
            self.y_predict_scaled[:,n] = y_pred.numpy().flatten()
        self.y_predict = self.scaler_y.inverse_transform(self.y_predict_scaled)
 
    def save_model(self):
        os.makedirs(self.model_fn, exist_ok=True)
        for n in range(self.n_bins_tot):
            self.models[n].predict_f_compiled = tf.function(self.models[n].predict_f, 
                                    input_signature=[tf.TensorSpec(shape=[None, self.n_params], dtype=tf.float64)])
            model_bin_fn = f'{self.model_fn}/model_bin{n}'
            tf.saved_model.save(self.models[n], model_bin_fn)

    def load_model(self):
        self.models = np.empty((self.n_bins_tot), dtype=object)
        for n in range(self.n_bins_tot):
            model_bin_fn = f'{self.model_fn}/model_bin{n}'
            self.models[n] = tf.saved_model.load(model_bin_fn)


class EmulatorGeorge(Emulator):

    def train(self, max_iter=1000):

        print("george version:", george.__version__)
        # don't need maxiter here, but want function handle to be consistent, just for ease of calling

        self.models = np.empty((self.n_bins), dtype=object)
        n_threads = self.n_bins
        pool = mp.Pool(processes=n_threads)

        models = pool.starmap(self.train_bin, zip(self.y_train_scaled.T, self.y_error_scaled))
        for i, n in enumerate(self.bins):
            self.models[i] = models[i]


    def train_bin(self, y_train_scaled_bin, y_error_scaled_bin):
        print(f"Training bin")
        p0 = np.exp(np.full(self.n_params, 0.1))
        k_expsq = george.kernels.ExpSquaredKernel(p0, ndim=self.n_params)
        k_m32 = george.kernels.Matern32Kernel(p0, ndim=self.n_params)
        k_const = george.kernels.ConstantKernel(0.1, ndim=self.n_params)    
                
        k_const2 = george.kernels.ConstantKernel(0.1, ndim=self.n_params)        
    
        kernel = k_expsq*k_const + k_m32 # this is "M32ExpConst"
        #kernel = k_expsq*k_const + k_m32 + k_const2 # this is "M32ExpConst2"
        #print("USING M32ExpConst2 KERNEL, MIGHT WANNA CHANGE BACK to M32ExpConst ??")

        model = george.GP(kernel, mean=np.mean(y_train_scaled_bin), solver=george.BasicSolver)
        model.compute(self.x_train_scaled, y_error_scaled_bin)
        print(y_train_scaled_bin)

        def neg_ln_like(p):
            model.set_parameter_vector(p)
            logl = model.log_likelihood(y_train_scaled_bin, quiet=True)
            return -logl if np.isfinite(logl) else 1e25

        print("Optimizing")
        # note: the parameter vector is one value for each kernel parameter in the combined kernel! 37d for us
        bounds = [np.log((1e-6, 1e+6)) for i in range(len(model.get_parameter_vector()))]
        result = scipy.optimize.minimize(neg_ln_like, model.get_parameter_vector(), method='L-BFGS-B', bounds=bounds)
        model.set_parameter_vector(result.x)

        return model

    def predict_scaled(self, x_to_predict_scaled):
        y_predict_scaled = np.empty(self.n_bins)
        for i, n in enumerate(self.bins):
            y_predict_scaled[i] = self.models[i].predict(self.y_train_scaled[:,n], x_to_predict_scaled, return_cov=False)
        return y_predict_scaled

    def test(self):
        # need padded versions so that scaler works properly
        self.y_predict_scaled_padded = np.empty((self.n_test, self.n_bins))
        for i, n in enumerate(self.bins):
            # y_predict_scaled_padded index is n because padded; models index is i because only bins
            print(np.sum(np.isfinite(self.y_train_scaled[:,n]))/len(self.y_train_scaled[:,n]))
            print(len(self.y_train_scaled[:,n]))
            print(np.where(np.isfinite(self.y_train_scaled[:,n])==False)[0])
            print(self.y_train_scaled[:,n])
            print(self.x_test_scaled)
            self.y_predict_scaled_padded[:,n] = self.models[i].predict(
                                    self.y_train_scaled[:,n], self.x_test_scaled, return_cov=False)
        self.y_predict_padded = self.scaler_y.inverse_transform(self.y_predict_scaled_padded)
        self.y_predict_scaled = self.y_predict_scaled_padded[:,self.bins]
        self.y_predict = self.y_predict_padded[:,self.bins]


    def save_model(self):
        ### pickle method
        os.makedirs(self.model_fn, exist_ok=True)
        for i, n in enumerate(self.bins):
            model_bin_fn = f'{self.model_fn}/model_bin{n}.pkl'
            with open(model_bin_fn, "wb") as fp:
                pickle.dump(self.models[i], fp)

    def load_model(self):
        ### pickle method
        self.models = [None]*self.n_bins
        for i, n in enumerate(self.bins):
            model_bin_fn = f'{self.model_fn}/model_bin{n}.pkl'
            with open(model_bin_fn, "rb") as fp:
                self.models[i] = pickle.load(fp)


class EmulatorGeorgeOrig(Emulator):

    def train(self, max_iter=1000):

        print("george version:", george.__version__)
        self.models = [None]*self.n_bins_tot
        print("Training commences!")
        print("Constructing pool")
        pool = mp.Pool(processes=self.n_bins_tot)
        print("Mapping bins")
        res = pool.map(self.train_bin, range(self.n_bins_tot))
        print("Done training!")
        print(np.array(res).shape)
        # 37 is len kernel
        self.hyperparams = np.empty((self.n_bins_tot, 37))
        for bb in range(self.n_bins_tot):
            #self.hyperparams[bb, :] = res[bb]
            self.models[bb] = res[bb]


    def train_bin(self, n):
        print(f"Training bin {n}")

        y_train_scaled_bin = self.y_train_scaled[:,n]    
        y_error_scaled_bin = self.y_error_scaled[n]

        if n==0:
            print('xtrain:', self.x_train_scaled)
            print('ytrain:', y_train_scaled_bin)
            print('yerr:', y_error_scaled_bin)

        p0 = np.full(self.n_params, 0.1)
        p0 = np.exp(p0) 
        k1 = george.kernels.ExpSquaredKernel(p0, ndim=len(p0))
        k2 = george.kernels.Matern32Kernel(p0, ndim=len(p0))
        k5 = george.kernels.ConstantKernel(0.1, ndim=len(p0))
        kernel = k1*k5 + k2

        mean = np.mean(y_train_scaled_bin)
        if n==0:
            print("mean:", mean)
        self.models[n] = george.GP(kernel, mean=mean, solver=george.BasicSolver)
        #gp.compute(self.training_params, self.gperr[bb])
        self.models[n].compute(self.x_train_scaled, y_error_scaled_bin)

        #p0 = gp.kernel.get_parameter_vector()
        p0 = self.models[n].get_parameter_vector()
        def nll(p):
            self.models[n].set_parameter_vector(p)
            ll = self.models[n].log_likelihood(y_train_scaled_bin, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        bnd = [np.log((1e-6, 1e+6)) for i in range(len(p0))]
        results = scipy.optimize.minimize(nll, p0, method='L-BFGS-B', bounds=bnd)
        self.models[n].set_parameter_vector(results.x)
        hyps = self.models[n].kernel.get_parameter_vector()
        if n==0:
            print("hyps:", hyps)
        #return hyps
        return self.models[n]

    def test(self):
        self.y_predict = np.empty(self.y_test.shape)
        self.y_predict_scaled = np.empty(self.y_test.shape)
        for n in range(self.n_bins_tot):
            self.y_predict_scaled[:,n] = self.models[n].predict(
                                    self.y_train_scaled[:,n], self.x_test_scaled, return_cov=False)
        self.y_predict = self.scaler_y.inverse_transform(self.y_predict_scaled)
        print("params (0,0):", self.x_test_scaled[0])
        print("vals_pred (0,0):", self.y_predict[0])
        print('hyps2:', self.models[0].get_parameter_vector())

    def save_model(self):
        os.makedirs(self.model_fn, exist_ok=True)
        for n in range(self.n_bins_tot):
            model_bin_fn = f'{self.model_fn}/model_bin{n}.pkl'
            with open(model_bin_fn, "wb") as fp:
                pickle.dump(self.models[n], fp)

    def load_model(self):
        for n in range(self.n_bins_tot):
            model_bin_fn = f'{self.model_fn}/model_bin{n}.pkl'
            with open(model_bin_fn, "rb") as fp:
                self.models[n] = pickle.load(fp)

    # def test(self):
    #     self.y_predict = np.empty(self.y_test.shape)
    #     self.y_predict_scaled = np.empty(self.y_test.shape)
    #     count = 0
    #     for pid, tparams in self.x_test_scaled.items():
    #         vals_pred = self.predict(tparams)
    #         self.y_predict[count,:] = vals_pred
    #         count += 1

    # def predict(self, params_pred):
    #     #print(params_pred)
    #     if type(params_pred)==dict:
    #         params_arr = []
    #         param_names_ordered = ['Omega_m', 'Omega_b', 'sigma_8', 'h', 'n_s', 'N_eff', 'w',
    #                                 'M_sat', 'alpha', 'M_cut', 'sigma_logM', 'v_bc', 'v_bs', 'c_vir', 'f',
    #                                'f_env', 'delta_env', 'sigma_env']
    #         for pn in param_names_ordered:
    #             params_arr.append(params_pred[pn])
    #     elif type(params_pred)==list or type(params_pred)==np.ndarray:
    #         params_arr = params_pred
    #     else:
    #         raise ValueError("Params to predict at must be dict or array")

    #     params_arr = np.atleast_2d(params_arr)
    #     y_pred = np.zeros(self.n_bins_tot)
    #     for n in range(self.n_bins_tot):
    #         # predict on all the training data in the bin
    #         val_pred, cov_pred = self.models[bb].predict(self.y_train_scaled[:,n], params_arr)
    #         val_pred = self.scaler_y.inverse_transform(val_pred)
    #         y_pred[bb] = val_pred

    #     return y_pred

 
    # def save_model(self):
    #     np.savetxt(f'{self.model_fn}.txt', self.hyperparams, fmt='%.7f')
    #     print(f"Saved hyperparameters to {self.model_fn}.txt")

        
    # def load_model(self):
    #     #we still need the training data to condition on;
    #     #load it back up even in test mode
    #     self.set_training_data()
    #     self.scale_training_data()
    #     self.hyperparams = np.loadtxt(f'{self.model_fn}.txt')
    #     for n in range(self.n_bins_tot):
    #         p0 = np.exp(np.full(self.n_params, 0.1))
    #         k_expsq = george.kernels.ExpSquaredKernel(p0, ndim=self.n_params)
    #         k_m32 = george.kernels.Matern32Kernel(p0, ndim=self.n_params)
    #         k_const = george.kernels.ConstantKernel(0.1, ndim=self.n_params)
    #         # this is "M32ExpConst"
    #         kernel = k_expsq*k_const + k_m32

    #         y_train_scaled_bin = self.y_train_scaled[:,n]
    #         self.models[n] = george.GP(kernel, mean=np.mean(y_train_scaled_bin), solver=george.BasicSolver)
    #         self.models[n].compute(self.x_train_scaled, self.y_error_scaled[n])
    #         self.models[n].set_parameter_vector(self.hyperparams[n])
    #         self.models[n].compute(self.x_train_scaled, self.y_error_scaled[n])


class EmulatorPyTorch(Emulator):

    # class NeuralNetwork(nn.Module):
    #     def __init__(self, n_input, n_output):
    #         super().__init__()
    #         self.flatten = nn.Flatten()
    #         self.layer_1 = nn.Linear(n_input, 36)
    #         self.layer_2 = nn.Linear(36, 18)
    #         self.layer_3 = nn.Linear(18, 18)
    #         self.layer_out = nn.Linear(18, n_output)
    #         self.relu = nn.ReLU()

    #     def forward(self, x):
    #         x = self.relu(self.layer_1(x))
    #         x = self.relu(self.layer_2(x))
    #         x = self.relu(self.layer_3(x))
    #         x = self.layer_out(x)
    #         return x

    # class Dataset(torch.utils.data.Dataset):
    #     'Characterizes a dataset for PyTorch'
    #     def __init__(self, xs, ys):
    #         'Initialization'
    #         self.xs = xs
    #         self.ys = ys

    #     def __len__(self):
    #         'Denotes the total number of samples'
    #         return len(self.xs)

    #     def __getitem__(self, index):
    #         'Generates one sample of data'
    #         # Select sample
    #         x = self.xs[index]
    #         y = self.ys[index]
    #         return x, y

    def train(self, max_iter=1000):

        # Construct model
        self.model = self.NeuralNetwork(self.n_params, self.n_bins_tot)

        # Set up data
        x_train = Variable(torch.from_numpy(self.x_train_scaled).float())
        y_train = Variable(torch.from_numpy(self.y_train_scaled).float())
        training_set = self.Dataset(x_train, y_train)
        dataloader_params = {'batch_size': 32,
                            'shuffle': True,
                            'num_workers': self.n_bins_tot}
        training_generator = torch.utils.data.DataLoader(training_set, **dataloader_params)

        # Set up optimization
        learning_rate = 1e-2
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        for epoch in range(max_iter):

            for x_batch, y_batch in training_generator:
                #forward feed
                y_pred = self.model(x_batch)

                #calculate the loss
                loss = loss_function(y_pred, y_batch)

                #backward propagation: calculate gradients
                loss.backward()

                #update the weights
                optimizer.step()

                #clear out the gradients from the last step loss.backward()
                optimizer.zero_grad()
                
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    def test(self):
        x_test = Variable(torch.from_numpy(self.x_test_scaled).float())
        self.y_predict_scaled = self.model(x_test).data.numpy()
        self.y_predict = self.scaler_y.inverse_transform(self.y_predict_scaled)

    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.model_fn}.pt')

    def load_model(self):  
        self.model = self.NeuralNetwork(self.n_params, self.n_bins_tot)
        self.model.load_state_dict(torch.load(f'{self.model_fn}.pt'))
        self.model.eval()

# class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
#     def __init__(self, **kwargs):
#         # this likelihood expects a single latent function F, and two columns in the data matrix Y:
#         super().__init__(latent_dim=1, observation_dim=2, **kwargs)

#     def _log_prob(self, F, Y):
#         # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
#         # Because variational_expectations is implemented analytically below, this is not actually needed,
#         # but is included for pedagogical purposes.
#         # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
#         print('log prob')
#         print(Y.shape)
#         print(Y[:, 0].shape, Y[:, 1].shape)
#         Y, NoiseVar = Y[:, 0], Y[:, 1]
#         #Y, NoiseVar = Y[0], Y[1]
#         return gpflow.logdensities.gaussian(Y, F, NoiseVar)

#     def _variational_expectations(self, Fmu, Fvar, Y):
#         Y, NoiseVar = Y[:, 0], Y[:, 1]
#         Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
#         return (
#             -0.5 * np.log(2 * np.pi)
#             - 0.5 * tf.math.log(NoiseVar)
#             - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
#         )

#     # The following two methods are abstract in the base class.
#     # They need to be implemented even if not used.

#     def _predict_log_density(self, Fmu, Fvar, Y):
#         raise NotImplementedError

#     def _predict_mean_and_var(self, Fmu, Fvar):
#         raise NotImplementedError
