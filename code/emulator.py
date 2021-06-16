import os
import numpy as np

import joblib
from sklearn.neural_network import MLPRegressor

class Emulator(object):

    def __init__(self, statistic):
        self.statistic = statistic
        self.set_training_data()


    def set_training_data(self):
        
        ### ID values (cosmo and hod numbers)

        self.id_pairs_train = []
        ids_cosmo = range(40)
        n_hods_per_cosmo = 100
        for id_cosmo in ids_cosmo:
            id_hod_min = id_cosmo * n_hods_per_cosmo
            id_hod_max = id_hod_min + n_hods_per_cosmo
            ids_hod = range(id_hod_min, id_hod_max)
            for id_hod in ids_hod:
                self.id_pairs_train.append((id_cosmo, id_hod))
        self.n_train = len(self.id_pairs_train)

        ### x values (data, cosmo and hod values)

        cosmos_train_fn = '../tables/cosmology_camb_full.dat'
        cosmos_train = np.loadtxt(cosmos_train_fn)
        n_cosmo_params = cosmos_train.shape[1]

        hods_train_fn = '../tables/HOD_design_np11_n5000_new_f_env.dat'
        hods_train = np.loadtxt(hods_train_fn)
        # Convert these columns (0: M_sat, 2: M_cut) to log to reduce range
        hods_train[:, 0] = np.log10(hods_train[:, 0])
        hods_train[:, 2] = np.log10(hods_train[:, 2])
        n_hod_params = hods_train.shape[1]

        n_params = n_cosmo_params + n_hod_params

        self.x_train = np.empty((self.n_train, n_params))
        for i in range(self.n_train):
            id_cosmo, id_hod = self.id_pairs_train[i]
            self.x_train[i,:] = np.concatenate((cosmos_train[id_cosmo], hods_train[id_hod]))

        ### y values (labels, value of statistics in each bin)

        n_bins = 9
        self.y_train = np.empty((self.n_train, n_bins))
        y_train_dir = '/home/users/ksf293/clust/results_aemulus_train'
        for i in range(self.n_train):
            id_cosmo, id_hod = self.id_pairs_train[i]
            y_train_fn = f'{y_train_dir}/results_{self.statistic}/{self.statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_test_0.dat'
            r_vals, y = np.loadtxt(y_train_fn, delimiter=',', unpack=True)
            self.y_train[i,:] = y
        # all r_vals are the same so just save the last one
        self.r_vals = r_vals


    def set_testing_data(self):

        ### ID values (cosmo and hod numbers)

        self.id_pairs_test = []
        ids_cosmo = range(7)
        ids_hod = range(100)
        for id_cosmo in ids_cosmo:
            for id_hod in ids_hod:
                self.id_pairs_test.append((id_cosmo, id_hod))
        self.n_test = len(self.id_pairs_test)

        ### x values (data, cosmo and hod values)

        cosmos_test_fn = '../tables/cosmology_camb_test_box_full.dat'
        cosmos_test = np.loadtxt(cosmos_test_fn)
        n_cosmo_params = cosmos_test.shape[1]

        hods_test_fn = '../tables/HOD_test_np11_n1000_new_f_env.dat'
        hods_test = np.loadtxt(hods_test_fn)
        # Convert these columns (0: M_sat, 2: M_cut) to log to reduce range
        hods_test[:, 0] = np.log10(hods_test[:, 0])
        hods_test[:, 2] = np.log10(hods_test[:, 2])
        n_hod_params = hods_test.shape[1]

        n_params = n_cosmo_params + n_hod_params

        self.x_test = np.empty((self.n_test, n_params))
        for i in range(self.n_test):
            id_cosmo, id_hod = self.id_pairs_test[i]
            self.x_test[i,:] = np.concatenate((cosmos_test[id_cosmo], hods_test[id_hod]))

        ### y values (labels, value of statistics in each bin)
        # Note: here we are using the mean of 5 boxes with the same parameters

        n_bins = 9
        self.y_test = np.empty((self.n_test, n_bins))
        y_test_dir = '/home/users/ksf293/clust/results_aemulus_test_mean'
        for i in range(self.n_test):
            id_cosmo, id_hod = self.id_pairs_test[i]
            y_test_fn = f'{y_test_dir}/results_{self.statistic}/{self.statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_mean.dat'
            _, y = np.loadtxt(y_test_fn, delimiter=',', unpack=True)
            self.y_test[i,:] = y


    def save_predictions(self, predictions_dir):
        os.makedirs(f'{predictions_dir}/results_{self.statistic}', exist_ok=True)
        
        for i in range(self.n_test):
            id_cosmo, id_hod = self.id_pairs_test[i]
            y_pred_fn = f'{predictions_dir}/results_{self.statistic}/{self.statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
            y_pred = self.y_predict[i]
            results = np.array([self.r_vals, y_pred])
            np.savetxt(y_pred_fn, results.T, delimiter=',', fmt=['%f', '%e'])


    def train(self):
        pass


    def test(self):
        pass



class EmulatorMLP(Emulator):

    def __init__(self, statistic):
        super().__init__(statistic)

    def train(self, max_iter=10000):
        self.model = MLPRegressor(max_iter=max_iter).fit(self.x_train, self.y_train)

    def test(self):
        self.y_predict = self.model.predict(self.x_test)
 
    def save_model(self, model_fn):
        joblib.dump(self.model, model_fn)

    def load_model(self, model_fn):
        self.model = joblib.load(model_fn)
            