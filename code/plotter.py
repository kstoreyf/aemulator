import numpy as np
from matplotlib import pyplot as plt

from utils import *


def plot_accuracy(statistic, train_tag):
    nrows = 3
    fig, axarr = plt.subplots(nrows, 1, figsize=(10,15), gridspec_kw={'height_ratios': [1]*nrows})
    plt.subplots_adjust(hspace=0.15)

    fn_test = '../tables/id_pairs_test.txt'
    id_pairs_test = np.loadtxt(fn_test, delimiter=',', dtype='int')
    n_test = id_pairs_test.shape[0]

    ids_cosmo_unique = set(id_pairs_test[:,0])
    color_idx = np.linspace(0, 1, len(ids_cosmo_unique))

    n_bins = 9
    y_test_dir = '/home/users/ksf293/clust/results_aemulus_test_mean'
    predictions_dir = f'../predictions_{statistic}{train_tag}'
    ys_test = np.empty((n_test, n_bins))
    ys_pred = np.empty((n_test, n_bins))

    alpha = 0.5
    colors = np.empty((n_test, 4)) # 4 for RGBA
    zorders = np.arange(n_test)
    np.random.shuffle(zorders)
    for i in range(n_test):
        id_cosmo, id_hod = id_pairs_test[i]
        y_test_fn = f'{y_test_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_mean.dat'
        r_vals, y_test = np.loadtxt(y_test_fn, delimiter=',', unpack=True)
        ys_test[i,:] = y_test

        y_pred_fn = f'{predictions_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
        _, y_pred = np.loadtxt(y_pred_fn, delimiter=',', unpack=True)
        ys_pred[i,:] = y_pred

        colors[i] = plt.cm.rainbow(color_idx[id_cosmo])

        err_frac = (y_pred - y_test)/y_test

        axarr[0].plot(r_vals, y_test, color=colors[i], alpha=alpha, label='Observed', 
                      ls='None', marker='o', markerfacecolor=None, zorder=zorders[i])
        axarr[0].plot(r_vals, y_pred, color=colors[i], alpha=alpha, label='Emu Predicted', 
                      marker=None, zorder=zorders[i]) 

        axarr[1].plot(r_vals, err_frac, color=colors[i], zorder=zorders[i])

    errs_frac = (ys_pred - ys_test)/ys_test
    err_frac_mean = np.std(errs_frac, axis=0)
    axarr[2].plot(r_vals, err_frac_mean, color='blue', label='error (stdev of fractional error)')

    axarr[0].set_xscale(scale_dict[statistic][0])
    axarr[0].set_yscale(scale_dict[statistic][1])
    axarr[0].set_ylabel(stat_labels[statistic])

    axarr[1].set_xscale(scale_dict[statistic][0])
    axarr[1].set_ylabel('fractional error')

    axarr[2].set_xscale(scale_dict[statistic][0])
    axarr[2].set_xlabel(r_labels[statistic])
    axarr[2].set_ylabel('error')