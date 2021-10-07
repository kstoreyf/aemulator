import h5py
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches

import getdist
from getdist import plots, MCSamples

import utils
from utils import *


def compare_accuracy(statistic, train_tags, labels, colors):
    nrows = 2
    fig, axarr = plt.subplots(nrows, 1, figsize=(10,10), gridspec_kw={'height_ratios': [1]*nrows})
    plt.subplots_adjust(hspace=0.25)

    fn_test = '../tables/id_pairs_test.txt'
    id_pairs_test = np.loadtxt(fn_test, delimiter=',', dtype='int')
    n_test = id_pairs_test.shape[0]

    ids_cosmo_unique = set(id_pairs_test[:,0])

    n_bins = 9
    y_test_dir = '/home/users/ksf293/clust/results_aemulus_test_mean'

    ys_test = np.empty((n_test, n_bins))
    for i in range(n_test):
        id_cosmo, id_hod = id_pairs_test[i]
        y_test_fn = f'{y_test_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_mean.dat'
        r_vals, y_test = np.loadtxt(y_test_fn, delimiter=',', unpack=True)
        ys_test[i,:] = y_test

    for j, train_tag in enumerate(train_tags):
        predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
        ys_pred = np.empty((n_test, n_bins))

        for i in range(n_test):
            id_cosmo, id_hod = id_pairs_test[i]
            y_pred_fn = f'{predictions_dir}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
            _, y_pred = np.loadtxt(y_pred_fn, delimiter=',', unpack=True)
            ys_pred[i,:] = y_pred

        errs_frac = (ys_pred - ys_test)/ys_test
        err_frac_std = np.std(errs_frac, axis=0)

        err_frac_p16 = np.percentile(errs_frac, 16, axis=0)
        err_frac_p84 = np.percentile(errs_frac, 84, axis=0)
        err_frac_symmetrized = (err_frac_p84 - err_frac_p16)/2.0

        axarr[0].plot(r_vals, err_frac_std, label=labels[j]+'(stdev)', color=colors[j], ls='--')
        axarr[0].plot(r_vals, err_frac_symmetrized, label=labels[j]+'(inner 68%)', color=colors[j], ls='-')
        axarr[1].plot(r_vals, err_frac_p16, label=labels[j], color=colors[j])
        axarr[1].plot(r_vals, err_frac_p84, label=labels[j], color=colors[j])

    axarr[0].set_xscale(scale_dict[statistic][0])
    axarr[0].set_xlabel(r_labels[statistic])
    axarr[0].set_ylabel('error')

    axarr[1].set_xscale(scale_dict[statistic][0])
    axarr[1].set_xlabel(r_labels[statistic])
    axarr[1].set_ylabel('error (16-84 percentile)')
    axarr[1].axhline(0, color='k', lw=0.5)

    axarr[0].legend()



def plot_accuracy(statistic, train_tag):
    nrows = 3
    fig, axarr = plt.subplots(nrows, 1, figsize=(10,15), gridspec_kw={'height_ratios': [1]*nrows})
    plt.subplots_adjust(hspace=0.12)

    fn_test = '../tables/id_pairs_test.txt'
    id_pairs_test = np.loadtxt(fn_test, delimiter=',', dtype='int')
    n_test = id_pairs_test.shape[0]

    ids_cosmo_unique = set(id_pairs_test[:,0])
    color_idx = np.linspace(0, 1, len(ids_cosmo_unique))

    n_bins = 9
    y_test_dir = '/home/users/ksf293/clust/results_aemulus_test_mean'
    predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
    ys_test = np.empty((n_test, n_bins))
    ys_pred = np.empty((n_test, n_bins))

    alpha = 0.4
    colors = np.empty((n_test, 4)) # 4 for RGBA
    zorders = np.arange(n_test)
    np.random.shuffle(zorders)
    for i in range(n_test):
        id_cosmo, id_hod = id_pairs_test[i]
        y_test_fn = f'{y_test_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_mean.dat'
        r_vals, y_test = np.loadtxt(y_test_fn, delimiter=',', unpack=True)
        ys_test[i,:] = y_test

        y_pred_fn = f'{predictions_dir}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
        _, y_pred = np.loadtxt(y_pred_fn, delimiter=',', unpack=True)
        ys_pred[i,:] = y_pred

        colors[i] = plt.cm.rainbow(color_idx[id_cosmo])

        err_frac = (y_pred - y_test)/y_test
        
        if statistic=='xi2':
            y_test *= r_vals**2
            y_pred *= r_vals**2
            axarr[1].set_ylim(-3,3)

        label_obs, label_pred = None, None
        if i==0:
            label_obs = 'Mock'
            label_pred = 'Emulator prediction'
        axarr[0].plot(r_vals, y_test, color=colors[i], alpha=alpha, label=label_obs, 
                      ls='None', marker='o', markerfacecolor=None, zorder=zorders[i])
        axarr[0].plot(r_vals, y_pred, color=colors[i], alpha=alpha, label=label_pred, 
                      marker=None, zorder=zorders[i]) 

        axarr[1].plot(r_vals, err_frac, color=colors[i], alpha=alpha, zorder=zorders[i])

    errs_frac = (ys_pred - ys_test)/ys_test
    #err_frac_mean = np.std(errs_frac, axis=0)
    err_frac_p16 = np.percentile(errs_frac, 16, axis=0)
    err_frac_p84 = np.percentile(errs_frac, 84, axis=0)
    err_frac_inner68 = (err_frac_p84 - err_frac_p16)/2.0
    #axarr[2].plot(r_vals, err_frac_mean, color='blue', label='error (stdev of fractional error)')
    #axarr[2].plot(r_vals, err_frac_inner68, color='blue', label='emulator error (inner 68%)')
    axarr[2].plot(r_vals, err_frac_p16, color='magenta', label='Emulator error (inner 68%)')
    axarr[2].plot(r_vals, err_frac_p84, color='magenta')

    err_fn = f"../../clust/covariances/error_aemulus_{statistic}_hod3_test0.dat"
    sample_var = np.loadtxt(err_fn)
    sample_var /= np.sqrt(5)
    axarr[2].fill_between(r_dict[statistic], -sample_var, sample_var, color='lightblue', alpha=0.5)

    axarr[0].set_xscale(scale_dict[statistic][0])
    axarr[0].set_yscale(scale_dict[statistic][1])
    axarr[0].set_ylabel(stat_labels[statistic])
    axarr[0].legend()

    axarr[1].set_xscale(scale_dict[statistic][0])
    axarr[1].set_ylabel('fractional error')

    axarr[2].set_xscale(scale_dict[statistic][0])
    axarr[2].set_xlabel(r_labels[statistic])
    axarr[2].set_ylabel('error')
    handles, labels = axarr[2].get_legend_handles_labels()
    sample_var_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Sample variance')
    handles.append(sample_var_patch)
    axarr[2].legend(handles=handles)
    axarr[2].axhline(0, color='k', lw=0.5)


def plot_contours(chaintags, legend_labels=None, params_toplot=None, colors=None,
                  legend_loc='upper center', legend_fontsize=20,
                  vertical_markers=None, vertical_marker_color='grey'):
    # Make dict of bounds for plot ranges
    bounds = utils.get_bounds()
    sample_arr = []
    for i, chaintag in enumerate(chaintags):

        chain_fn = f'../chains/param_files/chain_params_{chaintag}.h5'
        fw = h5py.File(chain_fn, 'r')
        param_names = fw.attrs['param_names_vary']
        if vertical_markers is None:
            vertical_markers_toplot = fw.attrs['true_values']
        else:
            vertical_markers_toplot = vertical_markers
        fw.close()

        chain_results_dir = '/home/users/ksf293/aemulator/chains/results'
        chain_results_fn = f'{chain_results_dir}/results_{chaintag}.pkl'
        with open(chain_results_fn, 'rb') as pf:
            res = pickle.load(pf)
            samples = res['samples']
            lnweight = np.array(res['logwt'])
            lnevidence = np.array(res['logz'])

        # Select only samples for desired parameters
        if params_toplot is None:
            params_toplot = param_names
        else:
            idxs = []
            for pm in params_toplot:
                idxs.append(np.where(param_names == pm))
            idxs = np.array(idxs).flatten()
            samples = samples[:,idxs]
            # Note: weight & evidence same for all params, shape (n_samps,), so don't need to slice
            vertical_markers_toplot = vertical_markers_toplot[idxs]

        # param_labels is in utils
        labels = [param_labels[pn] for pn in params_toplot]
        ranges = [bounds[pn] for pn in params_toplot]

        #[-1] bc just care about final evidence value
        weights = np.exp(lnweight - lnevidence[-1])
        weights = weights.flatten()

        samps = getdist.MCSamples(names=params_toplot, labels=labels)
        samps.setSamples(samples, weights=weights)
        sample_arr.append(samps)

    marker_args = {'color': vertical_marker_color}

    g = getdist.plots.get_subplot_plotter()
    g.settings.alpha_filled_add=0.4
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axis_marker_lw = 1.0
    g.settings.axis_marker_color = 'dimgrey'
    g.triangle_plot(sample_arr, filled=True, contour_colors=colors, names=params_toplot,
                   legend_labels=legend_labels, markers=vertical_markers_toplot, title_limit=1, legend_loc=legend_loc,
                    marker_args=marker_args, axis_marker_color='red')
    return g
