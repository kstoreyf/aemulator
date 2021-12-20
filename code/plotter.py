import h5py
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import gridspec
from scipy.stats import norm

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
    fig, axarr = plt.subplots(nrows, 1, figsize=(10,10), sharex=True, gridspec_kw={'height_ratios': [2,1,1]})
    plt.subplots_adjust(hspace=0)

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

        colors[i] = plt.cm.terrain(color_idx[id_cosmo])

        err_frac = (y_pred - y_test)/y_test
        
        ylabel = stat_labels[statistic]
        if statistic=='xi2':
            y_test *= r_vals**2
            y_pred *= r_vals**2
            axarr[1].set_ylim(-3,3)
            ylabel = r'$r^2$' + ylabel

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
    axarr[2].plot(r_vals, err_frac_p16, color='black', label='Emulator error (inner 68%)')
    axarr[2].plot(r_vals, err_frac_p84, color='black')

    err_fn = f"../../clust/covariances/error_aemulus_{statistic}_hod3_test0.dat"
    sample_var = np.loadtxt(err_fn)
    axarr[2].fill_between(r_dict[statistic], -sample_var, sample_var, color='lightblue', alpha=0.7)
    axarr[2].fill_between(r_dict[statistic], -sample_var/np.sqrt(5), sample_var/np.sqrt(5), color='steelblue', alpha=0.7)

    axarr[0].set_xscale(scale_dict[statistic][0])
    axarr[0].set_yscale(scale_dict[statistic][1])
    axarr[0].set_ylabel(ylabel)
    axarr[0].legend()
    axarr[0].xaxis.set_tick_params(direction='in', which='both')

    axarr[1].set_xscale(scale_dict[statistic][0])
    axarr[1].xaxis.set_tick_params(labelbottom=True)
    axarr[1].set_ylabel('fractional error')

    axarr[2].set_xscale(scale_dict[statistic][0])
    axarr[2].set_xlabel(r_labels[statistic])
    axarr[2].set_ylabel('error')
    handles, labels = axarr[2].get_legend_handles_labels()
    sample_var_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='Sample variance')
    sample_var_sqrt_patch = mpatches.Patch(color='steelblue', alpha=0.7, label=r'Sample variance / $\sqrt{N_\mathrm{boxes}}$')
    handles.append(sample_var_patch)
    handles.append(sample_var_sqrt_patch)
    axarr[2].legend(handles=handles)
    axarr[2].axhline(0, color='k', lw=0.5)


def plot_accuracy_figure(statistics, train_tags):
    
    fig = plt.figure(figsize=(20, 15))

    outer = gridspec.GridSpec(2, 3, wspace=0.5, hspace=0.25)

    for i, statistic in enumerate(statistics):
        train_tag = train_tags[i]
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, hspace=0,
                        subplot_spec=outer[i],
                        height_ratios=[2,1,1])

        ax0 = plt.Subplot(fig, inner[0])
        ax1 = plt.Subplot(fig, inner[1])
        ax2 = plt.Subplot(fig, inner[2])
        axarr = [ax0, ax1, ax2]

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

            colors[i] = plt.cm.terrain(color_idx[id_cosmo])

            err_frac = (y_pred - y_test)/y_test

            ylabel = stat_labels[statistic]
            if statistic=='xi2':
                y_test *= r_vals**2
                y_pred *= r_vals**2
                axarr[1].set_ylim(-3,3)
                ylabel = r'$s^2$' + ylabel

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
        axarr[2].plot(r_vals, err_frac_p16, color='black', label='Emulator error (inner 68%)')
        axarr[2].plot(r_vals, err_frac_p84, color='black')

        err_fn = f"../../clust/covariances/error_aemulus_{statistic}_hod3_test0.dat"
        sample_var = np.loadtxt(err_fn)
        axarr[2].fill_between(r_dict[statistic], -sample_var, sample_var, color='lightblue', alpha=0.7)
        axarr[2].fill_between(r_dict[statistic], -sample_var/np.sqrt(5), sample_var/np.sqrt(5), color='steelblue', alpha=0.7)

        axarr[0].set_xscale(scale_dict[statistic][0])
        axarr[0].set_yscale(scale_dict[statistic][1])
        axarr[0].set_ylabel(ylabel)
        axarr[0].xaxis.set_tick_params(direction='in', which='both')

        axarr[1].set_xscale(scale_dict[statistic][0])
        axarr[1].xaxis.set_tick_params(labelbottom=True)
        axarr[1].set_ylabel('frac. err.')

        axarr[2].set_xscale(scale_dict[statistic][0])
        axarr[2].set_xlabel(r_labels[statistic])
        axarr[2].set_ylabel('err.')
        axarr[2].axhline(0, color='k', lw=0.5)
        
        fig.add_subplot(axarr[0])
        fig.add_subplot(axarr[1])
        fig.add_subplot(axarr[2])
        
        fig.align_ylabels(axarr)

    handles, labels = axarr[0].get_legend_handles_labels()
    handles2, labels2 = axarr[2].get_legend_handles_labels()
    handles.extend(handles2)
    sample_var_patch = mpatches.Patch(color='lightblue', alpha=0.5, label='Sample variance')
    sample_var_sqrt_patch = mpatches.Patch(color='steelblue', alpha=0.7, label=r'Sample variance / $\sqrt{N_\mathrm{boxes}}$')
    handles.append(sample_var_patch) 
    handles.append(sample_var_sqrt_patch)
    plt.legend(handles=handles, fontsize=18, loc=(1.3, 0.9))


def plot_contours(chaintags, legend_labels=None, params_toplot=None, colors=None,
                  legend_loc='upper center', legend_fontsize=20,
                  vertical_markers=None, vertical_marker_color='grey', alpha=0.4):
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

        #chain_results_dir = '/home/users/ksf293/aemulator/chains/results'
        chain_results_dir = '/export/sirocco1/ksf293/aemulator/chains/results'
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
    g.settings.alpha_filled_add=alpha
    g.settings.figure_legend_frame = False
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axis_marker_lw = 1.0
    g.settings.axis_marker_color = 'dimgrey'
    g.triangle_plot(sample_arr, filled=True, contour_colors=colors, names=params_toplot,
                   legend_labels=legend_labels, markers=vertical_markers_toplot, title_limit=1, legend_loc=legend_loc,
                    marker_args=marker_args, axis_marker_color='red')
    return g


def plot_correlation_matrix(corr, statistics):

    nstats = len(statistics)
    plt.figure(figsize=(2.5*nstats,2.5*nstats))
    tick_labels = np.concatenate([np.round(r_dict[stat], 2) for stat in statistics])
    im = plt.imshow(corr, origin='lower left', cmap='bwr_r', vmin=-1, vmax=1)
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, rotation=90)
    plt.yticks(ticks=range(len(tick_labels)), labels=tick_labels)

    # Label statistics
    for i, statistic in enumerate(statistics):
        plt.text(9*i+3.5, -6, stat_labels[statistic], fontsize=20)
        plt.text(-6, 9*i+3.5, stat_labels[statistic], fontsize=20, rotation=90)
        if i==0:
            continue
        plt.axvline(9*i-0.5, color='k')
        plt.axhline(9*i-0.5, color='k')

    plt.xlabel(r"r ($h^{-1}$Mpc)", labelpad=40)
    plt.ylabel(r"r ($h^{-1}$Mpc)", labelpad=40)

    plt.colorbar(im, fraction=0.046, pad=0.04)


def plot_uncertainty_figure(results_dict, prior_dict, params_toplot, stat_strs_toplot, id_pairs, labels, colors, rotation=0,
                            nrows=2, ncols=2):
    subfig_width, subfig_height = (6,5)
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subfig_width*ncols, subfig_height*nrows))
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            label_xticks = False
            if i==(nrows-1):
                label_xticks = True
            plot_uncertainty_bar_chart(axarr[i,j], results_dict, prior_dict, params_toplot[count], stat_strs_toplot, id_pairs, 
                                              labels, colors, rotation=rotation, label_xticks=label_xticks)
            count += 1
            
            axarr[i,0].set_ylabel(fr"1/$\sigma$") # set ylabel on 1st column only (order [row, col])


def plot_uncertainty_bar_chart(ax, results_dict, prior_dict, param_toplot, stat_strs_toplot, id_pairs, labels, colors, 
                               rotation=0, label_xticks=False):
    xvals = range(len(stat_strs_toplot))

    uncertainties = np.empty(len(stat_strs_toplot))
    uncertainties_id_pairs = []
    yerrs_lo = np.empty(len(stat_strs_toplot))
    yerrs_hi = np.empty(len(stat_strs_toplot))
    for s, stat_str in enumerate(stat_strs_toplot):
        uncertainties_id_pairs = []
        for i, id_pair in enumerate(id_pairs):
            uncertainties_id_pairs.append(results_dict[stat_str][tuple(id_pair)][param_toplot]['uncertainty'])
        uncertainties[s] = np.mean(uncertainties_id_pairs)

        uncertainty_means_resampled = bootstrap(np.array(uncertainties_id_pairs), np.mean, n_resamples=100)
        inverse_uncertainty_means_resampled = 1/uncertainty_means_resampled
        yerrs_lo[s] = np.percentile(inverse_uncertainty_means_resampled, 16)
        yerrs_hi[s] = np.percentile(inverse_uncertainty_means_resampled, 84)

    # prior
    uncertainty_prior = prior_dict[param_toplot]['uncertainty']
    ax.axhline(1/uncertainty_prior, ls='--', color='grey')
    ax.text(0.3, 1.05*1/uncertainty_prior, 'prior', color='grey', fontsize=12)

    ax.bar(xvals, 1/uncertainties, yerr=[1/uncertainties-yerrs_lo, yerrs_hi-1/uncertainties], color=colors, width=0.2)#, tick_label=stat_strs)
    ax.set_xticks(xvals)
    if label_xticks:
        ax.set_xticklabels(labels, rotation=rotation)
    else:
        ax.set_xticklabels([])
        
    ax.set_title(fr'${param_labels[param_toplot]}$')


def plot_cumulative_dist_figure(results_dict, params_toplot, stat_strs_toplot, id_pairs, labels, colors, 
                                nrows=2, ncols=2, divide_by_error=False, value_to_compare='median'):
    subfig_width, subfig_height = (6,5)
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subfig_width*ncols, subfig_height*nrows))
    plt.subplots_adjust(hspace=0.22, wspace=0.15)
    
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            plot_cumulative_dist(results_dict, axarr[i,j], params_toplot[count], stat_strs_toplot, id_pairs, 
                                 labels, colors, divide_by_error=divide_by_error, value_to_compare=value_to_compare)
            count += 1
            
            if divide_by_error:
                axarr[nrows-1,j].set_xlabel(f"({value_to_compare} - truth)/uncertainty")
            else:
                axarr[nrows-1,j].set_xlabel(f"{value_to_compare} - truth")
        axarr[i,0].set_ylabel("fraction (cumulative)")
            
    handles, labels = axarr[0,0].get_legend_handles_labels()
    plt.legend(handles, labels, loc=(1.1,0.8))


def plot_cumulative_dist(results_dict, ax, param_toplot, stat_strs_toplot, id_pairs, labels, colors,
                         divide_by_error=False, value_to_compare='median'):

    for s, stat_str in enumerate(stat_strs_toplot):
        deltas = []
        for i, id_pair in enumerate(id_pairs):
            mean = results_dict[stat_str][tuple(id_pair)][param_toplot][value_to_compare]
            truth = results_dict[stat_str][tuple(id_pair)][param_toplot]['truth']
            uncertainty = results_dict[stat_str][tuple(id_pair)][param_toplot]['uncertainty']
            if divide_by_error:
                deltas.append((mean - truth)/uncertainty)
            else:
                deltas.append(mean - truth)
        N = len(deltas)
        deltas_sorted = np.sort(deltas)
        cdf_exact = np.array(range(N))/float(N)
        ax.plot(deltas_sorted, cdf_exact, color=colors[s], label=labels[s], lw=2)

    ax.axvline(0.0, color='k')
    ax.axhline(0.5, color='k')
    ax.set_title(fr'${param_labels[param_toplot]}$')
    if divide_by_error:
        xx = np.linspace(*ax.get_xlim())
        ax.plot(xx, norm(0, 1).cdf(xx), color='k', ls='--', label=r'$\mathcal{N}(0,1)$')

# copied from https://stackoverflow.com/a/49601444
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])



def plot_uncertainty_scales_figure(results_dicts, prior_dict, params_toplot, stat_strs_toplot, id_pairs, labels, colors, 
                                   scale_labels, rotation=0, nrows=2, ncols=2, fractional=False):
    subfig_width, subfig_height = (6,5)
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subfig_width*ncols, subfig_height*nrows))
    plt.subplots_adjust(hspace=0.15, wspace=0.15)
    
    

    count = 0
    for i in range(nrows):
        for j in range(ncols):
            label_xticks = False
            if i==(nrows-1):
                label_xticks = True
            plot_uncertainty_scales_bar_chart(axarr[i,j], results_dicts, prior_dict, params_toplot[count], stat_strs_toplot, id_pairs, 
                                              labels, colors, scale_labels, rotation=rotation, label_xticks=label_xticks,
                                              fractional=fractional)
            count += 1
            
            if fractional:
                ylabel = fr"$\Delta {params_toplot[count]} / {params_toplot[count]}$"
            else:
                ylabel = r"1/$\sigma$"
            axarr[i,0].set_ylabel(ylabel) # set ylabel on 1st column only (order [row, col])

    handles, labels = axarr[0,0].get_legend_handles_labels()
    plt.legend(handles, labels, loc=(1.1,0.8))


def plot_uncertainty_scales_bar_chart(ax, results_dicts, prior_dict, param_toplot, stat_strs_toplot, id_pairs, labels, colors, 
                                      scale_labels, rotation=0, label_xticks=False, fractional=False):
    bar_width = 0.25
    xvals = np.arange(len(stat_strs_toplot))
    lightness_vals = [1.6, 0.4, 1.0]
    
    for r, results_dict in enumerate(results_dicts):
        means = np.empty(len(stat_strs_toplot))
        uncertainties = np.empty(len(stat_strs_toplot))
        yerrs_lo = np.empty(len(stat_strs_toplot))
        yerrs_hi = np.empty(len(stat_strs_toplot))
        for s, stat_str in enumerate(stat_strs_toplot):
            uncertainties_id_pairs = []
            means_id_pairs = []
            for i, id_pair in enumerate(id_pairs):
                uncertainties_id_pairs.append(results_dict[stat_str][tuple(id_pair)][param_toplot]['uncertainty'])
                means_id_pairs.append(results_dict[stat_str][tuple(id_pair)][param_toplot]['mean'])
            uncertainties[s] = np.mean(uncertainties_id_pairs)
            means[s] = np.mean(means_id_pairs)
            uncertainty_means_resampled = bootstrap(np.array(uncertainties_id_pairs), np.mean, n_resamples=100)
            inverse_uncertainty_means_resampled = 1/uncertainty_means_resampled
            yerrs_lo[s] = np.percentile(inverse_uncertainty_means_resampled, 16)
            yerrs_hi[s] = np.percentile(inverse_uncertainty_means_resampled, 84)
        colors_adjusted = [adjust_lightness(color, lightness_vals[r]) for color in colors]
        if fractional:
            yvals = uncertainties/means
        else:
            yvals = 1/uncertainties
        ax.bar(xvals+r*bar_width, yvals, yerr=[1/uncertainties-yerrs_lo, yerrs_hi-1/uncertainties], color=colors_adjusted, width=0.2, label=scale_labels[r])

    # prior
    uncertainty_prior = prior_dict[param_toplot]['uncertainty']
    mean_prior = prior_dict[param_toplot]['mean']
    if fractional:
        yval_prior = uncertainty_prior/mean_prior
    else:
        yval_prior = 1/uncertainty_prior
    ax.axhline(yval_prior, ls='--', color='grey')
    ax.text(0.3, 1.05*1/uncertainty_prior, 'prior', color='grey', fontsize=12)

    ax.set_xticks(xvals+bar_width)
    if label_xticks:
        ax.set_xticklabels(labels, rotation=rotation)
    else:
        ax.set_xticklabels([])
        
    ax.set_title(fr'${param_labels[param_toplot]}$')


def plot_scale_dependence_figure(scales, results_dicts, prior_dict, params_toplot, stat_strs_toplot, 
                                 id_pair, labels, colors, 
                                 rotation=0, nrows=2, ncols=2, lss=None, lws=None,
                                 comparison_dicts=None, xlabel=None):
    subfig_width, subfig_height = (6,5)
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=(subfig_width*ncols, subfig_height*nrows))
    plt.subplots_adjust(hspace=0.35, wspace=0.15)
    
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            label_xticks = False
            if i==(nrows-1):
                label_xticks = True
            plot_scale_dependence(axarr[i,j], scales, results_dicts, prior_dict, params_toplot[count], 
                                  stat_strs_toplot, id_pair, labels, colors, 
                                  rotation=rotation, label_xticks=label_xticks,
                                  lss=lss, lws=lws, comparison_dicts=comparison_dicts, xlabel=xlabel)
            
            axarr[i,0].set_ylabel(r"1/$\sigma$") # set ylabel on 1st column only (order [row, col])
            
            count += 1
            
    handles, labels = axarr[0,0].get_legend_handles_labels()
    plt.legend(handles, labels, loc=(1.1,0.8))


def plot_scale_dependence(ax, scales, results_dicts, prior_dict, param_toplot, stat_strs_toplot, id_pairs, labels, colors, 
                          rotation=0, label_xticks=False, lss=None, lws=None, 
                          comparison_dicts=None, xlabel=None):
    
    if lss is None:
        lss = ['-']*len(stat_strs_toplot)
    if lws is None:
        lws = [1]*len(stat_strs_toplot)
    
    assert len(results_dicts)==len(scales), "Number of results dicts not same as number of scales!"
    for s, stat_str in enumerate(stat_strs_toplot):
        uncertainties_scales = np.empty(len(scales))
        for m, scale in enumerate(scales):
            uncertainties_scales_idpairs = np.empty(len(id_pairs))
            for i, id_pair in enumerate(id_pairs):
                uncertainties_scales_idpairs[i] = results_dicts[scale][stat_str][tuple(id_pair)][param_toplot]['uncertainty'] 
            uncertainties_scales[m] = np.mean(uncertainties_scales_idpairs)
            
        ax.plot(rlog, 1/uncertainties_scales, marker='None', color=colors[s], label=labels[s], ls=lss[s], lw=lws[s])
        ax.set_xscale('log')
        
        if comparison_dicts is not None:
            assert len(comparison_dicts)==len(scales), "Number of comparison dicts not same as number of scales!"
            uncertainties_scales_comp = np.empty(len(scales))
            for m, scale in enumerate(scales):
                uncertainties_scales_idpairs_comp = np.empty(len(id_pairs))
                for i, id_pair in enumerate(id_pairs):
                    #if stat_str in comparison_dicts[m] and tuple(id_pair) in comparison_dicts[m][stat_str]: 
                    uncertainties_scales_idpairs_comp[i] = comparison_dicts[scale][stat_str][tuple(id_pair)][param_toplot]['uncertainty'] 
                uncertainties_scales_comp[m] = np.mean(uncertainties_scales_idpairs_comp)
            
            r_intersect, y_intersect = utils.find_intersection_point(rlog, 1/uncertainties_scales, 1/uncertainties_scales_comp)
            ax.plot(r_intersect, y_intersect, marker='|', markersize=20, color=colors[s])
            #ax.plot(rlog, 1/uncertainties_scales_comp, marker='None', color=colors[s], 
            #        ls=lss[s], lw=lws[s], alpha=0.5)
            
            
    ax_t = ax.twiny()  # instantiate a second axes that shares the same x-axis
    ax_t.tick_params(axis='y')
    ax_t.set_xscale('linear')
    ax_t.set_xticks(np.log10(rlog))
    ax_t.set_xticklabels([int(r) for r in rlin])
    
    # prior
    uncertainty_prior = prior_dict[param_toplot]['uncertainty']
    mean_prior = prior_dict[param_toplot]['mean']

    adjust_by = 0.8
    yval_prior = 1/uncertainty_prior
    ax.axhline(yval_prior, ls='--', color='grey')
    ylim = ax.get_ylim()
    height = ylim[1]-ylim[0]
    p_to_max = ylim[1] - yval_prior
    #ax.set_ylim((ylim[0]-0.07*height, ylim[1]))
    #ax.text(0.3, adjust_by*yval_prior, 'prior', color='grey', fontsize=12)
    #ax.text(0.3, ylim[0]-0.03*height, 'prior', color='grey', fontsize=12)
    
    ax.set_ylim((yval_prior - p_to_max*0.1, ylim[1]))
    ax.text(0.3, yval_prior - p_to_max*0.07, 'prior', color='grey', fontsize=12)
    
    ax.set_title(fr'${param_labels[param_toplot]}$', fontsize=18, pad=10)    
    
    if label_xticks:
        ax.set_xlabel(xlabel)
    