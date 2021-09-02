import numpy as np
from matplotlib import pyplot as plt

from utils import *


def compare_accuracy(statistic, train_tags, labels, colors):
    nrows = 2
    fig, axarr = plt.subplots(nrows, 1, figsize=(10,10), gridspec_kw={'height_ratios': [1]*nrows})
    plt.subplots_adjust(hspace=0.15)

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
            y_pred_fn = f'{predictions_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
            _, y_pred = np.loadtxt(y_pred_fn, delimiter=',', unpack=True)
            ys_pred[i,:] = y_pred

        errs_frac = (ys_pred - ys_test)/ys_test
        err_frac_std = np.std(errs_frac, axis=0)
        axarr[0].plot(r_vals, err_frac_std, label=labels[j], color=colors[j])

    axarr[0].set_xscale(scale_dict[statistic][0])
    axarr[0].set_xlabel(r_labels[statistic])
    axarr[0].set_ylabel('error')

    axarr[0].legend()



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
    predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
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




def compare_emulators(statistic, testtags, acctags, errtag, savetags, labels=None, subsample=None, nbins=9, remove=None,
                 nhods=100):
    
    if labels==None:
        labels = acctags
    
    ncols = 2
    fig, ax = plt.subplots(ncols, 1, figsize=(10,10), gridspec_kw={'height_ratios': [1]*ncols})

    if statistic == 'upf':
        plt.title('UPF')
    elif statistic == 'wp':
        plt.title(r'$w_p$($r_p$)')
    
    CC_test = range(0, 7)
    HH_test = range(0, nhods)
    if remove:
        for rval in remove:
            #HH_test.remove(rval)
            CC_test.remove(rval)
 
    upf_mean = np.zeros(nbins)
    
    y_test_dir = '/home/users/ksf293/clust/results_aemulus_test_mean'

    Nemu = len(testtags)
    color_idx = np.linspace(0, 1, Nemu)

    for ee in range(Nemu):
        testtag = testtags[ee]
        acctag = acctags[ee]
        color=plt.cm.cool(color_idx[ee])
    
        i = 0
        fracerrs = []

        for cosmo in CC_test:
            for hod in HH_test:
                hod = int(hod)
                if "mean" in acctag:
                    idtag = '{}_cosmo_{}_HOD_{}_mean'.format(statistic, cosmo, hod)
                else:
                    idtag = '{}_cosmo_{}_Box_0_HOD_{}_test_0'.format(statistic, cosmo, hod)
                fnt = '{}testing_{}{}/{}.dat'.format(res_dir, statistic, testtag, idtag)

                y_test_fn = f'{y_test_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}_mean.dat'
                r_vals, y_test = np.loadtxt(y_test_fn, delimiter=',', unpack=True)
                
                lw =1
                rtest, ptest = np.loadtxt(fnt)

                fnp = '../testing_results/predictions_{}{}/{}.dat'.format(statistic, acctag, idtag)
                rpredic, ppredic = np.loadtxt(fnp, delimiter=',', unpack=True)

                y_pred_fn = f'{predictions_dir}/results_{statistic}/{statistic}_cosmo_{id_cosmo}_HOD_{id_hod}.dat'
                _, y_pred = np.loadtxt(y_pred_fn, delimiter=',', unpack=True)

                fracerr = (ppredic-ptest)/ptest
                fracerrs.append(fracerr)

                i += 1

        fracerrs = np.array(fracerrs)

        std = np.std(fracerrs, axis=0)
        p16 = np.percentile(fracerrs, 16, axis=0)
        p84 = np.percentile(fracerrs, 84, axis=0)
        pavg = (np.percentile(fracerrs, 84, axis=0)-np.percentile(fracerrs, 16, axis=0))/2.0
        p68 = np.percentile(fracerrs, 68, axis=0)

        ax[0].plot(rtest[:nbins], std[:nbins], color=color, ls='-', label=labels[ee])
        ax[1].plot(rtest[:nbins], p16[:nbins], color=color, ls='-')
        ax[1].plot(rtest[:nbins], p84[:nbins], color=color, ls='-')
        #ax[1].plot(rtest[:nbins], p68[:nbins], color='limegreen', ls='-', label='p68')
        #ax[1].plot(rtest[:nbins], pavg[:nbins], color='orange', ls='-', label='pavg')

    # if multiple savetags, aka they are different:
    single_savetag = False
    if type(savetags) is str:
        savetags = [savetags]
        single_savetag = True
    
    #lss = ['-', '--', ':']
    for ee, savetag in enumerate(savetags):
        stat_str = statistic
        err_str = errtag
        cov_dir = "../../clust/covariances/"
        #gperr = np.loadtxt(cov_dir+"error_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))
        gpcov = np.loadtxt(cov_dir+"cov_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))
        gperr = np.sqrt(1./5.*np.diag(gpcov))
        gpp16 = np.loadtxt(cov_dir+"p16_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))
        gpp84 = np.loadtxt(cov_dir+"p84_aemulus_{}{}{}.dat".format(stat_str, err_str, savetag))

        color=plt.cm.cool(color_idx[ee])
        ls = '--'
        if single_savetag:
            color='r'
            ls='-'
        ax[0].plot(rtest[:nbins], gperr[:nbins], color=color, ls=ls, label=f'Aemulus error, {savetag[1:]}')
        ax[1].plot(rtest[:nbins], gpp16[:nbins], color=color, ls=ls)
        ax[1].plot(rtest[:nbins], gpp84[:nbins], color=color, ls=ls)
        ax[1].axhline(0, color='k', ls=':')

        ax[0].set_ylabel("fractional error (std)")
        ax[1].set_ylabel("fractional error (16/84 percentile)")

    ax[0].legend()
