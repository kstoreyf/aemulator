import numpy as np

import utils


def main():
    #stat_strs = np.loadtxt('../tables/statistic_sets.txt', dtype=str)
    stat_strs = ['upf']
    mock_tag = '_aemulus_fmaxmocks_test'
    train_tag_extra = '_errstdev_fmaxmocks'
    for stat_str in stat_strs:
        run(mock_tag,stat_str, train_tag_extra)

def run(mock_tag, stat_str, train_tag_extra='', cov_tag_extra='',
        id_tag=''):

    statistics = stat_str.split('_')

    errtag = '_hod3_test0'

    stat_str = '_'.join(statistics)
    cov_dir = '../covariances'

    emu_names = [utils.get_fiducial_emu_name(statistic) for statistic in statistics]
    scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    train_tags_extra = [train_tag_extra]*len(statistics)

    fracerr_arrs = []
    for i, statistic in enumerate(statistics):
        train_tag = f'_{emu_names[i]}_{scalings[i]}{train_tags_extra[i]}'
        print("Computing emu error for", statistic, train_tag)
        fracerrs = load_fracerrs_aemulus(statistic, mock_tag, train_tag, id_tag=id_tag)
        fracerr_arrs.append(fracerrs)

    fracerrs = np.concatenate(fracerr_arrs, axis=1)
    cov_perf = utils.covariance(fracerrs, zeromean=True)
    print(cov_perf)
    save_fn_perf = f"{cov_dir}/cov_emuperf{mock_tag}{cov_tag_extra}_{stat_str}{errtag}.dat"
    print('Saving cov_perf to', save_fn_perf)
    np.savetxt(save_fn_perf, cov_perf)
    
    # p16 = np.percentile(fracerrs, 16, axis=0)
    # p84 = np.percentile(fracerrs, 84, axis=0)
    # save_fn_p16_perf = f"{cov_dir}/p16_emuperf{mock_tag}{cov_tag_extra}_{stat_str}{errtag}.dat"
    # save_fn_p84_perf = f"{cov_dir}/p84_emuperf{mock_tag}{cov_tag_extra}_{stat_str}{errtag}.dat"
    #np.savetxt(save_fn_p16_perf, p16)
    #np.savetxt(save_fn_p84_perf, p84)


def load_fracerrs_aemulus(statistic, mock_tag, train_tag, id_tag):

    testing_dir = f'/mount/sirocco1/ksf293/clust/results{mock_tag}_mean/results_{statistic}'    
    predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'

    ptests = []
    ppredicts = []

    id_pairs_test = utils.load_id_pairs_test(id_tag=id_tag)

    for cosmo, hod in id_pairs_test:
 
        id_tag = f'cosmo_{cosmo}_HOD_{hod}'
        ntest, ptest = np.loadtxt(f'{testing_dir}/{statistic}_{id_tag}_mean.dat', 
                                delimiter=',', unpack=True)
        npredict, ppredict = np.loadtxt(f'{predictions_dir}/{statistic}_{id_tag}.dat', 
                                delimiter=',', unpack=True)
        
        ptests.append(ptest)
        ppredicts.append(ppredict)
    
    fracerrs = (np.array(ppredicts)-np.array(ptests))/np.array(ptests)
    return fracerrs


if __name__=='__main__':
    main()
