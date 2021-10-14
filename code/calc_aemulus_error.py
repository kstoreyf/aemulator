import numpy as np

import utils



def main():
    run('wp_xi')

def run(stat_str):

    statistics = stat_str.split('_')
    errtag = '_hod3_test0'

    hod = 3 # choose a middle-of-the-road hod
    nbins = 9
    ncosmos = 7
    nboxes = 5
    ntests = 1

    cosmos = list(range(ncosmos))
    boxes = list(range(nboxes))
    tests = list(range(ntests)) 

    res_dir = '../../aemulator/covariances'

    devmean_arr = []
    for i, statistic in enumerate(statistics):
        testing_dir = f'../../clust/results_aemulus_test/results_{statistic}'
        devmean_arr.append(calculate_devmeans(testing_dir, statistic, hod, cosmos, boxes, tests))
    
    #compute covariance assuming the mean is zero, as that is the expectation value (should be unbiased)
    devmeans = np.concatenate(devmean_arr, axis=1)
    cov = utils.covariance(devmeans, zeromean=True)

    cov_fn = f"{res_dir}/cov_aemulus_{stat_str}{errtag}.dat"
    print(f"Saving to {cov_fn}")
    np.savetxt(cov_fn, cov)

    # save error for gp input error, and percentiles 
    # note that this is slightly different than standard dev, because of zero mean covariance
    err = np.diag(cov)
    np.savetxt(f"{res_dir}/error_aemulus_{stat_str}{errtag}.dat", err)
    
    p16 = np.percentile(devmeans, 16, axis=0)
    p84 = np.percentile(devmeans, 84, axis=0)
    np.savetxt(f"{res_dir}/p16_aemulus_{stat_str}{errtag}.dat", p16)
    np.savetxt(f"{res_dir}/p84_aemulus_{stat_str}{errtag}.dat", p84)



def calculate_devmeans(testing_dir, statistic, hod, cosmos, boxes, tests):

    devmeans = []
    for cosmo in cosmos:

        ys_box = [] #this will contain 5 statistics
        for box in boxes:
            
            # Compute the average over tests for a single box & model
            ys_test = []
            for test in tests:
                rad, y = np.loadtxt(f'{testing_dir}/{statistic}_cosmo_{cosmo}_Box_{box}_HOD_{hod}_test_{test}.dat', 
                                delimiter=',', unpack=True)
                ys_test.append(y)
            y_box = np.mean(ys_test, axis=0) #mean is our estimate for the statistic of the box with the given model

            ys_box.append(y_box)

        #The mean of the 5 boxes, for a given model (cosmo & HOD)
        y_mean = np.mean(ys_box, axis=0) 
        for y in ys_box:
            devmean = (y-y_mean)/y_mean
            devmeans.append(devmean)
    return devmeans


if __name__=="__main__":
    main()
