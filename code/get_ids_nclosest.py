import numpy as np

import utils


statistics = ['wp', 'xi', 'xi2', 'upf', 'mcf']

data_tag = '_uchuu'

mock_name = 'aemulus_fmaxmocks'
mock_name_train = f'{mock_name}_train'
mock_name_test = f'{mock_name}_test_mean'
mock_tag_train = f'_{mock_name_train}'
mock_tag_test = f'_{mock_name_test}'

ys_uchuu = []
rs = []
for statistic in statistics:    
    result_dir=f"/home/users/ksf293/clust/results{data_tag}/results_{statistic}"
    fn_stat=f"{result_dir}/{statistic}{data_tag}.dat"
    r, y_vals = np.loadtxt(fn_stat, delimiter=',', unpack=True)
    ys_uchuu.append(y_vals)
    rs.append(r)
ys_uchuu = np.array(ys_uchuu)
print(ys_uchuu.shape)


# training data
training_dir_base = f'/home/users/ksf293/clust/results{mock_tag_train}'
id_pairs_train = utils.load_id_pairs_train(mock_name_train)

y_train_arr = []
for i, statistic in enumerate(statistics):
    _, y_arr = utils.load_statistics(statistic, mock_name_train, id_pairs_train)
    y_train_arr.append(y_arr)
y_train_arr = np.array(y_train_arr)
print(y_train_arr.shape)

# use just the data covariance when calculating chi2 to choose the closest models, bc doesn't involve emu yet
mock_name_glam = 'glam'
stat_str = '_'.join(statistics)
cov_glam_fn = f'../covariances/cov_{mock_name_glam}_{stat_str}.dat'
cov_glam = np.loadtxt(cov_glam_fn)
L_uchuu = 2000.
L_glam = 1000.
cov_glam_scaled_uchuu = cov_glam*(L_glam/L_uchuu)**3
variances = np.diag(cov_glam_scaled_uchuu)
ys_uchuu_flat = np.hstack(ys_uchuu)
variances_nonfrac = variances*ys_uchuu_flat**2

n_closest = 2000
idxs_train_closest, chisq_thresh = utils.get_closest_models(ys_uchuu, y_train_arr, 
                                                          variances_nonfrac, n_closest=n_closest)
id_pairs_train_closest = id_pairs_train[idxs_train_closest]

fn_ids_train = f'../tables/id_pairs_train_{mock_name}_uchuuchi2nclosest{n_closest}.txt'
np.savetxt(fn_ids_train, id_pairs_train_closest, delimiter=',', fmt=['%d', '%d'])


# testing data
testing_dir_base = f'/home/users/ksf293/clust/results{mock_tag_test}'
id_pairs_test = utils.load_id_pairs_test()

y_test_arr = []
for i, statistic in enumerate(statistics):
    _, y_arr = utils.load_statistics(statistic, mock_name_test, id_pairs_test)
    y_test_arr.append(y_arr)

idxs_test_closest = utils.get_models_within_chi2(ys_uchuu, y_test_arr, 
                                                variances_nonfrac, chisq_thresh)
id_pairs_test_closest = id_pairs_test[idxs_test_closest]
print(len(id_pairs_test_closest))

fn_ids_test = f'../tables/id_pairs_test_{mock_name}_uchuuchi2nclosest{n_closest}.txt'
np.savetxt(fn_ids_test, id_pairs_test_closest, delimiter=',', fmt=['%d', '%d'])