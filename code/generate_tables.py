import numpy as np


def main():
    #generate_id_pairs_train()
    #generate_id_pairs_test()
    #generate_id_pairs_recovery_test()\
    generate_statistic_sets()

# non-overlapping HODs
def generate_id_pairs_train():
    fn_train = '../tables/id_pairs_train.txt'
    id_pairs_train = []
    ids_cosmo = range(40)
    n_hods_per_cosmo = 100
    for id_cosmo in ids_cosmo:
        id_hod_min = id_cosmo * n_hods_per_cosmo
        id_hod_max = id_hod_min + n_hods_per_cosmo
        ids_hod = range(id_hod_min, id_hod_max)
        for id_hod in ids_hod:
            id_pairs_train.append((id_cosmo, id_hod))
    np.savetxt(fn_train, id_pairs_train, delimiter=',', fmt=['%d', '%d'])

def generate_id_pairs_test():
    fn_test = '../tables/id_pairs_test.txt'
    id_pairs_test = []
    ids_cosmo = range(7)
    ids_hod = range(100)
    for id_cosmo in ids_cosmo:
        for id_hod in ids_hod:
            id_pairs_test.append((id_cosmo, id_hod))
    np.savetxt(fn_test, id_pairs_test, delimiter=',', fmt=['%d', '%d'])

# nice recovery set with non-overlapping HODs
def generate_id_pairs_recovery_test():
    #fn_recovery = '../tables/id_pairs_recovery_test.txt'
    fn_recovery = '../tables/id_pairs_recovery_test_70.txt'
    id_pairs_recovery = []
    ids_cosmo = range(7)
    n_hods_per_cosmo = 10
    for id_cosmo in ids_cosmo:
        id_hod_min = id_cosmo * 10
        id_hod_max = id_hod_min + n_hods_per_cosmo
        ids_hod = range(id_hod_min, id_hod_max)
        for id_hod in ids_hod:
            id_pairs_recovery.append((id_cosmo, id_hod))
    np.savetxt(fn_recovery, id_pairs_recovery, delimiter=',', fmt=['%d', '%d'])


def generate_statistic_sets():
    stat_str_sets = []

    statistics_all = ['wp', 'xi', 'xi2', 'upf', 'mcf']
    
    for s in statistics_all:
        # singles
        stat_str_sets.append(s)
    
    for s in statistics_all:
        # wp + second
        if s != 'wp':
            stat_str_sets.append(f'wp_{s}')
    
    stat_addin = []
    for s in statistics_all:
        # adding in one at a time
        stat_addin.append(s)
        stat_str_sets.append('_'.join(stat_addin))
    
    stat_str_sets_set_ordered = sorted(set(stat_str_sets), key=stat_str_sets.index)
    print(stat_str_sets_set_ordered)
    fn_stats = '../tables/statistic_sets.txt'
    np.savetxt(fn_stats, np.array(list(stat_str_sets_set_ordered)), fmt=['%s'])


if __name__=='__main__':
    main()
