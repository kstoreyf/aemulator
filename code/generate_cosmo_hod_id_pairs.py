import numpy as np


def main():
    generate_id_pairs_train()
    generate_id_pairs_test()

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

if __name__=='__main__':
    main()