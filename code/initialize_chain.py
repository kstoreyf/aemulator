from os import path
import numpy as np
from time import time
import h5py
import yaml
import warnings


def main(config_fname, overwrite_param_file=False, plaintext=False):
    """
    Control all other processes. Primarily just makes the hdf5 file in the designated
    location, and copies over relevant info to the attrs
    :param config_fname:
        Filename of a YAML config file.
    """

    if plaintext:
        cfg = yaml.safe_load(config_fname)
    else:

        assert path.isfile(config_fname), "%s is not a valid config filename."%config_fname

        with open(config_fname, 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)#, Loader=yaml.FullLoader)

    filename = cfg['save_fn']
    resultname = cfg['chain']['chain_results_fn']
    if path.exists(resultname):
         print(f"[Initialize chain] Chain results file {resultname} already exists in local directory, stopping!")
         return -1
    
    # PBS batch system can't see sirocco1, so no point in checking!
    #resultname_fn = path.basename(resultname)
    #resultname_in_datadir = f'/export/sirocco1/ksf293/aemulator/chains/results/{resultname_fn}'
    #if path.exists(resultname_in_datadir):
    #    print(f"[Initialize chain] Chain results file {resultname_in_datadir} already exists in big data directory, stopping!")
    #    return -1
                           
    if not overwrite_param_file and path.exists(filename):
         print(f"[Initialize chain] Chain param file {filename} already exists, stopping!")
         return -1

    if overwrite_param_file and path.exists(filename):
         print(f"[Initialize chain] Note: chain param file {filename} already exists, but \
             results file {resultname} does not, and overwrite_param_file=True, so overwriting \
             param file and rerunning chain!")

    #assert path.isfile(filename), "%s is not a valid output filename"%filename
    print('Fname', filename)
    f = h5py.File(filename, 'w')#, libver='lastest')
    #f.swmr_mode = True # enables the chains to be accessed while they're running

    emu_cfg = cfg['emu']
    data_cfg = cfg['data']
    chain_cfg = cfg['chain']

    emu_config(f, emu_cfg)
    data_config(f, data_cfg)
    chain_config(f, chain_cfg)

    f.close()

    return filename


def emu_config(f, cfg):
    """
    Attach the emu config info, putting in defaults for unspecified values
    :param f:
        File handle of hdf5 file
    :param cfg:
        Emu portion of the cfg
    """


    required_emu_keys = ['statistics', 'emu_names', 'scalings']
    for key in required_emu_keys:
        assert key in cfg, "%s not in config but is required."%key
        f.attrs[key] = cfg[key]

    #optional_keys = ['fixed_params', 'emu_hps', 'seed']
    optional_keys = ['nhods', 'err_fn']
    #default_vals = [{}, {}, {}, None] #gonna None all these if empty
    # want to clafiy nothing specified

    for key in optional_keys:
        if key in cfg:
            attr = cfg[key]
            attr = str(attr) if type(attr) is dict else attr
            f.attrs[key] = attr
        else:
            f.attrs[key] = float("NaN") 


def data_config(f, cfg):
    """
    Attach data config info.
    Additionally, compute new values, if required.
    :param f:
        A file hook to an HDF5 file
    :param cfg:
        cfg with the info for the data
    """
    #if 'true_data_fname' in cfg:
    #    f.attrs['true_data_fname'] = cfg['true_data_fname']
    #    f.attrs['true_cov_fname'] = cfg['true_cov_fname']
    #    data, cov = _load_data(cfg['true_data_fname'], cfg['true_cov_fname'])

    #else: #compute the data ourselves
    #    data, cov = _compute_data(f, cfg)

    required_data_keys = []
    for key in required_data_keys:
        assert key in cfg, "%s not in config but is required."%key
        f.attrs[key] = cfg[key]
    
    optional_keys = ['cosmo', 'hod', 'bins', 'data_name']

    # for bins, need to create a dataset, not an attribute, to handle
    # diff length bin lists for diff statistics (annoying i know)
    if 'bins' in cfg:
        dt = h5py.vlen_dtype(np.dtype('int32'))
        dset = f.create_dataset('bins', (len(cfg['bins']),), dtype=dt)
        for i, b_arr in enumerate(cfg['bins']):
            dset[i] = b_arr
    else:
        f.attrs['bins'] = float("NaN")

    for key in optional_keys:
        if key in cfg:
            if key=='bins':
                continue
            attr = cfg[key]
            attr = str(attr) if type(attr) is dict else attr
            f.attrs[key] = attr
        else:
            f.attrs[key] = float("NaN") 



def chain_config(f, cfg):
    """
    Attach relvant config info for the mcmc chains
    :param f:
        Handle to an HDF5 file to attach things to
    :param cfg:
        Cfg with MCMC data
    """

    required_mcmc_keys = ['chain_results_fn','param_names_vary']

    for key in required_mcmc_keys:
        assert key in cfg, "%s not in config but is required."%key
        f.attrs[key] = cfg[key]

    optional_keys = ['n_threads', 'dlogz', 'seed', 'n_bins', 'cov_fn', 'chain_results_fn']
    for key in optional_keys:
        if key in cfg:
            attr = cfg[key]
            attr = str(attr) if type(attr) is dict else attr
            f.attrs[key] = attr 
        else:
            if key=='nbins':
                f.attrs[key] = 9
            else:
                f.attrs[key] = float("NaN")


if __name__ == '__main__':
    from sys import argv
    print(argv[1])
    main(argv[1])
