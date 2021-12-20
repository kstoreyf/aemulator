import os
import yaml


count_exist = 0
count_dontexist = 0
config_dir = "/home/users/ksf293/aemulator/chains/configs"
for config_fname in os.listdir(config_dir):
    print("Config file:", config_fname)
    with open(f"{config_dir}/{config_fname}", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        param_filename = cfg['save_fn']
        resultname = cfg['chain']['chain_results_fn']
        resultname_filename = os.path.basename(resultname)
        results_dir_export = '/export/sirocco1/ksf293/aemulator/chains/results'
        resultname_export = f'{results_dir_export}/{resultname_filename}'
        if os.path.exists(resultname_export):
            print(f"Results file {resultname_export} exists, continuing")
            count_exist += 1
        else:
            print(f"Results file {resultname_export} doesn't exist!")
            if os.path.exists(param_filename):
                print(f"Deleting param h5 file {param_filename}")
                os.remove(param_filename)
            else:
                print(f"Param file {param_filename} doesn't exist, continuing")
            count_dontexist += 1
print(count_exist, count_dontexist)

