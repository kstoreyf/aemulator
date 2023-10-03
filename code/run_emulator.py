import time

import emulator
import utils


def main():
    # TODO make sure to manually update train_tag_extra w any changes!
    train_tag_extra = '_errstdev_fmaxmocks'
    #train_tag_extra = '_errstdev_fmaxmocks_kernelamp'
    #train_tag_extra = '_errstdev_Msatmocks_uchuuchi2nclosest2000'
    #train_tag_extra = '_errstdev_fmaxmocks_uchuuchi2nclosest2000_tiny'
    #statistics = ['wp', 'xi', 'xi2', 'upf', 'mcf']
    #statistics = ['wp', 'xi', 'xi2']
    #statistics = ['upf', 'mcf']
    #statistics = ['xi2', 'upf']
    statistics = ['xi2']
    scalings = ['logxrsq']
    #scalings = [utils.get_fiducial_emu_scaling(statistic) for statistic in statistics]
    #id_tag = '_aemulus_fmaxmocks_uchuuchi2nclosest2000'
    id_tag = '' # if not nclosest, just empty
    for i in range(len(statistics)):    
        run(statistics[i], scalings[i], train_tag_extra=train_tag_extra,
            train_mode=True, test_mode=True, id_tag=id_tag)


def run(statistic, scaling, emu_name='George', max_iter=1000,
        mock_name_train='aemulus_fmaxmocks_train', mock_name_test='aemulus_fmaxmocks_test',
        train_tag_extra='', train_mode=True, test_mode=True,
        id_tag=''):

    #train_tag = f'_{emu_name}_{scaling}'
    #train_tag = f'_{emu_name}_{scaling}_errstdev_Msatmocks_kM32ExpConst2'
    train_tag = f'_{emu_name}_{scaling}{train_tag_extra}'

    predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
    models_dir = '/mount/sirocco1/ksf293/aemulator/models'
    model_fn = f'{models_dir}/model_{statistic}{train_tag}' #emu will add proper file ending
    scaler_x_fn = f'{models_dir}/scaler_x_{statistic}{train_tag}.joblib'
    scaler_y_fn = f'{models_dir}/scaler_y_{statistic}{train_tag}.joblib'
    #err_fn = f"../covariances/error_aemulus_{statistic}_hod3_test0.dat"
    err_fn = f"../covariances/stdev_{mock_name_test}_{statistic}_hod3_test0.dat"
    print("Model name:", model_fn)
    print("Error filename:", err_fn)

    Emu = utils.get_emu(emu_name)
    print("Constructing emu")
    emu = Emu(statistic, scaling, model_fn, scaler_x_fn, scaler_y_fn, err_fn,
            train_mode=train_mode, test_mode=test_mode,
            mock_name_train=mock_name_train, mock_name_test=mock_name_test,
            id_tag=id_tag)

    if train_mode:
        print("Training")
        s = time.time()
        emu.train(max_iter=max_iter)
        emu.save_model()
        e = time.time()
        print(f"Train time: {e-s} s = {(e-s)/60} min")

    if test_mode:
        print("Testing")
        s = time.time()
        emu.load_model()
        emu.test()
        print("Saving predictions")
        emu.save_predictions(predictions_dir)
        e = time.time()
        print(f"Test time: {e-s} s")
        print("Emu done!")


if __name__=='__main__':
    main()