import time

import emulator
import utils


statistic = 'mcf'
emu_name = 'George'
#emu_name = 'GPFlowVGP'
#emu_name = 'PyTorch'
max_iter = 1000
scaling = 'log'
#scaling = 'xrsqmean'
#train_tag = f'_{emu_name}_{scaling}'
mock_tag_train = '_aemulus_Msatmocks_train'
mock_tag_test = '_aemulus_Msatmocks_test'
train_tag = f'_{emu_name}_{scaling}_errstdev_Msatmocks'
train_mode = True
test_mode = True

predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
model_fn = f'../models/model_{statistic}{train_tag}' #emu will add proper file ending
scaler_x_fn = f'../models/scaler_x_{statistic}{train_tag}.joblib'
scaler_y_fn = f'../models/scaler_y_{statistic}{train_tag}.joblib'
#err_fn = f"../covariances/error_aemulus_{statistic}_hod3_test0.dat"
err_fn = f"../covariances/stdev{mock_tag_test}_{statistic}_hod3_test0.dat"
print("Model name:", model_fn)
print("Error filename:", err_fn)

Emu = utils.get_emu(emu_name)
print("Constructing emu")
emu = Emu(statistic, scaling, model_fn, scaler_x_fn, scaler_y_fn, err_fn,
          train_mode=train_mode, test_mode=test_mode,
          mock_tag_train=mock_tag_train, mock_tag_test=mock_tag_test)

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
