import time

import emulator


statistic = 'mcf'
emu_name = 'George'
#emu_name = 'GPFlowVGP'
#emu_name = 'PyTorch'
max_iter = 1000
# defaults for george: 'logscaler_gpmean_logabserr_matchorig_pool'
train_tag = f'_{emu_name}_george4_n10test'
#train_tag = f'_{emu_name}_meanscaler_0gpmean'
#train_tag = f'_{emu_name}_meanscaler_ndimkernel_maxiter10000'
#train_tag = f'_{emu_name}_logscaler_ndimkernel_maxiter10000_bounds_noisevar_clean'
predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
model_fn = f'../models/model_{statistic}{train_tag}' #emu will add proper file ending
scaler_x_fn = f'../models/scaler_x_{statistic}{train_tag}.joblib'
scaler_y_fn = f'../models/scaler_y_{statistic}{train_tag}.joblib'
err_fn = f"../../clust/covariances/error_aemulus_{statistic}_hod3_test0.dat"

emu_dict = {'MLP': emulator.EmulatorMLP,
            'GPFlow': emulator.EmulatorGPFlow, 
            'GPFlowVGP': emulator.EmulatorGPFlowVGP,
            'GPFlowBinned': emulator.EmulatorGPFlowBinned,
            'George': emulator.EmulatorGeorge,
            'GeorgeOrig': emulator.EmulatorGeorgeOrig,
            'PyTorch': emulator.EmulatorPyTorch}
Emu = emu_dict[emu_name]
print("Model name:", model_fn)
print("Constructing emu")
emu = Emu(statistic, model_fn, scaler_x_fn, scaler_y_fn, err_fn)

print("Training")
s = time.time()
emu.train(max_iter=max_iter)
emu.save_model()
e = time.time()
print(f"Train time: {e-s} s = {(e-s)/60} min")

print("Testing")
s = time.time()
emu.load_model()
emu.test()
print("Saving predictions")
emu.save_predictions(predictions_dir)
e = time.time()
print(f"Test time: {e-s} s")
print("Emu done!")
