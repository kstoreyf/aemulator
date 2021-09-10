import numpy as np
import time

import emulator
import utils
from utils import param_names


statistics = ['wp', 'xi', 'upf', 'mcf']
emu_names = ['George', 'George', 'George', 'George']
max_iter = 1000
scalings = ['log', 'log', 'log', 'log']

n_statistics = len(statistics)
emus = [None]*n_statistics
for i, statistic in enumerate(statistics):

    train_tag = f'_{emu_names[i]}_{scalings[i]}'
    predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
    model_fn = f'../models/model_{statistic}{train_tag}' #emu will add proper file ending
    scaler_x_fn = f'../models/scaler_x_{statistic}{train_tag}.joblib'
    scaler_y_fn = f'../models/scaler_y_{statistic}{train_tag}.joblib'
    err_fn = f"../../clust/covariances/error_aemulus_{statistic}_hod3_test0.dat"
    print("Model name:", model_fn)

    Emu = utils.get_emu(emu_names[i])
    print("Constructing emu")
    emu = Emu(statistic, scaling, model_fn, scaler_x_fn, scaler_y_fn, err_fn,
            predict_mode=True)

    print("Loading")
    emu.load_model()
    emus[i] = emu

start = time.time()
n_predict = 10
bounds = utils.get_bounds()
bounds_ordered_low = [bounds[pn][0] for pn in param_names]
bounds_ordered_high = [bounds[pn][1] for pn in param_names]

for _ in range(n_predict):
    x_to_predict = np.random.uniform(low=bounds_ordered_low, high=bounds_ordered_high)
    for i, emu in enumerate(emus):
        s = time.time()
        y_predict = emu.predict(x_to_predict)
        e = time.time()
        print("Stat:", statistics[i], "| Predict time:", e-s, "s | Prediction:", y_predict)

end = time.time()
print(f"Predict time: {end-start} s")
print("Done!")
