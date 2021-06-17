from emulator import EmulatorMLP

statistic = 'xi'
emu_name = 'MLP'
train_tag = f'_{emu_name}'
predictions_dir = f'../predictions/predictions_{statistic}{train_tag}'
model_fn = f'../models/model_{statistic}{train_tag}.joblib'
scaler_x_fn = f'../models/scaler_x_{statistic}{train_tag}.joblib'
scaler_y_fn = f'../models/scaler_y_{statistic}{train_tag}.joblib'

emu_dict = {'MLP': EmulatorMLP}
Emu = emu_dict[emu_name]
print("Constructing emu")
emu = Emu(statistic, model_fn, scaler_x_fn, scaler_y_fn)

#print("Setting training data")
#emu.set_training_data()
print("Training")
emu.train()
emu.save_model()

#print("Setting testing data")
#emu.set_testing_data()
print("Testing")
emu.load_model()
emu.test()
print("Saving predictions")
emu.save_predictions(predictions_dir)
print("Emu done!")