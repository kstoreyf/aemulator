from emulator import EmulatorMLP

statistic = 'xi'
train_tag = ''
predictions_dir = f'../predictions_{statistic}_MLP{train_tag}'
model_fn = f'../models/model_{statistic}_MLP{train_tag}.joblib'

emu = EmulatorMLP(statistic)

print("Setting training data")
emu.set_training_data()
print("Training")
emu.train()
emu.save_model(model_fn)

print("Setting testing data")
emu.set_testing_data()
print("Testing")
emu.load_model(model_fn)
emu.test()
print("Saving predictions")
emu.save_predictions(predictions_dir)
print("Emu done!")