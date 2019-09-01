import pickle
import sys
from keras import models, Input, Model, Sequential
from keras.layers import Dense, concatenate, add
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt

# read data chair
with open('files\\train_data2.pickle', 'rb') as f:
    symmetry_funcs_train = pickle.load(f)

with open('files\\test_data2.pickle', 'rb') as f:
    symmetry_funcs_test = pickle.load(f)

with open('files\\train_energy2.pickle', 'rb') as f:
    energies_train = np.array(pickle.load(f))

with open('files\\test_energy2.pickle', 'rb') as f:
    energies_test = np.array(pickle.load(f))

# print(np.mean(symmetry_funcs_train), np.std(symmetry_funcs_train))
mean = np.mean(symmetry_funcs_train)
symmetry_funcs_train = np.array(symmetry_funcs_train) - mean
std = np.std(symmetry_funcs_train)
symmetry_funcs_train = symmetry_funcs_train / std
symmetry_funcs_train = list(symmetry_funcs_train)
symmetry_funcs_test = (np.array(symmetry_funcs_test) - mean) / std
symmetry_funcs_test = list(symmetry_funcs_test)
# print(np.mean(symmetry_funcs_train), np.std(symmetry_funcs_train))
# mean_en = np.mean(energies_train)
# energies_train -= mean
# std = np.std(energies_train)
# energies_train /= std
# energies_test -= mean
# energies_test /= std
sh = symmetry_funcs_train[0][0].shape
# print(n)
# sys.exit(0)

ac = 'sigmoid'


def create_tower():
    inputA = Input(shape=sh)
    model = Dense(50, activation=ac)(inputA)
    model = Dense(50, activation=ac)(model)
    # model = Dense(8, activation=ac)(model)
    # model = Dense(8, activation=ac)(model)
    # model = Dense(32, activation='tanh')(model)
    model = Dense(1)(model)
    model = Model(inputs=inputA, outputs=model)
    return model


# create model:
x = []
for i in range(13):
    x.append(create_tower())

added = add([i.output for i in x])
model = Model(inputs=[i.input for i in x], outputs=added)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(symmetry_funcs_train, energies_train, validation_data=(symmetry_funcs_test, energies_test),
                    epochs=150, batch_size=128, verbose=2)
model.save('files\\luckyModel2.h5')
pred = model.predict(symmetry_funcs_test)
# print(pred[::30])

with open('files\\predict_data1.pickle', 'wb') as f:
    pickle.dump(pred, f)

real = np.array(energies_test)
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(np.array(pred).reshape((len(real),)),
                     np.array(real).reshape((len(real),)))
print('R = {:.3f}'.format(r_value))
# 0.84 / 5.8
hist = history.history['val_mean_absolute_error']
plt.plot(range(1, len(hist) + 1), hist)
# plt.plot(np.array(pred).reshape((len(real),)),
#          np.array(real).reshape((len(real),)), 'ko')
# R = 0.994 вввв
plt.show()
