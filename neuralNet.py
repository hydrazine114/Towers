import pickle
import sys
from keras import models, Input, Model, Sequential
from keras.layers import Dense, concatenate, add
import numpy as np
from scipy.stats import stats
import matplotlib.pyplot as plt

# Читаем данные и приводим в более удобный для нейросети вид
with open('files\\train_data2.pickle', 'rb') as f:
    symmetry_funcs_train = pickle.load(f)

with open('files\\test_data2.pickle', 'rb') as f:
    symmetry_funcs_test = pickle.load(f)

with open('files\\train_energy2.pickle', 'rb') as f:
    energies_train = np.array(pickle.load(f))

with open('files\\test_energy2.pickle', 'rb') as f:
    energies_test = np.array(pickle.load(f))

mean = np.mean(symmetry_funcs_train)
symmetry_funcs_train = np.array(symmetry_funcs_train) - mean
std = np.std(symmetry_funcs_train)
symmetry_funcs_train = symmetry_funcs_train / std
symmetry_funcs_train = list(symmetry_funcs_train)
symmetry_funcs_test = (np.array(symmetry_funcs_test) - mean) / std
symmetry_funcs_test = list(symmetry_funcs_test)
sh = symmetry_funcs_train[0][0].shape

ac = 'sigmoid'


# Создаем образец одной "башни"
def create_tower():
    inputA = Input(shape=sh)
    model = Dense(50, activation=ac)(inputA)
    model = Dense(50, activation=ac)(model)
    model = Dense(1)(model)
    model = Model(inputs=inputA, outputs=model)
    return model


# создаем общую модель из 13 башен
x = []
n_towers = 13
for i in range(n_towers):
    x.append(create_tower())

added = add([i.output for i in x])
model = Model(inputs=[i.input for i in x], outputs=added)
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# обучаем и сохраняем нейросеть
history = model.fit(symmetry_funcs_train, energies_train, validation_data=(symmetry_funcs_test, energies_test),
                    epochs=150, batch_size=128, verbose=2)
model.save('files\\luckyModel2.h5')
pred = model.predict(symmetry_funcs_test)

# сохраняем предсказанные значения
with open('files\\predict_data1.pickle', 'wb') as f:
    pickle.dump(pred, f)
# выводим величину средней ошибки в зависимости от эпохи и финальных предсказаний r ** 2
real = np.array(energies_test)
slope, intercept, r_value, p_value, std_err = \
    stats.linregress(np.array(pred).reshape((len(real),)),
                     np.array(real).reshape((len(real),)))
print('R = {:.3f}'.format(r_value))
hist = history.history['val_mean_absolute_error']
plt.plot(range(1, len(hist) + 1), hist)
plt.show()
