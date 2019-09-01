import numpy as np
import pickle

with open('files\\test_energy2.pickle', 'rb') as f:
    energies_test = np.array(pickle.load(f))

with open('files\\predict_data1.pickle', 'rb') as f:
    pred = np.array(pickle.load(f))

import matplotlib.pyplot as plt
pred = pred.reshape(51,)
# pred.shape, energies_test.shape
pred, energies_test = energies_test, pred
fig, ax = plt.subplots(1, figsize=(10, 5))
z = np.polyfit(pred, energies_test, 1)
p = np.poly1d(z)
ax.plot(pred, energies_test, 'Xk', markersize=7)
ax.plot(pred, p(pred), '-b', markersize=10)
ax.set_title('R = 0.95', style='italic', color='k', fontsize=16)
ax.set_xlabel('test values', color='k', fontsize=14)
ax.set_ylabel('predicted by neural net', color='k', fontsize=14)
fig.savefig('1.png')
