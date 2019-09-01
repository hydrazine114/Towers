import itertools
import re
import numpy as np
import pickle


def calc_neib_dist(inp_mass, num_neib):
    dist = np.sum((inp_mass[:, np.newaxis, :] - inp_mass[np.newaxis, :, :])
                  ** 2, axis=-1)
    closer = np.argsort(dist, axis=1)[:, :num_neib]
    closer_comb = []
    for i in closer:
        it = list(itertools.combinations(i[1:], 2))
        for j in it:
            closer_comb.append([i[0], *j])
    pass
    return closer, closer_comb


def calc_cos(a, b):
    return np.sum(a * b, axis=-1) / (np.sum(a ** 2, axis=-1) ** 0.5 * np.sum(b ** 2, axis=-1) ** 0.5)


def for_angle_func(molec, data):
    coss = []
    distances = []
    for i in range(len(data[1])):
        vecs = molec[data[1]][i][1:] - molec[data[1]][i][0]
        vecs2 = vecs[0] - vecs[1]
        vecs = np.vstack((vecs, vecs2))
        dist = np.sum(vecs ** 2, axis=-1) ** 0.5
        coss.append(calc_cos(vecs[0], vecs[1]))
        distances.append(list(dist))
    return coss, distances


def for_dist_func(molec, data):
    distances = []
    for i in range(len(data[0])):
        vecs = molec[data[0]][i][1:] - molec[data[0]][i][0]
        dist = np.sum(vecs ** 2, axis=-1) ** 0.5
        distances.append(dist)
    pass
    return distances


def fc(r, Rc):
    if r < Rc:
        return 0.5 * np.cos((np.math.pi * r) / Rc) + 1
    return 0


def G1_func(distances, Rc):
    G1 = []
    for i in distances:
        g = 0
        for r in i:
            g += fc(r, Rc)
        G1.append(g)
    return G1


def G2_func(distances, Rc, gapta, Rs):
    G2 = []
    for i in distances:
        g = 0
        for r in i:
            g += np.exp(-gapta * (r - Rs) ** 2) * fc(r, Rc)
        G2.append(g)
    return G2


def G3_func(distances, Rc, k):
    G3 = []
    for i in distances:
        g = 0
        for r in i:
            g += np.cos(k * r) * fc(r, Rc)
        G3.append(g)
    return G3


def G4_func(data, Rc, gapta, dzetta, lam, num_at, num_neib):
    num_neib -= 1
    num_comb = np.math.factorial(num_neib) // (np.math.factorial(num_neib - 2) * 2)
    # print(num_comb)
    G4 = []
    coss = data[0]
    dist = data[1]
    for i in range(num_at):
        g = 0
        for j in range(len(coss) // num_at):
            ri = dist[i * num_comb + j][0]
            rj = dist[i * num_comb + j][1]
            rk = dist[i * num_comb + j][2]
            g += (2 ** (1 - dzetta)) * ((1 + lam * coss[i]) ** dzetta) * \
                 np.exp(-gapta * (ri ** 2 + rj ** 2 + rk ** 2) * \
                        fc(ri, Rc) * fc(rj, Rc) + fc(rk, Rc))
        G4.append(g)
    return G4


def G5_func(data, Rc, gapta, dzetta, lam, num_at, num_neib):
    num_neib -= 1
    num_comb = np.math.factorial(num_neib) // (np.math.factorial(num_neib - 2) * 2)
    G5 = []
    coss = data[0]
    dist = data[1]
    for i in range(num_at):
        g = 0
        for j in range(len(coss) // num_at):
            ri = dist[i * num_comb + j][0]
            rj = dist[i * num_comb + j][1]
            g += 2 ** (1 - dzetta) * (1 + lam * coss[i]) ** dzetta * \
                 np.exp(-gapta * (ri ** 2 + rj ** 2) * \
                        fc(ri, Rc) * fc(rj, Rc))
        G5.append(g)
    return np.array(G5)


def give_all_G(func_num, molec, num_neib, Rc, gapta, dzetta, lam, num_at, Rs, k):
    G = []
    closer_neqib_data = calc_neib_dist(molec, num_neib)
    fordist = for_dist_func(molec, closer_neqib_data)
    forcos = for_angle_func(molec, closer_neqib_data)
    for i in range(func_num):
        G.append(G1_func(fordist, Rc=Rc))
        G.append(G2_func(fordist, Rc=Rc, gapta=gapta[i], Rs=Rs[i]))
        G.append(G3_func(fordist, Rc=Rc, k=k[i]))
        G.append(G4_func(forcos, Rc=Rc, gapta=gapta[i], dzetta=dzetta[i], lam=lam[i], num_at=num_at, num_neib=num_neib))
        G.append(G5_func(forcos, Rc=Rc, gapta=gapta[i], dzetta=dzetta[i], lam=lam[i], num_at=num_at, num_neib=num_neib))
    a = []
    G = np.array(G)
    for i in range(num_at):
        a.append(G[:, i])
    pass
    return np.array(a)


"""




"""


class SymmetryFunctions:
    def __init__(self, input_file, num_at, num_neib):
        self.energies_train = []
        self.energies_test = []
        self.symmetry_funcs_train = []
        self.coords = []
        self.energies = []
        self.input_file = input_file
        self.num_at = num_at
        self.num_neib = num_neib + 1
        self.symmetry_funcs_test = None

    def get_known_system(self):
        """

        :return: symmetry functions for atoms set
        with energy for one system, it mixes atom in system
        """
        mass = self.symmetry_funcs_train.copy()
        full_data = []
        for i in range(13):
            current = []
            for j in range(i, 13):
                for k in mass[j]:
                    current.append(k)

            for j in range(i):
                for k in mass[j]:
                    current.append(k)

            full_data.append(np.array(current))
        full_energy = self.energies_train * 13
        return full_data, full_energy

    def get_unknown_system(self):
        """

        :return: symmetry functions for atoms set
        """
        pass

    def read_file(self):
        """
        read file with systems
        :return: set of systems
        """
        count = 0
        for line in self.input_file:
            count += 1
            if line[:2] == 'Pt':
                self.coords.append(re.split(r'[\s]{4,}', line[20:60]))
            if count > 25925:
                try:
                    self.energies.append(float(line[-30:-15]))  # line[-30:-15]
                except Exception as e:
                    pass
        for i in range(len(self.coords)):
            for j in range(3):
                self.coords[i][j] = float(self.coords[i][j])

        self.coords = np.array(self.coords)
        self.energies = np.array(self.energies)
        self.coords = self.coords.reshape(len(self.coords) // self.num_at, self.num_at, 3)

    def create_func(self):
        """
        create set of symmetry functions for one system
        :return:
        """
        Rc = 10
        num_of_fun = 10
        gapta = np.linspace(0.05, 18, num_of_fun // 1)
        Rs = np.linspace(1, 5.3, num_of_fun // 1)
        k = np.linspace(1, 3.78, num_of_fun // 1)
        dzetta = np.linspace(32, 64, num_of_fun // 1)
        # gapta = np.concatenate((gapta, gapta), axis=0)
        # Rs = np.concatenate((Rs, Rs), axis=0)
        # k = np.concatenate((k, k), axis=0)
        # dzetta = np.concatenate((dzetta, dzetta), axis=0)
        lam = np.linspace(-1, 1, num_of_fun)
        symmetry_funcs_train = []
        symmetry_funcs_test = []
        num_of_count = 5
        num_of_struc = len(self.coords)  # len(self.coords)
        for i in range(num_of_struc):
            print('{:.3f}%'.format(i/num_of_struc * 100))
            if i > 50:  # i % num_of_count != 0:
                symmetry_funcs_train.append(give_all_G(func_num=num_of_fun,
                                                       molec=self.coords[i], num_neib=self.num_neib,
                                                       Rc=Rc, gapta=gapta, dzetta=dzetta, lam=lam,
                                                       num_at=self.num_at, Rs=Rs, k=k))
            else:
                symmetry_funcs_test.append(give_all_G(func_num=num_of_fun,
                                                      molec=self.coords[i], num_neib=self.num_neib,
                                                      Rc=Rc, gapta=gapta, dzetta=dzetta, lam=lam,
                                                      num_at=self.num_at, Rs=Rs, k=k))

        self.symmetry_funcs_train = np.zeros(
            shape=(len(symmetry_funcs_train[0]), len(symmetry_funcs_train), num_of_fun * 5))
        self.symmetry_funcs_test = np.zeros(
            shape=(len(symmetry_funcs_test[0]), len(symmetry_funcs_test), num_of_fun * 5))

        for i in range(len(symmetry_funcs_train[0])):
            for j in range(len(symmetry_funcs_train)):
                self.symmetry_funcs_train[i][j] = symmetry_funcs_train[j][i]

        for i in range(len(symmetry_funcs_test[0])):
            for j in range(len(symmetry_funcs_test)):
                self.symmetry_funcs_test[i][j] = symmetry_funcs_test[j][i]

        for i in range(num_of_struc):
            if i > 50:  # i % num_of_count != 0:
                self.energies_train.append(self.energies[i])
            else:
                self.energies_test.append(self.energies[i])

        self.symmetry_funcs_train = list(self.symmetry_funcs_train)
        self.symmetry_funcs_test = list(self.symmetry_funcs_test)


input_file = open('files\\plt.lm-Pt13')
sf = SymmetryFunctions(input_file, 13, 12)
sf.read_file()
sf.create_func()
full = sf.get_known_system()
with open('files\\train_data2.pickle', 'wb') as f:
    pickle.dump(full[0], f)

with open('files\\test_data2.pickle', 'wb') as f:
    pickle.dump(sf.symmetry_funcs_test, f)

with open('files\\train_energy2.pickle', 'wb') as f:
    pickle.dump(full[1], f)

with open('files\\test_energy2.pickle', 'wb') as f:
    pickle.dump(sf.energies_test, f)
print(np.array(full[0]).shape)
# with open('train_data.pickle', 'rb') as f:
#     train = pickle.load(f)
# 1677
