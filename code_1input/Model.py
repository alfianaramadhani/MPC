import numpy as np
from abc import abstractmethod

try:
    from abc import ABC
except:
    # for Python 2.7
    from abc import ABCMeta

    class ABC(object):
        __metaclass__ = ABCMeta
        pass

# Definisi kelas spasial digunakan untuk prediksi mpc
class SpatialState(ABC):
    @abstractmethod
    def __init__(self):
        self.members = None
        self.e_y = None
        self.e_psi = None

    # Definisi identitas unik state
    # Digunakan untuk melakukan operasi pangkat
    def __getitem__(self, item):
        if isinstance(item, int):
            members = [self.members[item]]
        else:
            members = self.members[item]
        return [vars(self)[key] for key in members]

    # Definisi jumlah state
    def __len__(self):
        return len(self.members)

    # Daftar state
    def list_states(self):
        return self.members

# Definisi referensi dan nama state, x_referensi dan x
class SimpleSpatialState(SpatialState):
    def __init__(self, e_y=0.0, e_psi=0.0): # referensi state, x_referensi
        super(SimpleSpatialState, self).__init__()
        # State
        self.e_y = e_y     # kesalahan lateral, x (x1)
        self.e_psi = np.degrees(e_psi) # kesalahan sudut yaw, x (x2)
        self.members = ['e_y', 'e_psi'] # Definisi daftar state

# Definisi Parameter pada prototipe
class SpatialBicycleModel(ABC):
    def __init__(self, length):
        self.length = length # Panjang prototipe
    @abstractmethod
    def linearize(self, kappa_ref, delta_s):
        pass

# Definisi Model yang digunakan
class BicycleModel(SpatialBicycleModel):
    def __init__(self, length):
        # inisiasi base class
        super(BicycleModel, self).__init__(length=length)
        self.spatial_state = SimpleSpatialState() # Definisi state
        self.n_states = len(self.spatial_state) # Definisi jumlah state

    # Definisi matriks yang digunakan untuk MPC
    def linearize(self, kappa_ref, delta_s):
        # Baris pada matriks A
        a_1 = np.array([1, delta_s]) # kesalahan lateral, x1
        a_2 = np.array([-kappa_ref ** 2 * delta_s, 1]) # kesalahan sudut yaw, x2
        # Baris pada matriks B
        b_1 = np.array([0])
        b_2 = np.array([delta_s])
        # Definisi  matriks A dan Matriks B
        A = np.stack((a_1, a_2), axis=0)
        B = np.stack((b_1, b_2), axis=0)
        return A, B
