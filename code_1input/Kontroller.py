import numpy as np
import osqp
from scipy import sparse

## Model Predictive Control##
class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints):
        # Parameter
        self.N = N    # Panjang horizon prediksi
        self.Q = Q    # Bobot kesalahan lateral prediksi
        self.R = R    # Bobot penalti input
        self.QN = QN  # Bobot kesalahan lateral di ujung horizon
        # Model
        self.model = model
        # Dimensi
        self.nx = self.model.n_states
        self.nu = 1
        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints
        # Penghitung jika solusi tidak ditemukan
        self.infeasibility_counter = 0
        # Current control signals
        self.current_control = np.zeros((self.nu * self.N))
        # Inisiasi masalah optimisasi
        self.optimizer = osqp.OSQP()

    # Parameter dari hasil pengolahan citra
    def kom(self, delta_s, kappa_ref, traj):
        self.delta_s = delta_s
        self.kappa_ref = kappa_ref
        self.traj = traj

    # Formulasi masalah
    def _init_problem(self):
        # Kendala
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        xmin = self.state_constraints['xmin']
        xmax = self.state_constraints['xmax']
        # Matriks Sistem LTV
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * self.N))
        # Referensi state dan input
        ur = np.zeros(self.nu * self.N)
        xr = np.zeros(self.nx * (self.N + 1))
        # Matriks untuk kendala berupa persamaan
        uq = np.zeros(self.N * self.nx)
        # Kendala dinamis state(tidak ada)
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        # Kendala dinamis input(tidak ada)
        umax_dyn = np.kron(np.ones(self.N), umax)
        # Panjang langkah
        delta_s = self.delta_s
        
        # Iterasi prediksi pada tiap titik horizon prediksi
        for n in range(self.N):
            # kurvatur pada titik ke n
            kappa_ref = self.kappa_ref
            # Definisi matriks A dan matriks B ke n
            A_lin, B_lin = self.model.linearize(kappa_ref, delta_s)
            A[(n + 1) * self.nx: (n + 2) * self.nx, n * self.nx:(n + 1) * self.nx] = A_lin
            B[(n + 1) * self.nx: (n + 2) * self.nx, n * self.nu:(n + 1) * self.nu] = B_lin
            # Perhitungan sinyal input, x
            ur[n * self.nu:(n + 1) * self.nu] = np.array([kappa_ref])
            # Definisi ulang kendala untuk setiap matriks prediksi
            uq[n * self.nx:(n + 1) * self.nx] = B_lin.dot(np.array([kappa_ref]))
        # Definisi titik lintasan yang terdeteksi, x\tilde (x1)
        xr[self.nx::self.nx] = self.traj
        # Matriks Persamaan untuk kendala
        Ax = sparse.kron(sparse.eye(self.N + 1), -sparse.eye(self.nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        # Matriks pertidaksamaan untuk kendala
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        # Matriks kendala digabungkan
        A = sparse.vstack([Aeq, Aineq], format='csc')
        # Batas atas dan batas bawah dari kendala dalam bentuk persamaan
        lineq = np.hstack([xmin_dyn, np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
        # Batas atas dan batas bawah dari kendala dalam bentuk pertidaksamaan
        x0 = np.array(self.model.spatial_state[:])
        leq = np.hstack([-x0, uq])
        ueq = leq
        # Combine upper and lower bound vectors
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])
        # Definisi bobot untuk fungsi objektif
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN, sparse.kron(sparse.eye(self.N), self.R)], format='csc')
        q = np.hstack([-np.tile(np.diag(self.Q.A), self.N) * xr[:-self.nx], -self.QN.dot(xr[-self.nx:]), -np.tile(np.diag(self.R.A), self.N) * ur])
        # Inisiasi operasi optimisasi
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self):
        # Jumlah variabel state(x) dan input(u)
        nx = self.model.n_states
        nu = 1
        # Operasi optimisasi
        self._init_problem()         # Masalah
        dec = self.optimizer.solve() # Solusi
        try:
            # input hasil optimisasi
            control_signals = np.array(dec.x[-self.N * nu:])
            #control_signals[0::1] = np.arctan(control_signals[0::1] * self.model.length) #radian
            control_signals[0::1] = np.rad2deg(np.arctan(control_signals[0::1] * self.model.length)) #derajat
            delta = control_signals[0]
            # Pembaruan input
            self.current_control = control_signals
            # definisi ulang input hasil optimisasi
            u = np.array([delta])
            # jika ditemukan solusi, maka feasible
            self.infeasibility_counter = 0
        except:
            # jika tidak ditemukan solusi
            # maka digunakan input dari prediksi sebelumnya
            id = nu * (self.infeasibility_counter + 1)
            u = np.array(self.current_control[id:id + 1])
            # jika tidak ditemukan solusi, maka infeasible
            self.infeasibility_counter += 1
        # jika infeasibility terjadi sebanyak horizon
        # maka seluruh operasi dihentikan
        if self.infeasibility_counter == (self.N - 1):
            exit(1)
        return u
