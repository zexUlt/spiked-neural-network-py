import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator


class SpikedNN:
    def __init__(self,
                 state_size,
                 u_size,
                 k1,
                 k2,
                 nn1,
                 nn2,
                 wa_init,
                 wb_init,
                 p_init,
                 a_init,
                 sigma,
                 phi,
                 stop_time,
                 use_x_hat):
        self._state_size = state_size
        self._u_size = u_size

        self.k1 = k1
        self.k2 = k2

        self.nn1 = nn1
        self.nn2 = nn2

        self.Wa_init = wa_init
        self.Wb_init = wb_init

        self.A_init = a_init
        self.P_init = p_init

        self.stop_time = stop_time

        self.sigma = sigma
        self.phi = phi

        self.history_x = []
        self.history_Wa = []
        self.history_Wb = []
        self.history_dx_dt = []

        self.history_sigma = []
        self.history_phi_U = []
        self.history_loss = []

        self.use_x_hat = use_x_hat

    def law_Wa(self, psi, delta):
        res = self.k1 * self.P_init @ delta @ psi
        # (mat_K1@mat_P@delta_x@activation_values.T)
        return res

    def law_Wb(self, psi, delta):
        res = self.k2 * self.P_init @ delta @ psi
        return res

    def run(self, x, u, step_size):
        # Initialize vector
        self.history_x = []
        # cur_x = np.zeros(shape=(self._state_size, 1))
        cur_x = np.array([0.0, 1.0]).reshape((self._state_size, 1))
        self.history_x.append(cur_x.copy())

        # Identification error
        delta = np.zeros(x.shape)

        self.history_Wa = []
        cur_Wa = self.Wa_init.copy()
        self.history_Wa.append(self.Wa_init.copy())

        self.history_Wb = []
        cur_Wb = self.Wb_init.copy()
        self.history_Wb.append(self.Wb_init.copy())

        cur_P = self.P_init.copy()

        self.history_loss = []

        for i in tqdm(range(1, len(x))):
            cur_t = i * step_size

            delta_x = x[i] - cur_x
            self.history_loss.append(np.abs(delta_x))

            if self.use_x_hat:
                sigma_result = self.sigma(cur_x)
                phi_result_U = self.phi(cur_x) @ u[i]
            else:
                sigma_result = self.sigma(x[i])
                phi_result_U = self.phi(x[i]) @ u[i]

            self.history_sigma.append(sigma_result.copy())
            self.history_phi_U.append(phi_result_U.copy())

            if cur_t <= self.stop_time:
                dWa_dt = self.law_Wa(psi=sigma_result.T, delta=delta_x)
                dWb_dt = self.law_Wb(psi=phi_result_U.T, delta=delta_x)
            else:
                dWa_dt = np.zeros_like(self.Wa_init)
                dWb_dt = np.zeros_like(self.Wb_init)

            cur_Wa = step_size * dWa_dt
            self.history_Wa.append(cur_Wa.copy())

            cur_Wb = step_size * dWb_dt
            self.history_Wb.append(cur_Wb.copy())

            dx_dt = self.A_init @ cur_x + cur_Wa @ sigma_result + cur_Wb @ phi_result_U
            self.history_dx_dt.append(dx_dt.copy())

            cur_x += step_size * dx_dt
            self.history_x.append(cur_x.copy())

        return True
