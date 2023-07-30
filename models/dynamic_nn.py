from .base import SpikedNN

import numpy as np
from tqdm import tqdm
from activation_functions import Izhikevich, Sigmoid
from projectors import Projector


class ProjectedSDNN(SpikedNN):
    def __init__(self,
                 state_size: int,
                 u_size: int,
                 k1: np.array,
                 k2: np.array,
                 nn1: int,
                 nn2: int,
                 wa_init: np.array,
                 wb_init: np.array,
                 p_init: np.array,
                 a_init: np.array,
                 projector: Projector,
                 sigma: Izhikevich | Sigmoid,
                 phi: Izhikevich | Sigmoid,
                 stop_time: float,
                 use_x_hat: bool):
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

        self.proj = projector

        self.sigma_1 = sigma
        self.sigma_2 = phi

        self.history_x = []
        self.history_Wa = []
        self.history_Wb = []
        self.history_dx_dt = []

        self.history_sigma = []
        self.history_phi_U = []
        self.history_loss = []

        self.use_x_hat = use_x_hat

    def _law_Wa(self, psi, delta):
        res = self.k1 * self.P_init @ delta @ psi
        # (mat_K1@mat_P@delta_x@activation_values.T)
        return res

    def _law_Wb(self, psi, delta):
        res = self.k2 * self.P_init @ delta @ psi
        return res

    def fit(self, x, u, step):
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
            cur_t = i * step

            delta_x = x[i] - self.proj(cur_x)
            self.history_loss.append(np.abs(delta_x))

            if self.use_x_hat:
                sigma_result = self.sigma_1(self.proj(cur_x))
                phi_result_U = self.sigma_2(self.proj(cur_x)) @ u[i]
            else:
                sigma_result = self.sigma_1(x[i])
                phi_result_U = self.sigma_2(x[i]) @ u[i]

            self.history_sigma.append(sigma_result.copy())
            self.history_phi_U.append(phi_result_U.copy())

            if cur_t <= self.stop_time:
                dWa_dt = self._law_Wa(psi=sigma_result.T, delta=delta_x)
                dWb_dt = self._law_Wb(psi=phi_result_U.T, delta=delta_x)
            else:
                dWa_dt = np.zeros_like(self.Wa_init)
                dWb_dt = np.zeros_like(self.Wb_init)

            cur_Wa = step * dWa_dt
            self.history_Wa.append(cur_Wa.copy())

            cur_Wb = step * dWb_dt
            self.history_Wb.append(cur_Wb.copy())

            dx_dt = self.A_init @ self.proj(cur_x) + cur_Wa @ sigma_result + cur_Wb @ phi_result_U
            self.history_dx_dt.append(dx_dt.copy())

            cur_x += step * dx_dt
            self.history_x.append(cur_x.copy())

        return True

    def predict(self, x: np.array, u: np.array, step: float = 0.001) -> np.array:
        prediction = x.copy()

        Wa = self.history_Wa[-1]
        Wb = self.history_Wb[-1]

        for i in tqdm(range(len(x))):
            sigma_out = self.sigma_1(prediction[i])
            phi_out = self.sigma_2(prediction[i])

            prediction[i + 1] += step * (self.A_init @ prediction[i] + Wa @ sigma_out + Wb @ phi_out @ u[i])

        return prediction
