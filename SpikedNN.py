from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
from activation_functions import Izhikevich, Sigmoid
from projectors import AbstractProjector
from gamma_func import GammaFunc
from misc import fast_inv


class SpikedNN(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs): ...
    def predict(self, *args, **kwargs): ...


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
                 projector: AbstractProjector,
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


class ProjectorlessSDNN(SpikedNN):
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
                 p_internal: np.array,
                 p_external: np.array,
                 sigma_1: Izhikevich | Sigmoid,
                 sigma_2: Izhikevich | Sigmoid,
                 gamma_1: GammaFunc,
                 gamma_2: GammaFunc,
                 stop_time: float,
                 **kwargs):
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

        self.P_i = p_internal
        self.P_e = p_external

        self.stop_time = stop_time

        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

        self.history_x = []
        self.history_Wa = []
        self.history_Wb = []
        self.history_dx_dt = []

        self.history_sigma = []
        self.history_phi_U = []
        self.history_loss = []

    def _law_Wa(self, act_result, delta, w):
        h = np.broadcast_to(act_result, (act_result.shape[0], self.P_i.shape[1]))
        one_over_2gamma = 1. / (2. * self.gamma_1(w))
        return np.squeeze(
                one_over_2gamma *
                fast_inv(
                    np.eye(*self.P_i.shape) +
                    self.gamma_1.param * one_over_2gamma * w.T @ w * self.P_i
                ) / self.k1 @ h.T @ self.P_init @ delta
        )

    def _law_Wb(self, act_result, delta, w):
        h = np.broadcast_to(act_result, (act_result.shape[0], self.P_e.shape[1]))
        one_over_2gamma = 1. / (2. * self.gamma_2(w))
        return np.squeeze(
                one_over_2gamma *
                fast_inv(
                    np.eye(*self.P_e.shape) +
                    self.gamma_2.param * one_over_2gamma * w.T @ w * self.P_e
                ) / self.k2 @ h.T @ self.P_init @ delta
        )

    def fit(self, x: np.array, u: np.array, step: float):
        self.history_x = []
        self.history_Wa = []
        self.history_Wb = []
        self.history_loss = []

        cur_x = np.array([0., 1.]).reshape((self._state_size, 1))
        cur_Wa = self.Wa_init.copy()
        cur_Wb = self.Wb_init.copy()

        self.history_x.append(cur_x)
        self.history_Wa.append(cur_Wa)
        self.history_Wb.append(cur_Wb)

        for i in tqdm(range(1, len(x))):
            cur_t = i * step

            delta = x[i] - cur_x
            self.history_loss.append(np.abs(delta))

            sigma_1_result = self.sigma_1(cur_x)
            sigma_2_result = self.sigma_2(cur_x) @ u[i]

            self.history_sigma.append(sigma_1_result.copy())
            self.history_phi_U.append(sigma_2_result.copy())

            cur_Wa_vec = np.squeeze(cur_Wa.view().reshape(1, -1))
            cur_Wb_vec = np.squeeze(cur_Wb.view().reshape(1, -1))

            if cur_t <= self.stop_time:
                dWa_dt = self._law_Wa(act_result=sigma_1_result, delta=delta, w=cur_Wa_vec)
                dWb_dt = self._law_Wb(act_result=sigma_2_result, delta=delta, w=cur_Wb_vec)
            else:
                dWa_dt = np.zeros_like(self.Wa_init)
                dWb_dt = np.zeros_like(self.Wb_init)

            cur_Wa_vec += step * dWa_dt
            cur_Wb_vec += step * dWb_dt
            dx_dt = self.A_init @ cur_x + cur_Wa @ sigma_1_result + cur_Wb @ sigma_2_result
            cur_x += step * dx_dt

            self.history_Wa.append(cur_Wa.copy())
            self.history_Wb.append(cur_Wb.copy())
            self.history_dx_dt.append(dx_dt.copy())
            self.history_x.append(cur_x.copy())

        return self

    def predict(self, *args, **kwargs): ...
