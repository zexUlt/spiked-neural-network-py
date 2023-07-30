from ..base import SpikedNN

import numpy as np
from tqdm import tqdm
from activation_functions import Izhikevich, Sigmoid
from projectors import EllipsoidProjector
from gamma_func import MultiplicativeGamma
from misc import fast_inv


class ProjectedSDNN(SpikedNN):
    def __init__(self,
                 state_size: int,
                 u_size: int,
                 k1: np.array,
                 k2: np.array,
                 nn1: int,
                 nn2: int,
                 w1_init: np.array,
                 w2_init: np.array,
                 p_init: np.array,
                 a_init: np.array,
                 l: np.array,
                 p_external: np.array,
                 beta: float,
                 w1_projector: EllipsoidProjector,
                 w2_projector: EllipsoidProjector,
                 sigma_1: Izhikevich | Sigmoid,
                 sigma_2: Izhikevich | Sigmoid,
                 gamma_1: MultiplicativeGamma,
                 gamma_2: MultiplicativeGamma,
                 stop_time: float,
                 sampling_step: float,
                 **kwargs):
        self._state_size = state_size
        self._u_size = u_size

        self.k1 = k1
        self.k2 = k2

        self.nn1 = nn1
        self.nn2 = nn2

        self.W1_init = w1_init
        self.W2_init = w2_init

        self.A_init = a_init
        self.P = p_init
        self.L = l

        self.P_e = p_external
        self.P_i = self.P_e / beta

        self.stop_time = stop_time
        self.sampling_step = sampling_step

        self.w1_proj = w1_projector
        self.w2_proj = w2_projector

        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

        self.history_x = []
        self.history_Wa = []
        self.history_Wb = []
        self.history_dx_dt = []
        self.gamma_1_hist = []
        self.gamma_2_hist = []

        self.history_sigma = []
        self.history_phi_U = []
        self.history_loss = []

    def _law_w1(self, act_result, delta, w):
        h = np.broadcast_to(act_result, (act_result.shape[0], self.P_i.shape[1]))  # H(sigma(x))
        one_over_2gamma = 1. / (2. * self.gamma_1(w))
        return (
                one_over_2gamma *
                fast_inv(
                    np.eye(*self.P_i.shape) +
                    self.gamma_1.param * one_over_2gamma * w.T @ w * self.P_i  # W_tilde = W_1_cur - W_0?
                ) / self.k1 @ h.T @ self.P @ delta
        )

    def _law_w2(self, act_result, delta, w):
        h = np.broadcast_to(act_result, (act_result.shape[0], self.P_e.shape[1]))
        one_over_2gamma = 1. / (2. * self.gamma_2(w))
        return (
                one_over_2gamma *
                fast_inv(
                    np.eye(*self.P_e.shape) +
                    self.gamma_2.param * one_over_2gamma * w.T @ w * self.P_e
                ) / self.k2 @ h.T @ self.P @ delta
        )

    def fit(self, x: np.array, u: np.array, step: float):
        self.history_x = []
        self.history_Wa = []
        self.history_Wb = []
        self.history_loss = []

        x_hat = np.array([-0.05, -0.05]).reshape((self._state_size, 1))

        W1_a = self.W1_init.copy()
        W2_a = self.W2_init.copy()

        self.history_x.append(x_hat)
        self.history_Wa.append(W1_a)
        self.history_Wb.append(W2_a)

        for i in tqdm(range(1, len(x))):
            cur_t = i * self.sampling_step

            delta = x[i] - x_hat
            self.history_loss.append(np.abs(delta))

            sigma_1_result = self.sigma_1(x_hat)
            sigma_2_result = self.sigma_2(x_hat) @ u[i]

            W1_a_vec = W1_a.view().reshape(-1, 1)
            W2_a_vec = W2_a.view().reshape(-1, 1)

            self.gamma_1_hist.append(np.squeeze(self.gamma_1(W1_a_vec)))
            self.gamma_2_hist.append(np.squeeze(self.gamma_2(W2_a_vec)))
            if cur_t <= self.stop_time:
                dW1_dt = self._law_w1(act_result=sigma_1_result, delta=delta, w=W1_a_vec)
                dW2_dt = self._law_w2(act_result=sigma_2_result, delta=delta, w=W2_a_vec)
            else:
                dW1_dt = np.zeros_like(W1_a_vec)
                dW2_dt = np.zeros_like(W2_a_vec)

            W1_a_vec += step * dW1_dt
            W2_a_vec += step * dW2_dt

            if cur_t <= self.stop_time:
                dx_dt = self.A_init @ x_hat + W1_a @ sigma_1_result + W2_a @ sigma_2_result + self.L @ delta
            else:
                dx_dt = self.A_init @ x_hat + W1_a @ sigma_1_result + W2_a @ sigma_2_result
            x_hat += step * dx_dt

            self.history_sigma.append((W1_a @ sigma_1_result).copy())
            self.history_phi_U.append((W2_a @ sigma_2_result).copy())
            self.history_Wa.append(W1_a.copy())
            self.history_Wb.append(W2_a.copy())
            self.history_dx_dt.append(dx_dt.copy())
            self.history_x.append(x_hat.copy())

        return self

    def predict(self, *args, **kwargs): ...
