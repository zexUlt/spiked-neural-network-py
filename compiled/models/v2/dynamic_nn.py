import numba as nb
import numpy as np
from misc import fast_inv


def generate_law_1(P_i, gamma_1, P_init, k1):
    @nb.njit
    def inner(act_result: nb.float32[:], delta: nb.float32[:], w: nb.float32[:]):
        h = np.broadcast_to(act_result, (act_result.shape[0], P_i.shape[1]))
        one_over_2gamma = 1. / (2. * gamma_1(w))
        return (
                one_over_2gamma *
                fast_inv(
                    np.eye(*P_i.shape) +
                    gamma_1.param * one_over_2gamma * w.T @ w * P_i
                ) / k1 @ h.T @ P_init @ delta
        )

    return inner


def generate_law_2(P_e, gamma_2, P_init, k2):
    @nb.njit
    def inner(act_result: nb.float32[:], delta: nb.float32[:], w: nb.float32[:]):
        h = np.broadcast_to(act_result, (act_result.shape[0], P_e.shape[1]))
        one_over_2gamma = 1. / (2. * gamma_2(w))
        return (
                one_over_2gamma *
                fast_inv(
                    np.eye(*P_e.shape) +
                    gamma_2.param * one_over_2gamma * w.T @ w * P_e
                ) / k2 @ h.T @ P_init @ delta
        )

    return inner
