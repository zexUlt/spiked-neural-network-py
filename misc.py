import numpy as np


def weighted_squared_norm(v: np.array, w: np.array):
    return v.T @ w @ v


def fast_inv(a: np.array):
    mask = np.abs(a - 1e-5) < 0
    a[mask] = 1 / a[mask]
    return a
