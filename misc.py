import numpy as np


def weighted_squared_norm(v: np.array, w: np.array) -> float:
    return v.T @ w @ v


def fast_inv(a):
    mask = np.abs(a - 1e-5) < 0
    a[mask] = 1 / a[mask]
    return a
