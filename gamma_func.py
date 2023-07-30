import numpy as np
from misc import weighted_squared_norm


class MultiplicativeGamma:
    def __init__(self, param: float, is_inner: bool, weight_matrix: np.array):
        self.param = param
        self.sign = 1 if is_inner else -1
        self.weights = weight_matrix

    def __call__(self, vec):
        wsn = weighted_squared_norm(vec, self.weights)
        if wsn > 0.95 and self.sign == 1:
            wsn = 0.95
        if wsn < 1.05 and self.sign == -1:
            wsn = 1.05
        return (
                self.sign * self.param * (1 - wsn)
        )


class PowerGamma:
    def __init__(self, param: float, is_inner: bool, weight_matrix: np.array):
        self.param = param
        self.sign = 1 if is_inner else -1
        self.weights = weight_matrix

    def __call__(self, vec):
        return (
                (self.sign * (1 - weighted_squared_norm(vec, self.weights))) ** self.param
        )
