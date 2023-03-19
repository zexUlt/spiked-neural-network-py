import numpy as np
from abc import ABC, abstractmethod
from misc import weighted_squared_norm


class GammaFunc(ABC):
    def __init__(self, param: float, is_inner: bool, weight_matrix: np.array):
        self.param = param
        self.sign = 1 if is_inner else -1
        self.weights = weight_matrix

    @abstractmethod
    def __call__(self, vec): ...


class MultiplicativeGamma(GammaFunc):
    def __init__(self, param: float, is_inner: bool, weight_matrix: np.array):
        super(MultiplicativeGamma, self).__init__(param, is_inner, weight_matrix)

    def __call__(self, vec):
        return (
                self.sign * self.param * (1 - weighted_squared_norm(vec, self.weights))
        )


class PowerGamma(GammaFunc):
    def __init__(self, param: float, is_inner: bool, weight_matrix: np.array):
        super(PowerGamma, self).__init__(param, is_inner, weight_matrix)

    def __call__(self, vec):
        return (
                self.sign * (1 - weighted_squared_norm(vec, self.weights)) ** self.param
        )
