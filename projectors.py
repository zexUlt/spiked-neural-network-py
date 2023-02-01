from abc import ABC, abstractmethod

import numpy as np


class AbstractProjector(ABC):
    @abstractmethod
    def __init__(self, left_border: np.array, right_border: np.array):
        self.left_border: np.array = left_border
        self.right_border: np.array = right_border

    @abstractmethod
    def __call__(self, x: np.array) -> np.array:
        raise NotImplementedError


class NullProjector(AbstractProjector):
    def __init__(self, left_border: np.array, right_border: np.array):
        super().__init__(left_border, right_border)

    def __call__(self, x: np.array) -> np.array:
        return x


class SaturatedProjector(AbstractProjector):
    def __init__(self, left_border: np.array, right_border: np.array):
        self.left_border = left_border
        self.right_border = right_border

    def __call__(self, x: np.array) -> np.array:
        lesser_left = x < self.left_border
        greater_right = x > self.right_border

        x[lesser_left] = self.left_border[lesser_left]
        x[greater_right] = self.right_border[greater_right]

        return x
