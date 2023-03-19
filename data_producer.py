from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AbstractProducer(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs): ...

    @abstractmethod
    def get_state_shape(self): ...


class IzhikevichProducer(AbstractProducer):
    def __init__(self,
                 shape: Tuple[int, int],
                 in_scale: float = 1.,
                 out_scale: float = 1.,
                 izh_border: float = 30,
                 param_a: float = 2e-2,
                 param_b: float = 2e-1,
                 param_c: float = -65.,
                 param_d: float = 8.,
                 param_e: float = -65.):
        super(IzhikevichProducer, self).__init__()
        self.izh_border = izh_border
        self.param_a = param_a
        self.param_b = param_b
        self.param_c = param_c
        self.param_d = param_d
        self.param_e = param_e
        self.control = np.ones(shape=shape) * self.param_b * self.param_e
        self.state = np.ones(shape=shape) * self.param_e
        self.in_scale_factor = in_scale
        self.out_scale_factor = out_scale

    def __call__(self, x: np.array = None, step: float = 0.01):
        if x is None:
            x = np.ones(shape=self.state.shape) * 15.

        x *= self.in_scale_factor

        vec_scale = np.ones(shape=self.state.shape)

        self.state += step / 2 * (.04 * self.state ** 2 + 5. * self.state + 140 - self.control + x)
        self.state += step / 2 * (.04 * self.state ** 2 + 5. * self.state + 140 - self.control + x)

        self.control += step * (self.param_a * (self.param_b * self.state - self.control))

        beyond_border = self.state > self.izh_border

        self.state[beyond_border] = vec_scale[beyond_border] * self.param_c
        self.control[beyond_border] = vec_scale[beyond_border] * self.param_d

        return self.out_scale_factor * self.state

    def get_state_shape(self):
        return self.state.shape
