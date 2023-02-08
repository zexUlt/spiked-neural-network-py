import numpy as np
from typing import Tuple


class Izhikevich:
    def __init__(self,
                 shape: Tuple[int, int],
                 in_scale: float = 80.,
                 out_scale: float = 1 / 60.,
                 izh_border: float = 30,
                 param_a: float = 2e-2,
                 param_b: float = 2e-1,
                 param_c: float = -65.,
                 param_d: float = 8.,
                 param_e: float = -65):
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

    def __call__(self, x: np.array, step: float = 0.01):
        x *= self.in_scale_factor

        vec_scale = np.ones(shape=self.state.shape)

        self.state += step / 2 * (.04 * self.state * self.state + 5. * self.state + 140 - self.control + x)
        self.state += step / 2 * (.04 * self.state * self.state + 5. * self.state + 140 - self.control + x)

        self.control += step * (self.param_a * (self.param_b * self.state - self.control))

        beyond_border = self.state > self.izh_border

        self.state[beyond_border] = vec_scale[beyond_border] * self.param_c
        self.control[beyond_border] = vec_scale[beyond_border] * self.param_d

        return self.state * self.out_scale_factor


class Sigmoid:
    def __init__(self,
                 out: int,
                 u_size: int,
                 param_b: np.array,
                 param_c: np.array,
                 param_d: np.array,
                 param_e: np.array):
        self.out = out
        self.u_size = u_size
        self.param_b = param_b
        self.param_c = param_c
        self.param_d = param_d
        self.param_e = param_e

    def __call__(self, x: np.array):
        s = np.zeros((self.out, self.u_size))
        nin = x.shape[0]

        if self.u_size > 1:
            for j in range(self.u_size):
                for neurons_out in range(self.out):
                    z = 0
                    for neurons_in in range(nin):
                        z += x[neurons_in] * self.param_c[neurons_in, neurons_out, j] + \
                             self.param_b[neurons_out, j]
                    s[neurons_out, j] = (1 / (1 + np.exp(z))) + \
                                        self.param_d[neurons_out, j] - \
                                        self.param_e[neurons_out, j]
        else:
            for neurons_out in range(self.out):
                z = 0
                for neurons_in in range(nin):
                    z += x[neurons_in] * self.param_c[neurons_in, neurons_out] + \
                         self.param_b[neurons_out]
                s[neurons_out] = (1 / (1 + np.exp(z))) + \
                                 self.param_d[neurons_out] - \
                                 self.param_e[neurons_out]
        return s
