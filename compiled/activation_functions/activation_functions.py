import numba as nb
import numpy as np


# @nb.njit
def generate_sigmoid_call(out: nb.int32, u_size: nb.int32,
                          param_b: nb.float32[:], param_c: nb.float32[:],
                          param_d: nb.float32[:], param_e: nb.float32[:]):
    @nb.njit
    def inner(x: np.array([])):
        s = np.zeros((out, u_size), dtype=np.float32)
        nin = x.shape[0]

        if u_size > 1:
            for j in range(u_size):
                for neurons_out in range(out):
                    z = np.zeros((1,))
                    for neurons_in in range(nin):
                        z = z + x[neurons_in] * param_c[neurons_in, neurons_out, j] + \
                             param_b[neurons_out, j]
                    s[neurons_out, j] = (1 / (1 + np.exp(z))) + \
                                        param_d[neurons_out, j] - \
                                        param_e[neurons_out, j]
        else:
            for neurons_out in range(out):
                z = np.zeros((1,))
                for neurons_in in range(nin):
                    z = z + x[neurons_in] * param_c[neurons_in, neurons_out] + \
                         param_b[neurons_out]
                s[neurons_out] = (1 / (1 + np.exp(z))) + \
                                 param_d[neurons_out] - \
                                 param_e[neurons_out]
        return s
    return inner
