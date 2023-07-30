import numpy as np
cimport numpy as np
cimport cython


ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def weighted_squared_norm(np.ndarray[DTYPE_t, ndim=2] vec, np.ndarray[DTYPE_t, ndim=2] w):
    return vec.T @ w @ vec

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_inv(np.ndarray[DTYPE_t, ndim=2] a):
    mask = np.abs(a - 1e-6) < 0
    a[mask] = 1 / a[mask]
    return a