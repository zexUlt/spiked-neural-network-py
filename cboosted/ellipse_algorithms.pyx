import numpy as np
from scipy.optimize import minimize

cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def nearest_on_ellipse(np.ndarray[DTYPE_t, ndim=2] point, np.ndarray[:, :] L not None, np.ndarray[:, :] L_inv not None):
    # cdef np.float64_t[:, :] l_view = L
    x_0 = np.zeros_like(point)
    x_0[0] = L_inv.flatten()[0]
    y_0 = L.T @ x_0

    def functional(np.ndarray[DTYPE_t, ndim=2] y):
        return (
            y.T @ L_inv @ L_inv.T @ y / 2. -
            point.T @ y
        )

    result = minimize(functional, y_0, method='Nelder-Mead', tol=1e-6)
    projection = -L_inv.T @ result.x
    return projection