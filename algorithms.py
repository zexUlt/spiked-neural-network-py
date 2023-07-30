import numpy as np
from scipy.optimize import minimize


def nearest_on_ellipse(point: np.array, L: np.array, L_inv: np.array, approximation: np.array, shift=0):
    """
        Algorithm was taken from https://tcg.mae.cornell.edu/pubs/Pope_FDA_08.pdf
    """
    x_0 = approximation.copy()
    y_0 = L.T @ (x_0 - shift)

    def functional(y: np.array):
        return (
            y.T @ L_inv @ L_inv.T @ y / 2. +
            (shift - point).T @ y
        )

    result = minimize(functional, y_0, method='L-BFGS-B', tol=1e-6)
    projection = shift - L_inv.T @ result.x
    return projection


def nearest_on_sphere(point: np.array, sphere_radius: float) -> np.array:
    return sphere_radius * point / np.linalg.norm(point)

