from abc import ABC, abstractmethod
import numpy as np
from algorithms import nearest_on_ellipse, nearest_on_sphere
from misc import weighted_squared_norm


class Projector(ABC):
    def __init__(self, left_border: np.array, right_border: np.array):
        self.left_border: np.array = left_border
        self.right_border: np.array = right_border

    @abstractmethod
    def __call__(self, x: np.array, in_place=True) -> np.array:
        raise NotImplementedError


class NullProjector(Projector):
    def __init__(self):
        super(NullProjector, self).__init__(None, None)

    def __call__(self, x: np.array, in_place=True) -> np.array:
        return x


class SaturatedProjector(Projector):
    def __init__(self, left_border: np.array, right_border: np.array):
        super(SaturatedProjector, self).__init__(left_border, right_border)
        self.left_border = left_border
        self.right_border = right_border

    def __call__(self, x: np.array, in_place=True) -> np.array:
        if in_place:
            lesser_left = x < self.left_border
            greater_right = x > self.right_border

            x[lesser_left] = self.left_border[lesser_left]
            x[greater_right] = self.right_border[greater_right]

            return None
        else:
            x_ = x.copy()
            lesser_left = x_ < self.left_border
            greater_right = x_ > self.right_border

            x_[lesser_left] = self.left_border[lesser_left]
            x_[greater_right] = self.right_border[greater_right]

            return x_


class EllipsoidProjector:
    def __init__(self, ellipsoid_form: np.array):
        self.L = np.linalg.cholesky(ellipsoid_form)
        self.is_sphere = np.all(np.diagonal(self.L) == self.L[0, 0])
        self.L_inv = np.linalg.inv(self.L)
        self.last_closest_point = np.zeros((self.L.shape[0], 1))
        self.last_closest_point[0] = self.L_inv.flatten()[0]

    def __call__(self, point: np.array, project_in: bool, in_place: bool = True) -> np.array:
        norm = weighted_squared_norm(point, self.L @ self.L.T)

        if in_place:
            if (project_in and norm > 1) or (not project_in and norm < 1):
                if self.is_sphere:
                    point[:] = nearest_on_sphere(point, self.L_inv[0, 0])
                else:
                    self.last_closest_point = nearest_on_ellipse(
                        point, self.L, self.L_inv, self.last_closest_point
                    ).reshape((-1, 1))
                    point[:] = self.last_closest_point.copy()
        else:
            if (project_in and norm > 1) or (not project_in and norm < 1):
                if self.is_sphere:
                    return nearest_on_sphere(point, self.L_inv[0, 0])
                else:
                    self.last_closest_point = nearest_on_ellipse(
                        point, self.L, self.L_inv, self.last_closest_point
                        ).reshape((-1, 1))
                    return self.last_closest_point.copy()
            else:
                return point
