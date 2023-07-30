import numba as nb
from misc import weighted_squared_norm


def generate_gamma(sign: nb.int32, param: nb.float32, weights: nb.float32[:],
                   is_multiplicative: nb.boolean):
    @nb.njit
    def multiplicative(vec: nb.float32[:]):
        return (
            sign * param * (1 - weighted_squared_norm(vec, weights))
        )

    @nb.njit
    def power(vec: nb.float32[:]):
        return (
            (sign * (1 - weighted_squared_norm(vec, weights))) ** param
        )

    return multiplicative if is_multiplicative else power
