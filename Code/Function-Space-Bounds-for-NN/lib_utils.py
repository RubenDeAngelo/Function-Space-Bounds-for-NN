"""
Numba-accelerated scalar interval arithmetic helpers used by the Hessian bounding code.
"""

import numpy as np
import numba


@numba.jit(nopython=True, fastmath=True)
def W_plus(W):
    """Returns the element-wise positive part of a matrix."""
    return np.maximum(0, W)


@numba.jit(nopython=True, fastmath=True)
def W_minus(W):
    """Returns the element-wise negative part of a matrix."""
    return np.minimum(0, W)


@numba.jit(nopython=True, fastmath=True)
def interval_mul_intervals(a_low, a_high, b_low, b_high):
    """Interval multiplication [a_low, a_high] * [b_low, b_high]."""
    p1 = a_low * b_low
    p2 = a_low * b_high
    p3 = a_high * b_low
    p4 = a_high * b_high
    return min(p1, p2, p3, p4), max(p1, p2, p3, p4)


@numba.jit(nopython=True, fastmath=True)
def scale_interval_by_scalar(scalar, low, high):
    """Scalar multiplication scalar * [low, high]."""
    if scalar >= 0:
        return scalar * low, scalar * high
    else:
        return scalar * high, scalar * low
