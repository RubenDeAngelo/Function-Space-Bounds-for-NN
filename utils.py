import numpy as np
import numba # <-- 1. Import numba

@numba.jit(nopython=True, fastmath=True)
def W_plus(W):
    """Returns the element-wise positive part of a matrix."""
    return np.maximum(0, W)

@numba.jit(nopython=True, fastmath=True)
def W_minus(W):
    """Returns the element-wise negative part of a matrix."""
    return np.minimum(0, W)


@numba.jit(nopython=True, fastmath=True)  # Add fastmath
def interval_mul_intervals(a_low, a_high, b_low, b_high):
    """
    Computes the interval product [a_low, a_high] * [b_low, b_high].
    """
    p1 = a_low * b_low
    p2 = a_low * b_high
    p3 = a_high * b_low
    p4 = a_high * b_high

    # --- (THE FIX) ---
    # Use direct scalar comparison, which Numba handles correctly.
    lo = min(p1, p2, p3, p4)
    hi = max(p1, p2, p3, p4)
    # --- (END FIX) ---
    return lo, hi

@numba.jit(nopython=True, fastmath=True) # Add fastmath
def scale_interval_by_scalar(scalar, low, high):
    """
    Computes scalar * [low, high].
    """
    if scalar >= 0:
        return scalar * low, scalar * high
    else:
        return scalar * high, scalar * low