"""
jacobian_bounds.py

Implements rigorous upper and lower bounds for the Jacobian of a neural network
using Interval Bound Propagation (IBP).

OPTIMIZATION:
- Vectorized the interval multiplication loop to remove redundant calculations.
"""

import numpy as np
import numba
from flex_model import extract_numpy_params
from activations import (
    sigma, compute_sigma_jac_intervals_all_layers
)
from network_ops import forward_pass, backward_pass, compute_jacobian
from utils import W_plus, W_minus, interval_mul_intervals, scale_interval_by_scalar


def compute_deltas(model, x_l, delta):
    weights, biases = extract_numpy_params(model)
    L = len(weights) - 1
    x_l = [np.atleast_1d(np.squeeze(x)) for x in x_l]
    biases = [np.atleast_1d(np.squeeze(b)) for b in biases]
    delta_up_bar = [delta * np.ones_like(x_l[0])]
    delta_low_bar = [delta * np.ones_like(x_l[0])]
    for l in range(L):
        w = weights[l]
        w_plus = W_plus(w)
        w_minus = W_minus(w)
        w_times_x = w @ x_l[l]
        x_next = x_l[l + 1]
        upper = sigma(model, w_times_x + w_plus @ delta_up_bar[l] - w_minus @ delta_low_bar[l] + biases[l]) - x_next
        lower = x_next - sigma(model, w_times_x - w_plus @ delta_low_bar[l] + w_minus @ delta_up_bar[l] + biases[l])
        delta_up_bar.append(upper)
        delta_low_bar.append(lower)
    W_L = weights[L]
    W_L_plus = W_plus(W_L)
    W_L_minus = W_minus(W_L)

    delta_up_bar.append(W_L_plus @ delta_up_bar[L] - W_L_minus @ delta_low_bar[L])
    delta_low_bar.append(W_L_plus @ delta_low_bar[L] - W_L_minus @ delta_up_bar[L])

    delta_prime = []
    for l in range(len(delta_up_bar)):
        d_ell = delta_up_bar[l].shape[0]
        delta_prime_ell = [
            max(np.abs(delta_up_bar[l][i]), np.abs(delta_low_bar[l][i]))
            for i in range(d_ell)
        ]
        delta_prime.append(np.array(delta_prime_ell))
    return delta_prime, delta_low_bar, delta_up_bar


def compute_omega(L, weights, delta_prime_jac):
    omega_all = []
    for l in range(L):
        W = weights[l]
        delta_prime_ell = delta_prime_jac[l]
        omega = []
        for k in range(W.shape[0]):
            omega_k = np.abs(W[k]) @ delta_prime_ell
            omega.append(omega_k)
        omega_all.append(np.array(omega))
    return omega_all


# --- OPTIMIZED JIT FUNCTION ---
@numba.jit(nopython=True, fastmath=True)
def compute_jac_bounds(weights, alpha_all, beta_all, d_out):
    """
    Propagates the Jacobian bounds backward from the output layer.
    """
    L = len(weights)

    jac_bound_up = [np.empty((1, 1), dtype=np.float64) for _ in range(L)]
    jac_bound_down = [np.empty((1, 1), dtype=np.float64) for _ in range(L)]

    W_L_minus_1 = weights[L - 1]
    jac_bound_down[L - 1] = W_L_minus_1.copy().astype(np.float64)
    jac_bound_up[L - 1] = W_L_minus_1.copy().astype(np.float64)

    # Iterate from L-2 down to 0
    for l in range(L - 2, -1, -1):
        W = weights[l]
        d_in = W.shape[1]
        d_layer = W.shape[0]

        J_next_low = jac_bound_down[l + 1]
        J_next_high = jac_bound_up[l + 1]
        alpha_vec = alpha_all[l]
        beta_vec = beta_all[l]

        J_low = np.zeros((d_out, d_in), dtype=np.float64)
        J_high = np.zeros((d_out, d_in), dtype=np.float64)

        for p in range(d_out):
            # Optimization: Pre-compute the scaled J_next interval for this output p
            # This avoids re-multiplying by sigma inside the inner q loop
            # J_scaled shape: (d_layer,)
            J_scaled_lo = np.zeros(d_layer)
            J_scaled_hi = np.zeros(d_layer)

            for k in range(d_layer):
                j_lo, j_hi = J_next_low[p, k], J_next_high[p, k]
                s_lo, s_hi = alpha_vec[k], beta_vec[k]

                # Inline interval mult
                p1 = j_lo * s_lo
                p2 = j_lo * s_hi
                p3 = j_hi * s_lo
                p4 = j_hi * s_hi
                J_scaled_lo[k] = min(p1, p2, p3, p4)
                J_scaled_hi[k] = max(p1, p2, p3, p4)

            # Now project with Weights
            for q in range(d_in):
                sum_lo = 0.0
                sum_hi = 0.0

                for k in range(d_layer):
                    w_val = float(W[k, q])
                    # Scalar * Interval
                    if w_val >= 0:
                        sum_lo += w_val * J_scaled_lo[k]
                        sum_hi += w_val * J_scaled_hi[k]
                    else:
                        sum_lo += w_val * J_scaled_hi[k]
                        sum_hi += w_val * J_scaled_lo[k]

                J_low[p, q] = sum_lo
                J_high[p, q] = sum_hi

        jac_bound_down[l] = J_low
        jac_bound_up[l] = J_high

    return jac_bound_up, jac_bound_down


def compute_jacobian_and_bounds(model, x, delta):
    weights, _ = extract_numpy_params(model)
    L = len(weights)
    d_out = weights[-1].shape[0]

    x = x.reshape(-1, 1)

    x_l, z_l = forward_pass(model, x)
    delta_prime_jac, _, _ = compute_deltas(model, x_l, delta)
    omega_all = compute_omega(L, weights, delta_prime_jac)
    alpha_all, beta_all = compute_sigma_jac_intervals_all_layers(
        model, z_l, omega_all
    )

    jac_bound_up_list, jac_bound_down_list = compute_jac_bounds(
        weights, alpha_all, beta_all, d_out
    )

    J_low = jac_bound_down_list[0]
    J_high = jac_bound_up_list[0]

    return  J_low, J_high