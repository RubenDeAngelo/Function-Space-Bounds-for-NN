"""
Lp norm enclosures for network output (phi), Jacobian (grad), and Hessian (hess).
Implements the integrand bounding step used in Corollaries 4.22 and 4.27.
"""

import numpy as np
from lib_jacobian_enclosures import (
    compute_deltas,
    compute_jac_bounds,
    compute_recurjac_bounds,
    compute_omega
)
from lib_nn_interval_ops import forward_pass
from lib_activation_enclosures import compute_sigma_jac_intervals_all_layers
from lib_nn_model import extract_numpy_params
from lib_hessian_enclosures import compute_hessian_bounds


def _compute_lp_integrand_bounds_vectorized(low_tensor, high_tensor, p):
    """
    Returns (lower_sum, upper_sum) of sum |f_i|^p over the interval [low, high].
    Handles zero-crossing entries (lower bound = 0) via vectorized masking.
    """
    abs_low = np.abs(low_tensor)
    abs_high = np.abs(high_tensor)

    max_vals = np.maximum(abs_low, abs_high)
    sum_max_p = np.sum(max_vals ** p)

    crosses_zero = (low_tensor < 0) & (high_tensor > 0)
    min_vals = np.where(crosses_zero, 0.0, np.minimum(abs_low, abs_high))
    sum_min_p = np.sum(min_vals ** p)

    return float(sum_min_p), float(sum_max_p)


def compute_phi_lp_bounds(model, x, delta, p=2.0, volume_lambda=None):
    weights, _ = extract_numpy_params(model)
    d_in = weights[0].shape[1]

    if volume_lambda is None:
        volume_lambda = (2.0 * delta) ** d_in

    x_l, z_l = forward_pass(model, x.reshape(-1, 1))
    _, delta_low_bar, delta_up_bar = compute_deltas(model, x_l, delta)

    # Reshape to (-1,1) to prevent broadcasting (d_out,1) with (d_out,) into (d_out,d_out)
    center_output = x_l[-1].reshape(-1, 1)
    d_low = delta_low_bar[-1].reshape(-1, 1)
    d_up = delta_up_bar[-1].reshape(-1, 1)

    phi_low = center_output - d_low
    phi_high = center_output + d_up

    sum_min_p, sum_max_p = _compute_lp_integrand_bounds_vectorized(phi_low, phi_high, float(p))

    lower_Lp = (volume_lambda * sum_min_p) ** (1.0 / p)
    upper_Lp = (volume_lambda * sum_max_p) ** (1.0 / p)

    return lower_Lp, upper_Lp, x_l, z_l


def compute_grad_lp_bounds(model, x, delta, x_l=None, z_l=None, p=2.0, volume_lambda=None, jacobian_method='recurjac'):
    weights, _ = extract_numpy_params(model)
    d_out = weights[-1].shape[0]
    d_in = weights[0].shape[1]

    if volume_lambda is None:
        volume_lambda = (2.0 * delta) ** d_in
    if x_l is None or z_l is None:
        x_l, z_l = forward_pass(model, x.reshape(-1, 1))

    delta_prime_jac, _, _ = compute_deltas(model, x_l, delta)
    omega_all = compute_omega(len(weights), weights, delta_prime_jac)
    alpha_all, beta_all = compute_sigma_jac_intervals_all_layers(model, z_l, omega_all)

    if jacobian_method == "recurjac":
        jac_bound_up_list, jac_bound_down_list = compute_recurjac_bounds(weights, alpha_all, beta_all, d_out)
    else:
        jac_bound_up_list, jac_bound_down_list = compute_jac_bounds(weights, alpha_all, beta_all, d_out)

    J_low = jac_bound_down_list[0]
    J_high = jac_bound_up_list[0]

    sum_min_p, sum_max_p = _compute_lp_integrand_bounds_vectorized(J_low, J_high, float(p))

    lower_Lp = (volume_lambda * sum_min_p) ** (1.0 / p)
    upper_Lp = (volume_lambda * sum_max_p) ** (1.0 / p)

    return lower_Lp, upper_Lp


def compute_hess_lp_bounds(model, x, delta, p=2.0, volume_lambda=None, jacobian_method='recurjac'):
    weights, _ = extract_numpy_params(model)
    d_in = weights[0].shape[1]

    if volume_lambda is None:
        volume_lambda = (2.0 * delta) ** d_in

    H_low, H_high = compute_hessian_bounds(
        model, x, delta, jacobian_bound_method=jacobian_method, entries_to_compute=None
    )

    sum_min_p, sum_max_p = _compute_lp_integrand_bounds_vectorized(H_low, H_high, float(p))

    lower_Lp = (volume_lambda * sum_min_p) ** (1.0 / p)
    upper_Lp = (volume_lambda * sum_max_p) ** (1.0 / p)

    return lower_Lp, upper_Lp
