"""
lp_bounds.py
"""

import numpy as np
import numba
from jacobian_bounds import compute_deltas, compute_jac_bounds, compute_omega
from network_ops import forward_pass
from activations import compute_sigma_jac_intervals_all_layers
from flex_model import extract_numpy_params
from hessian_bounds import compute_hessian_bounds


# --- 1. Numba JIT-compiled Helper (Theorem Logic) ---

@numba.jit(nopython=True, fastmath=True)
def _compute_lp_integrand_bounds_jit(low_tensor, high_tensor, p):
    """
    Computes bounds on sum(|x_i|^p).
    Logic derived exactly from Theorem:
      - Upper: sum(max(|low|, |high|)^p)
      - Lower: If low < 0 < high -> 0, else sum(min(|low|, |high|)^p)
    """
    low_flat = low_tensor.flatten()
    high_flat = high_tensor.flatten()

    sum_min_p = 0.0
    sum_max_p = 0.0

    for i in range(low_flat.size):
        l = low_flat[i]
        h = high_flat[i]

        max_val = max(abs(l), abs(h))
        sum_max_p += max_val ** p

        if l < 0 and h > 0:
            current_min_p = 0.0
        else:
            min_val = min(abs(l), abs(h))
            current_min_p = min_val ** p

        sum_min_p += current_min_p

    return sum_min_p, sum_max_p


# --- 2. Functional Bounds (Phi) ---

def compute_phi_lp_bounds(model, x, delta, p=2.0, volume_lambda=None):
    weights, _ = extract_numpy_params(model)
    d_in = weights[0].shape[1]

    if volume_lambda is None:
        volume_lambda = (2.0 * delta) ** d_in

    x_l, z_l = forward_pass(model, x.reshape(-1, 1))
    _, delta_low_bar, delta_up_bar = compute_deltas(model, x_l, delta)

    center_output = x_l[-1]
    phi_low = center_output - delta_low_bar[-1]
    phi_high = center_output + delta_up_bar[-1]

    # Pass float(p) to the JIT compiler to ensure type stability
    sum_min_p, sum_max_p = _compute_lp_integrand_bounds_jit(phi_low, phi_high, float(p))

    # Match Theorem: ( lambda(Omega) * sum(h^p) )^(1/p)
    lower_Lp = (volume_lambda * sum_min_p) ** (1.0 / p)
    upper_Lp = (volume_lambda * sum_max_p) ** (1.0 / p)

    return lower_Lp, upper_Lp, x_l, z_l


# --- 3. Derivative Bounds (Gradient/Jacobian) ---

def compute_grad_lp_bounds(model, x, delta, x_l=None, z_l=None, p=2.0, volume_lambda=None):
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

    jac_bound_up_list, jac_bound_down_list = compute_jac_bounds(weights, alpha_all, beta_all, d_out)
    J_low = jac_bound_down_list[0]
    J_high = jac_bound_up_list[0]

    sum_min_p, sum_max_p = _compute_lp_integrand_bounds_jit(J_low, J_high, float(p))

    lower_Lp = (volume_lambda * sum_min_p) ** (1.0 / p)
    upper_Lp = (volume_lambda * sum_max_p) ** (1.0 / p)

    return lower_Lp, upper_Lp


# --- 4. Hessian Bounds (Second Derivative) ---

def compute_hess_lp_bounds(model, x, delta, p=2.0, volume_lambda=None):
    weights, _ = extract_numpy_params(model)
    d_in = weights[0].shape[1]

    if volume_lambda is None:
        volume_lambda = (2.0 * delta) ** d_in

    # Get Intervals [H_low, H_high] from IBP
    H_low, H_high = compute_hessian_bounds(
        model, x, delta, jacobian_bound_method='ibp', entries_to_compute=None
    )

    # Apply Case Distinction Lemma (Theorem logic)
    sum_min_p, sum_max_p = _compute_lp_integrand_bounds_jit(H_low, H_high, float(p))

    lower_Lp = (volume_lambda * sum_min_p) ** (1.0 / p)
    upper_Lp = (volume_lambda * sum_max_p) ** (1.0 / p)

    return lower_Lp, upper_Lp