"""
hessian_bounds.py

Implements rigorous upper and lower bounds for the Hessian of a neural network
using Interval Bound Propagation (IBP).

OPTIMIZATIONS:
- Decomposed the O(N^4) backpropagation term into two O(N^3) matrix multiplications.
- Reduced memory overhead by processing per-output-dimension.
"""

import numpy as np
import numba
from flex_model import extract_numpy_params
from jacobian_bounds import (
    compute_deltas, compute_omega,
    compute_jac_bounds as compute_jacobian_ibp_bounds
)
from activations import (
    compute_sigma_jac_intervals_all_layers,
    compute_sigma_hess_intervals_all_layers,
    sigma_derivative,
    sigma_second_derivative
)
from network_ops import forward_pass, backward_pass


# --- Helper for Interval Arithm ---
@numba.jit(nopython=True, fastmath=True, inline='always')
def _interval_mul(a_lo, a_hi, b_lo, b_hi):
    """Interval multiplication."""
    p1 = a_lo * b_lo
    p2 = a_lo * b_hi
    p3 = a_hi * b_lo
    p4 = a_hi * b_hi
    return min(p1, p2, p3, p4), max(p1, p2, p3, p4)

@numba.jit(nopython=True, fastmath=True, inline='always')
def _scalar_mul_interval(scalar, lo, hi):
    """Scalar * Interval multiplication."""
    if scalar >= 0:
        return scalar * lo, scalar * hi
    else:
        return scalar * hi, scalar * lo


# --- 1. JIT-compiled Core Loops (Optimized) ---

@numba.jit(nopython=True, fastmath=True)
def _jit_hessian_loop_full(d_out, d_l, d_lp1, H_next_low, H_next_high,
                           alpha_vec, beta_vec, W, J_next_low, J_next_high,
                           gamma_vec, eta_vec):
    """
    Optimized core loop for the FULL Hessian matrix.
    Decomposes the W^T * H * W operation to reduce complexity from O(N^4) to O(N^3).
    """
    H_low = np.zeros((d_out, d_l, d_l))
    H_high = np.zeros((d_out, d_l, d_l))

    # Iterate over each output dimension k
    for k in range(d_out):

        # --- TERM 1: Backpropagated Curvature (W^T * (H * sigma_prod) * W) ---

        # 1. Compute Intermediate Matrix M = (H * sigma_prod) @ W
        # M has shape (d_lp1, d_l)
        # This reduces the complexity significantly.
        M_low = np.zeros((d_lp1, d_l))
        M_high = np.zeros((d_lp1, d_l))

        for p in range(d_lp1):
            # Pre-load Hessian row and sigma intervals for p
            alpha_p, beta_p = alpha_vec[p], beta_vec[p]

            for q in range(d_lp1):
                # Calculate Sigma Product Interval for (p, q)
                # sigma_prod = [alpha_p, beta_p] * [alpha_q, beta_q]
                alpha_q, beta_q = alpha_vec[q], beta_vec[q]
                if p == q:
                    # We are squaring the interval [alpha_p, beta_p]^2
                    if alpha_p < 0 and beta_p > 0:
                        s_lo = 0.0
                        s_hi = max(alpha_p ** 2, beta_p ** 2)
                    else:
                        s_lo = min(alpha_p ** 2, beta_p ** 2)
                        s_hi = max(alpha_p ** 2, beta_p ** 2)
                else:
                    # Independent interval multiplication
                    s_lo, s_hi = _interval_mul(alpha_p, beta_p, alpha_q, beta_q)

                # Calculate V = H_next[k, p, q] * sigma_prod
                h_lo, h_hi = H_next_low[k, p, q], H_next_high[k, p, q]
                v_lo, v_hi = _interval_mul(h_lo, h_hi, s_lo, s_hi)

                # Row multiply: M[p, j] += V * W[q, j]
                for j in range(d_l):
                    w_val = W[q, j]
                    # term = v * w_val
                    t_lo, t_hi = _scalar_mul_interval(w_val, v_lo, v_hi)

                    M_low[p, j] += t_lo
                    M_high[p, j] += t_hi

        # 2. Compute Final Term 1: H = W^T @ M
        # H[i, j] += sum_p W[p, i] * M[p, j]
        for i in range(d_l):
            for p in range(d_lp1):
                w_val = W[p, i]
                m_lo, m_hi = M_low[p, :], M_high[p, :]

                # Vectorized update over j
                if w_val >= 0:
                    H_low[k, i, :] += w_val * m_lo
                    H_high[k, i, :] += w_val * m_hi
                else:
                    H_low[k, i, :] += w_val * m_hi
                    H_high[k, i, :] += w_val * m_lo

        # --- TERM 2: Local Curvature (W^T * diag(J * sigma'') * W) ---

        # Precompute the diagonal vector D = J * [gamma, eta]
        # D_lo, D_hi are shape (d_lp1,)
        D_low = np.zeros(d_lp1)
        D_high = np.zeros(d_lp1)

        for p in range(d_lp1):
            j_lo, j_hi = J_next_low[k, p], J_next_high[k, p]
            g_lo, g_hi = gamma_vec[p], eta_vec[p]
            D_low[p], D_high[p] = _interval_mul(j_lo, j_hi, g_lo, g_hi)

        # Compute W^T * D * W
        # H[i, j] += sum_p (D[p] * W[p, i] * W[p, j])
        for i in range(d_l):
            for j in range(d_l):
                t2_min = 0.0
                t2_max = 0.0

                for p in range(d_lp1):
                    # Scalar factor = W[p, i] * W[p, j]
                    w_prod = W[p, i] * W[p, j]

                    t_lo, t_hi = _scalar_mul_interval(w_prod, D_low[p], D_high[p])
                    t2_min += t_lo
                    t2_max += t_hi

                H_low[k, i, j] += t2_min
                H_high[k, i, j] += t2_max

    return H_low, H_high


@numba.jit(nopython=True, fastmath=True)
def _jit_hessian_loop_sparse(d_out, d_l, d_lp1, H_next_low, H_next_high,
                             alpha_vec, beta_vec, W, J_next_low, J_next_high,
                             gamma_vec, eta_vec, entries_to_compute):
    """
    Numba-compiled core loop for SPARSE Hessian entries.
    Kept as explicit loops because decomposition is inefficient for sparse entries.
    """
    H_low = np.zeros((d_out, d_l, d_l))
    H_high = np.zeros((d_out, d_l, d_l))

    for k in range(d_out):
        for entry_idx in range(entries_to_compute.shape[0]):
            i, j = entries_to_compute[entry_idx]

            if i >= d_l or j >= d_l: continue

            t1_min, t1_max = 0.0, 0.0
            t2_min, t2_max = 0.0, 0.0

            # Term 1: Backprop Curvature
            for p in range(d_lp1):
                alpha_p, beta_p = alpha_vec[p], beta_vec[p]

                for q in range(d_lp1):
                    # Sigma product
                    alpha_q, beta_q = alpha_vec[q], beta_vec[q]
                    s_lo, s_hi = _interval_mul(alpha_p, beta_p, alpha_q, beta_q)

                    # H * Sigma
                    h_lo, h_hi = H_next_low[k, p, q], H_next_high[k, p, q]
                    v_lo, v_hi = _interval_mul(h_lo, h_hi, s_lo, s_hi)

                    # Weight product
                    w_scalar = W[p, i] * W[q, j]
                    term_lo, term_hi = _scalar_mul_interval(w_scalar, v_lo, v_hi)

                    t1_min += term_lo
                    t1_max += term_hi

                # Term 2: Local Curvature (inside p loop)
                j_lo, j_hi = J_next_low[k, p], J_next_high[k, p]
                g_lo, g_hi = gamma_vec[p], eta_vec[p]

                v_lo_J, v_hi_J = _interval_mul(j_lo, j_hi, g_lo, g_hi)

                w_scalar_J = W[p, i] * W[p, j]
                term_lo_J, term_hi_J = _scalar_mul_interval(w_scalar_J, v_lo_J, v_hi_J)

                t2_min += term_lo_J
                t2_max += term_hi_J

            H_low[k, i, j] = t1_min + t2_min
            H_high[k, i, j] = t1_max + t2_max

    return H_low, H_high


def compute_entrywise_hessian_bounds_layer(W, H_next_low, H_next_high, J_next_low, J_next_high,
                                           alpha_vec, beta_vec, gamma_vec, eta_vec,
                                           entries_to_compute=None):
    d_lp1, d_l = W.shape
    d_out = H_next_low.shape[0]

    # Ensure inputs are float64 for stability
    W = W.astype(np.float64)

    args = (d_out, d_l, d_lp1, H_next_low, H_next_high,
            alpha_vec, beta_vec, W, J_next_low, J_next_high,
            gamma_vec, eta_vec)

    if entries_to_compute is None:
        H_low, H_high = _jit_hessian_loop_full(*args)
    else:
        entries_arr = np.array(entries_to_compute, dtype=np.int64)
        H_low, H_high = _jit_hessian_loop_sparse(*args, entries_arr)

    return H_low, H_high


def compute_jacobian_bounds_for_hessian(model, weights, z_l, J_all, omega_all, d_out,
                                        alpha_all, beta_all, method='ibp'):
    if method == 'ibp':
        jac_bound_up, jac_bound_down = compute_jacobian_ibp_bounds(
            weights, alpha_all, beta_all, d_out
        )
        return jac_bound_up, jac_bound_down
    else:
        raise ValueError("Only 'ibp' method is supported in this fix.")


def compute_hessian_bounds_only(model, x, delta, weights=None, x_l=None, z_l=None,
                                delta_prime_jac=None, omega_all=None, J_all=None,
                                alpha_all=None, beta_all=None, gamma_all=None, eta_all=None,
                                jac_bound_up=None, jac_bound_down=None,
                                H_low=None, H_high=None,
                                jacobian_bound_method='ibp',
                                entries_to_compute=None):
    if weights is None: weights, _ = extract_numpy_params(model)
    L, d_out = len(weights), weights[-1].shape[0]
    if x_l is None or z_l is None:
        x_l, z_l = forward_pass(model, x.reshape(-1, 1))
    if delta_prime_jac is None or omega_all is None:
        delta_prime_jac, _, _ = compute_deltas(model, x_l, delta)
        omega_all = compute_omega(L, weights, delta_prime_jac)
    if J_all is None:
        J_all = backward_pass(model, z_l)

    if alpha_all is None or beta_all is None:
        alpha_all, beta_all = compute_sigma_jac_intervals_all_layers(model, z_l, omega_all)
    if gamma_all is None or eta_all is None:
        gamma_all, eta_all = compute_sigma_hess_intervals_all_layers(model, z_l, omega_all)

    if jac_bound_up is None or jac_bound_down is None:
        jac_bound_up, jac_bound_down = compute_jacobian_bounds_for_hessian(
            model, weights, z_l, J_all, omega_all, d_out,
            alpha_all=alpha_all, beta_all=beta_all,
            method=jacobian_bound_method
        )

    H_low_full = H_low if H_low is not None else [None] * (L + 1)
    H_high_full = H_high if H_high is not None else [None] * (L + 1)

    if H_low_full[L] is None:
        H_low_full[L] = H_high_full[L] = np.zeros((d_out, d_out, d_out))

    for l in range(L - 1, -1, -1):
        if l == L - 1:
            J_next_low = weights[L - 1]
            J_next_high = weights[L - 1]
            alpha_l = np.ones_like(alpha_all[l])
            beta_l = np.ones_like(beta_all[l])
            gamma_l = np.zeros_like(gamma_all[l])
            eta_l = np.zeros_like(eta_all[l])
        else:
            J_next_low = jac_bound_down[l + 1]
            J_next_high = jac_bound_up[l + 1]
            alpha_l = alpha_all[l]
            beta_l = beta_all[l]
            gamma_l = gamma_all[l]
            eta_l = eta_all[l]

        current_layer_entries = None
        if l == 0 and entries_to_compute is not None:
            current_layer_entries = entries_to_compute

        Hl, Hh = compute_entrywise_hessian_bounds_layer(
            W=weights[l],
            H_next_low=H_low_full[l + 1],
            H_next_high=H_high_full[l + 1],
            J_next_low=J_next_low, J_next_high=J_next_high,
            alpha_vec=alpha_l, beta_vec=beta_l,
            gamma_vec=gamma_l, eta_vec=eta_l,
            entries_to_compute=current_layer_entries
        )

        H_low_full[l], H_high_full[l] = Hl, Hh

    return H_low_full[0], H_high_full[0]


def compute_hessian_bounds(model, x, delta,
                           jacobian_bound_method='ibp',
                           entries_to_compute=None):
    weights, _ = extract_numpy_params(model)
    return compute_hessian_bounds_only(
        model, x, delta, weights=weights,
        jacobian_bound_method=jacobian_bound_method,
        entries_to_compute=entries_to_compute
    )


# --- 5. EXACT Hessian Computation ---
def compute_hessian(model, x, weights=None, x_l=None, z_l=None, J_all=None, **kwargs):
    if weights is None: weights, _ = extract_numpy_params(model)
    L, d_out = len(weights), weights[-1].shape[0]

    if x_l is None or z_l is None:
        x_l, z_l = forward_pass(model, x.reshape(-1, 1))
    if J_all is None:
        J_all = backward_pass(model, z_l)

    H = [None] * (L + 1)
    H[L] = np.zeros((d_out, d_out, d_out))

    for l in range(L - 1, -1, -1):
        W = weights[l]
        d_l, d_lp1 = W.shape[1], W.shape[0]

        if l == L - 1:
            sigma_p = np.ones(d_lp1)
            sigma_pp = np.zeros(d_lp1)
        else:
            z_vec = np.atleast_1d(np.squeeze(z_l[l]))
            sigma_p = sigma_derivative(model, z_vec).flatten()
            sigma_pp = sigma_second_derivative(model, z_vec).flatten()

        S = sigma_p[:, None] * W
        H[l] = np.zeros((d_out, d_l, d_l))
        for k in range(d_out):
            T1 = S.T @ H[l + 1][k] @ S
            grad_output_wrt_z_next = np.atleast_1d(J_all[l + 1][:, k].flatten())
            v = grad_output_wrt_z_next * sigma_pp
            T2 = W.T @ (np.diag(v) @ W)
            H[l][k] = T1 + T2

    return H[0]