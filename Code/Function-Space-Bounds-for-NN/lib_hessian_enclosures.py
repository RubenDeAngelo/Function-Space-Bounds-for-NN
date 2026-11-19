"""
Hessian enclosures via vectorized IBP with optional RecurJac Jacobian sub-bounds.
Entry point: compute_hessian_bounds(model, x, delta).

Two computational paths:
  - Dense (default): fully vectorized NumPy/einsum, O(N^3).
  - Sparse: Numba JIT loop for individual entry queries (entries_to_compute).
"""

import numpy as np
import numba
from lib_nn_model import extract_numpy_params
from lib_jacobian_enclosures import (
    compute_deltas, compute_omega,
    compute_jac_bounds as compute_jacobian_ibp_bounds,
    compute_recurjac_bounds
)
from lib_activation_enclosures import (
    compute_sigma_jac_intervals_all_layers,
    compute_sigma_hess_intervals_all_layers,
    sigma_derivative,
    sigma_second_derivative
)
from lib_nn_interval_ops import forward_pass, backward_pass


@numba.jit(nopython=True, fastmath=True, inline='always')
def _interval_mul(a_lo, a_hi, b_lo, b_hi):
    p1 = a_lo * b_lo
    p2 = a_lo * b_hi
    p3 = a_hi * b_lo
    p4 = a_hi * b_hi
    return min(p1, p2, p3, p4), max(p1, p2, p3, p4)


@numba.jit(nopython=True, fastmath=True, inline='always')
def _scalar_mul_interval(scalar, lo, hi):
    if scalar >= 0:
        return scalar * lo, scalar * hi
    else:
        return scalar * hi, scalar * lo


@numba.jit(nopython=True, fastmath=True)
def _jit_hessian_loop_sparse(d_out, d_l, d_lp1, H_next_low, H_next_high,
                             alpha_vec, beta_vec, W, J_next_low, J_next_high,
                             gamma_vec, eta_vec, entries_to_compute):
    """Numba sparse loop for individual Hessian entry queries."""
    H_low = np.zeros((d_out, d_l, d_l))
    H_high = np.zeros((d_out, d_l, d_l))

    for k in range(d_out):
        for entry_idx in range(entries_to_compute.shape[0]):
            i, j = entries_to_compute[entry_idx]

            if i >= d_l or j >= d_l: continue

            t1_min, t1_max = 0.0, 0.0
            t2_min, t2_max = 0.0, 0.0

            for p in range(d_lp1):
                alpha_p, beta_p = alpha_vec[p], beta_vec[p]

                for q in range(d_lp1):
                    alpha_q, beta_q = alpha_vec[q], beta_vec[q]

                    if p == q:
                        if alpha_p < 0 and beta_p > 0:
                            s_lo = 0.0
                            s_hi = max(alpha_p ** 2, beta_p ** 2)
                        else:
                            s_lo = min(alpha_p ** 2, beta_p ** 2)
                            s_hi = max(alpha_p ** 2, beta_p ** 2)
                    else:
                        s_lo, s_hi = _interval_mul(alpha_p, beta_p, alpha_q, beta_q)

                    h_lo, h_hi = H_next_low[k, p, q], H_next_high[k, p, q]
                    v_lo, v_hi = _interval_mul(h_lo, h_hi, s_lo, s_hi)

                    w_scalar = W[p, i] * W[q, j]
                    term_lo, term_hi = _scalar_mul_interval(w_scalar, v_lo, v_hi)

                    t1_min += term_lo
                    t1_max += term_hi

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
    """
    One backward step of the Hessian recursion.
    Uses the sparse Numba path for entry queries; otherwise fully vectorized NumPy.
    """
    d_lp1, d_l = W.shape
    d_out = H_next_low.shape[0]

    if entries_to_compute is not None:
        entries_arr = np.array(entries_to_compute, dtype=np.int64)
        W_np = W.astype(np.float64)
        return _jit_hessian_loop_sparse(d_out, d_l, d_lp1, H_next_low, H_next_high,
                                        alpha_vec, beta_vec, W_np, J_next_low, J_next_high,
                                        gamma_vec, eta_vec, entries_arr)

    W_f64 = W.astype(np.float64)
    W_p = np.maximum(W_f64, 0.0)
    W_n = np.minimum(W_f64, 0.0)

    # Term 1: backprop curvature H_new = W^T (H_next * S) W
    p1_s = np.outer(alpha_vec, alpha_vec)
    p2_s = np.outer(alpha_vec, beta_vec)
    p3_s = np.outer(beta_vec, alpha_vec)
    p4_s = np.outer(beta_vec, beta_vec)

    s_lo = np.minimum.reduce([p1_s, p2_s, p3_s, p4_s])
    s_hi = np.maximum.reduce([p1_s, p2_s, p3_s, p4_s])

    # Fix diagonal: [alpha, beta]^2 lower bound is 0 when interval crosses zero
    crosses_zero = (alpha_vec < 0) & (beta_vec > 0)
    if np.any(crosses_zero):
        diag_idx = np.arange(len(alpha_vec))
        s_lo[diag_idx, diag_idx] = np.where(crosses_zero, 0.0, s_lo[diag_idx, diag_idx])

    p1_v = H_next_low * s_lo
    p2_v = H_next_low * s_hi
    p3_v = H_next_high * s_lo
    p4_v = H_next_high * s_hi

    V_low = np.minimum.reduce([p1_v, p2_v, p3_v, p4_v])
    V_high = np.maximum.reduce([p1_v, p2_v, p3_v, p4_v])

    M_high = V_high @ W_p + V_low @ W_n
    M_low  = V_low @ W_p + V_high @ W_n

    T1_high = W_p.T @ M_high + W_n.T @ M_low
    T1_low  = W_p.T @ M_low + W_n.T @ M_high

    # Term 2: local curvature D = J * sigma''
    p1_d = J_next_low * gamma_vec
    p2_d = J_next_low * eta_vec
    p3_d = J_next_high * gamma_vec
    p4_d = J_next_high * eta_vec

    D_low = np.minimum.reduce([p1_d, p2_d, p3_d, p4_d])
    D_high = np.maximum.reduce([p1_d, p2_d, p3_d, p4_d])

    W_sq = W_f64[:, :, None] * W_f64[:, None, :]
    W_sq_p = np.maximum(W_sq, 0.0)
    W_sq_n = np.minimum(W_sq, 0.0)

    T2_high = np.einsum('kp, pij -> kij', D_high, W_sq_p) + np.einsum('kp, pij -> kij', D_low, W_sq_n)
    T2_low  = np.einsum('kp, pij -> kij', D_low, W_sq_p) + np.einsum('kp, pij -> kij', D_high, W_sq_n)

    H_high = T1_high + T2_high
    H_low  = T1_low + T2_low

    return H_low, H_high


def compute_jacobian_bounds_for_hessian(model, weights, z_l, J_all, omega_all, d_out,
                                        alpha_all, beta_all, method='ibp'):
    if method == 'ibp':
        return compute_jacobian_ibp_bounds(weights, alpha_all, beta_all, d_out)
    elif method == 'recurjac':
        return compute_recurjac_bounds(weights, alpha_all, beta_all, d_out)
    else:
        raise ValueError("Method must be 'ibp' or 'recurjac'.")


def compute_hessian_bounds_only(model, x, delta, weights=None, x_l=None, z_l=None,
                                delta_prime_jac=None, omega_all=None, J_all=None,
                                alpha_all=None, beta_all=None, gamma_all=None, eta_all=None,
                                jac_bound_up=None, jac_bound_down=None,
                                H_low=None, H_high=None,
                                jacobian_bound_method='ibp',
                                entries_to_compute=None):
    """
    Full Hessian enclosure with optional precomputed intermediates for reuse.
    Returns (H_low, H_high) of shape (d_out, d_in, d_in).
    """
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
            # At the output layer the Jacobian of the network w.r.t. itself is identity
            J_next_low = np.eye(d_out, dtype=np.float64)
            J_next_high = np.eye(d_out, dtype=np.float64)
            alpha_l = np.ones_like(alpha_all[l]).flatten()
            beta_l = np.ones_like(beta_all[l]).flatten()
            gamma_l = np.zeros_like(gamma_all[l]).flatten()
            eta_l = np.zeros_like(eta_all[l]).flatten()
        else:
            J_next_low = jac_bound_down[l + 1]
            J_next_high = jac_bound_up[l + 1]
            alpha_l = alpha_all[l].flatten()
            beta_l = beta_all[l].flatten()
            gamma_l = gamma_all[l].flatten()
            eta_l = eta_all[l].flatten()

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
    """Convenience wrapper around compute_hessian_bounds_only."""
    weights, _ = extract_numpy_params(model)
    return compute_hessian_bounds_only(
        model, x, delta, weights=weights,
        jacobian_bound_method=jacobian_bound_method,
        entries_to_compute=entries_to_compute
    )


def compute_hessian(model, x, weights=None, x_l=None, z_l=None, J_all=None, **kwargs):
    """Exact (non-interval) Hessian at a single point x, via the chain rule."""
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
