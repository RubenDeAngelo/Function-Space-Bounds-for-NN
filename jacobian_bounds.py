"""
jacobian_bounds.py

Implements rigorous upper and lower bounds for the Jacobian of a neural network
using Interval Bound Propagation (IBP) and RecurJac.

OPTIMIZATION:
- Vectorized the interval multiplication loop to remove redundant calculations.
- Added EXACT RecurJac matrix-recursion (Algorithm 1) for significantly tighter bounding.
- Fully vectorized compute_deltas and compute_omega.
"""

import numpy as np
from flex_model import extract_numpy_params
from activations import (
    sigma, compute_sigma_jac_intervals_all_layers
)
from network_ops import forward_pass, backward_pass


def compute_deltas(model, x_l, delta):
    weights, biases = extract_numpy_params(model)
    L = len(weights) - 1

    x_l = [np.atleast_1d(np.squeeze(x)) for x in x_l]
    biases = [np.atleast_1d(np.squeeze(b)) for b in biases]

    delta_up_bar = [delta * np.ones_like(x_l[0])]
    delta_low_bar = [delta * np.ones_like(x_l[0])]

    for l in range(L):
        W = weights[l]
        W_p = np.maximum(W, 0.0)
        W_m = np.minimum(W, 0.0)

        w_times_x = W @ x_l[l]

        upper = sigma(model, w_times_x + W_p @ delta_up_bar[l] - W_m @ delta_low_bar[l] + biases[l]) - x_l[l + 1]
        lower = x_l[l + 1] - sigma(model, w_times_x - W_p @ delta_low_bar[l] + W_m @ delta_up_bar[l] + biases[l])

        delta_up_bar.append(upper)
        delta_low_bar.append(lower)

    W_L = weights[L]
    W_L_plus = np.maximum(W_L, 0.0)
    W_L_minus = np.minimum(W_L, 0.0)

    delta_up_bar.append(W_L_plus @ delta_up_bar[L] - W_L_minus @ delta_low_bar[L])
    delta_low_bar.append(W_L_plus @ delta_low_bar[L] - W_L_minus @ delta_up_bar[L])

    # Bugfixed: Clean vectorized zipping ensures no layer is skipped
    delta_prime = [np.maximum(np.abs(u), np.abs(l)) for u, l in zip(delta_up_bar, delta_low_bar)]

    return delta_prime, delta_low_bar, delta_up_bar


def compute_omega(L, weights, delta_prime_jac):
    """
    Vectorized computation of omega across all layers.
    Replaces the slow per-neuron loop with a single batched matrix multiplication.
    """
    return [np.abs(weights[l]) @ delta_prime_jac[l] for l in range(L)]


# --- ORIGINAL IBP FUNCTION (VECTORIZED) ---
def compute_jac_bounds(weights, alpha_all, beta_all, d_out):
    """
    Propagates the Jacobian bounds backward using completely vectorized affine IBP.
    Runs significantly faster than nested loops.
    """
    L = len(weights)

    jac_bound_up = [None] * L
    jac_bound_down = [None] * L

    W_L_minus_1 = weights[L - 1].astype(np.float64)
    jac_bound_down[L - 1] = W_L_minus_1.copy()
    jac_bound_up[L - 1] = W_L_minus_1.copy()

    A = W_L_minus_1.copy()
    B = W_L_minus_1.copy()

    for l in range(L - 2, -1, -1):
        W = weights[l].astype(np.float64)
        D_under = alpha_all[l].astype(np.float64)
        D_over = beta_all[l].astype(np.float64)

        U_Y_Sigma = np.maximum(A, 0.0) * D_over + np.minimum(A, 0.0) * D_under
        L_Y_Sigma = np.maximum(B, 0.0) * D_under + np.minimum(B, 0.0) * D_over

        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        A = U_Y_Sigma @ W_pos + L_Y_Sigma @ W_neg
        B = L_Y_Sigma @ W_pos + U_Y_Sigma @ W_neg

        jac_bound_up[l] = A.copy()
        jac_bound_down[l] = B.copy()

    return jac_bound_up, jac_bound_down


# --- NEW RECURJAC FUNCTION (EXACT ALGORITHM 1) ---
def _recurjac_compute_lu(l_curr, W_curr, L_hist, U_hist, weights, alpha_all, beta_all, j_out):
    """
    Recursive helper function implementing ComputeLU from Algorithm 1 of the RecurJac paper.
    Evaluates Term I and Term II (Eq 13-19) for a specific target output neuron j_out.
    """
    if l_curr == len(weights) - 1:
        return W_curr[j_out:j_out+1, :].copy(), W_curr[j_out:j_out+1, :].copy()

    L_prev = L_hist[l_curr + 1]
    U_prev = U_hist[l_curr + 1]

    n_r, n_k = W_curr.shape

    l_prime = alpha_all[l_curr]
    u_prime = beta_all[l_curr]

    term1_mask = (L_prev[0] < 0) & (U_prev[0] > 0)

    W_pos = np.maximum(W_curr, 0.0)
    W_neg = np.minimum(W_curr, 0.0)

    D_L = np.where(term1_mask, u_prime * L_prev[0], 0.0)
    D_U = np.where(term1_mask, u_prime * U_prev[0], 0.0)

    L_new = np.dot(D_L, W_pos) + np.dot(D_U, W_neg)
    U_new = np.dot(D_U, W_pos) + np.dot(D_L, W_neg)

    L_new = L_new.reshape(1, n_k)
    U_new = U_new.reshape(1, n_k)

    if not np.all(term1_mask):
        W_above = weights[l_curr + 1]

        L_ge_0 = (L_prev[0] >= 0)[:, None]
        U_le_0 = (U_prev[0] <= 0)[:, None]
        W_lt_0 = (W_curr < 0)
        W_gt_0 = (W_curr > 0)

        cond_tilde = (L_ge_0 & W_lt_0) | (U_le_0 & W_gt_0)
        sigma_tilde = np.where(cond_tilde, u_prime[:, None], l_prime[:, None])

        cond_hat = (L_ge_0 & W_gt_0) | (U_le_0 & W_lt_0)
        sigma_hat = np.where(cond_hat, u_prime[:, None], l_prime[:, None])

        V_tilde = sigma_tilde * W_curr
        V_hat = sigma_hat * W_curr

        V_tilde[term1_mask, :] = 0.0
        V_hat[term1_mask, :] = 0.0

        W_tilde = np.dot(W_above, V_tilde)
        W_hat = np.dot(W_above, V_hat)

        L_rec, _ = _recurjac_compute_lu(l_curr + 1, W_tilde, L_hist, U_hist, weights, alpha_all, beta_all, j_out)
        _, U_rec = _recurjac_compute_lu(l_curr + 1, W_hat, L_hist, U_hist, weights, alpha_all, beta_all, j_out)

        L_new += L_rec
        U_new += U_rec

    return L_new, U_new


def compute_recurjac_bounds(weights, alpha_all, beta_all, d_out):
    """
    Propagates the Jacobian bounds backward using the EXACT RecurJac algorithm (Algorithm 1).
    Applies the Term I / Term II split and recursively re-evaluates the bounds.
    """
    L_total = len(weights)

    jac_bound_up = [np.zeros((d_out, W.shape[1])) for W in weights]
    jac_bound_down = [np.zeros((d_out, W.shape[1])) for W in weights]

    jac_bound_up[L_total - 1] = weights[L_total - 1].copy()
    jac_bound_down[L_total - 1] = weights[L_total - 1].copy()

    for j in range(d_out):
        L_hist = [None] * L_total
        U_hist = [None] * L_total
        L_hist[L_total - 1] = weights[L_total - 1][j:j+1, :]
        U_hist[L_total - 1] = weights[L_total - 1][j:j+1, :]

        for l in range(L_total - 2, -1, -1):
            L_l, U_l = _recurjac_compute_lu(
                l, weights[l], L_hist, U_hist, weights, alpha_all, beta_all, j
            )
            L_hist[l] = L_l
            U_hist[l] = U_l

            jac_bound_down[l][j:j+1, :] = L_l
            jac_bound_up[l][j:j+1, :] = U_l

    return jac_bound_up, jac_bound_down


# --- MAIN ENTRY POINT ---
def compute_jacobian_and_bounds(model, x, delta, use_recurjac=False):
    """
    Main function to compute Jacobian bounds.
    Set use_recurjac=False to fall back to the older IBP method.
    """
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

    if use_recurjac:
        jac_bound_up_list, jac_bound_down_list = compute_recurjac_bounds(
            weights, alpha_all, beta_all, d_out
        )
    else:
        jac_bound_up_list, jac_bound_down_list = compute_jac_bounds(
            weights, alpha_all, beta_all, d_out
        )

    J_low = jac_bound_down_list[0]
    J_high = jac_bound_up_list[0]

    return J_low, J_high