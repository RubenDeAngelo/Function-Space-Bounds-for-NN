"""Dev: compares global Jacobian bound gaps for IBP, CROWN, and RecurJac on the 2D smooth-coin domain."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import numba
import torch
import itertools

from lib_nn_model import FlexibleNet, extract_numpy_params
from lib_nn_interval_ops import forward_pass
from lib_jacobian_enclosures import (
    compute_deltas,
    compute_omega,
    compute_jac_bounds,
    compute_recurjac_bounds
)
from lib_activation_enclosures import compute_sigma_jac_intervals_all_layers

LAYERS = [2, 32, 32, 32, 1]
ACTIVATION = 'sigmoid'
EPSILON_TRANSITION = 1
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}_{EPSILON_TRANSITION}.pth"

THETA = 0.35
MAX_ITER = 15
@numba.jit(nopython=True, fastmath=True)
def compute_crown_bounds(weights, alpha_all, beta_all, d_out):
    """
    Computes bounds using simple affine propagation (similar to CROWN/Fast-Lin).
    This separates positive/negative weights but skips the RecurJac recursion.
    """
    L = len(weights)
    jac_bound_up = [np.empty((1, 1), dtype=np.float64) for _ in range(L)]
    jac_bound_down = [np.empty((1, 1), dtype=np.float64) for _ in range(L)]

    W_L_minus_1 = weights[L - 1].astype(np.float64)
    jac_bound_down[L - 1] = W_L_minus_1.copy()
    jac_bound_up[L - 1] = W_L_minus_1.copy()

    # A is Upper Bound, B is Lower Bound
    A = W_L_minus_1.copy()
    B = W_L_minus_1.copy()

    for l in range(L - 2, -1, -1):
        W = weights[l].astype(np.float64)
        D_under = alpha_all[l].astype(np.float64)
        D_over = beta_all[l].astype(np.float64)

        # 1. Extreme possible values of (Y * Sigma)
        U_Y_Sigma = np.maximum(A, 0.0) * D_over + np.minimum(A, 0.0) * D_under
        L_Y_Sigma = np.maximum(B, 0.0) * D_under + np.minimum(B, 0.0) * D_over

        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        # 2. Project through weights
        # Upper bound: Max_incoming * positive_weights + Min_incoming * negative_weights
        A_new = np.dot(U_Y_Sigma, W_pos) + np.dot(L_Y_Sigma, W_neg)
        # Lower bound: Min_incoming * positive_weights + Max_incoming * negative_weights
        B_new = np.dot(L_Y_Sigma, W_pos) + np.dot(U_Y_Sigma, W_neg)

        A = A_new
        B = B_new

        jac_bound_up[l] = A.copy()
        jac_bound_down[l] = B.copy()

    return jac_bound_up, jac_bound_down

def split_points(points, delta):
    """Splits a 2D box into 4 smaller boxes."""
    new_points = []
    new_delta = delta / 2.0
    d = points[0].shape[0]
    offset_combinations = list(itertools.product([-1, 1], repeat=d))
    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape) * new_delta
            new_points.append(x + offset)
    return new_points, new_delta

def evaluate_cell_all_methods(model, x, delta):
    """Evaluates the Jacobian bounds using IBP, CROWN, and RecurJac efficiently."""
    weights, _ = extract_numpy_params(model)
    L = len(weights)
    d_out = weights[-1].shape[0]

    # Forward Pass & Sigmoid Intervals (Calculated only ONCE per cell)
    x_reshaped = x.reshape(-1, 1)
    x_l, z_l = forward_pass(model, x_reshaped)
    delta_prime_jac, _, _ = compute_deltas(model, x_l, delta)
    omega_all = compute_omega(L, weights, delta_prime_jac)
    alpha_all, beta_all = compute_sigma_jac_intervals_all_layers(model, z_l, omega_all)

    # 1. Standard IBP
    up_ibp, down_ibp = compute_jac_bounds(weights, alpha_all, beta_all, d_out)
    gap_ibp = np.sum(up_ibp[0] - down_ibp[0])

    # 2. CROWN Variant
    up_crown, down_crown = compute_crown_bounds(weights, alpha_all, beta_all, d_out)
    gap_crown = np.sum(up_crown[0] - down_crown[0])

    # 3. Exact RecurJac
    up_rj, down_rj = compute_recurjac_bounds(weights, alpha_all, beta_all, d_out)
    gap_rj = np.sum(up_rj[0] - down_rj[0])

    return gap_ibp, gap_crown, gap_rj

def run_doerfler_comparison():
    print("=== Dörfler Refinement: IBP vs. CROWN vs. RecurJac ===\n")

    # 1. Setup Model
    model = FlexibleNet(LAYERS, activation=ACTIVATION)
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained weights from {MODEL_PATH}...\n")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"WARNING: {MODEL_PATH} not found. Using random weights.\n")
    model.eval()

    # 2. Initial Grid Setup
    domain_start, domain_end = -1.5, 1.5
    N_side = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # Initialize active cells
    active_cells = []
    for x in centers_init:
        g_ibp, g_crown, g_rj = evaluate_cell_all_methods(model, x, delta_init)
        active_cells.append({
            'center': x,
            'delta': delta_init,
            'gap_ibp': g_ibp,
            'gap_crown': g_crown,
            'gap_rj': g_rj
        })

    print(f"{'Iter':<5} | {'Cells':<6} | {'Avg Gap IBP':<15} | {'Avg Gap CROWN':<15} | {'Avg Gap RecurJac':<16}")
    print("-" * 75)

    # 3. Dörfler Refinement Loop
    for i in range(MAX_ITER + 1):
        num_cells = len(active_cells)

        # Calculate global metrics
        avg_ibp = sum(c['gap_ibp'] for c in active_cells) / num_cells
        avg_crown = sum(c['gap_crown'] for c in active_cells) / num_cells
        avg_rj = sum(c['gap_rj'] for c in active_cells) / num_cells

        print(f"{i:<5} | {num_cells:<6} | {avg_ibp:<15.6f} | {avg_crown:<15.6f} | {avg_rj:<16.6f}")

        if i == MAX_ITER:
            break

        # Dörfler Marking: Drive refinement using the most precise gap (RecurJac)
        total_error = sum(c['gap_rj'] for c in active_cells)
        active_cells.sort(key=lambda c: c['gap_rj'], reverse=True)

        marked_set = []
        unmarked_set = []
        current_sum = 0.0
        condition_met = False

        for cell in active_cells:
            if not condition_met:
                marked_set.append(cell)
                current_sum += cell['gap_rj']
                if current_sum >= THETA * total_error:
                    condition_met = True
            else:
                unmarked_set.append(cell)

        # Splitting marked cells
        new_cells = []
        for parent in marked_set:
            child_centers, child_delta = split_points([parent['center']], parent['delta'])
            for cx in child_centers:
                g_ibp, g_crown, g_rj = evaluate_cell_all_methods(model, cx, child_delta)
                new_cells.append({
                    'center': cx,
                    'delta': child_delta,
                    'gap_ibp': g_ibp,
                    'gap_crown': g_crown,
                    'gap_rj': g_rj
                })

        active_cells = unmarked_set + new_cells

if __name__ == "__main__":
    run_doerfler_comparison()