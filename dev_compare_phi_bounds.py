"""Dev: compares forward-pass (phi) bound gaps for IBP vs affine propagation on the 2D smooth-coin domain."""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import itertools

from lib_nn_model import FlexibleNet, extract_numpy_params

LAYERS = [2, 32, 32, 32, 32, 1]
ACTIVATION = 'sigmoid'
EPSILON_TRANSITION = 0.2
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}_{EPSILON_TRANSITION}.pth"

THETA = 0.35
MAX_ITER = 15

def compute_phi_ibp(model, x, delta):
    weights, biases = extract_numpy_params(model)
    L = len(weights)

    x_curr = np.atleast_1d(np.squeeze(x))
    lb = x_curr - delta
    ub = x_curr + delta

    for l in range(L):
        W = weights[l]
        b = np.atleast_1d(np.squeeze(biases[l]))

        W_p = np.maximum(W, 0.0)
        W_m = np.minimum(W, 0.0)

        ub_next = W_p @ ub + W_m @ lb + b
        lb_next = W_p @ lb + W_m @ ub + b

        if l < L - 1:
            lb = 1 / (1 + np.exp(-lb_next))
            ub = 1 / (1 + np.exp(-ub_next))
        else:
            lb, ub = lb_next, ub_next

    return lb[0], ub[0]

def compute_phi_affine(model, x, delta):
    weights, biases = extract_numpy_params(model)
    L = len(weights)

    x_curr = np.atleast_1d(np.squeeze(x))
    d_in = len(x_curr)

    A_lb = np.eye(d_in)
    A_ub = np.eye(d_in)
    b_lb = x_curr.copy()
    b_ub = x_curr.copy()

    for l in range(L):
        W = weights[l]
        b = np.atleast_1d(np.squeeze(biases[l]))

        W_p = np.maximum(W, 0.0)
        W_m = np.minimum(W, 0.0)

        A_lb_new = W_p @ A_lb + W_m @ A_ub
        A_ub_new = W_p @ A_ub + W_m @ A_lb
        b_lb_new = W_p @ b_lb + W_m @ b_ub + b
        b_ub_new = W_p @ b_ub + W_m @ b_lb + b

        lower_bound = b_lb_new - np.sum(np.abs(A_lb_new) * delta, axis=1)
        upper_bound = b_ub_new + np.sum(np.abs(A_ub_new) * delta, axis=1)

        if l < L - 1:
            sig_lb = 1 / (1 + np.exp(-lower_bound))
            sig_ub = 1 / (1 + np.exp(-upper_bound))

            slope = np.where(upper_bound > lower_bound,
                             (sig_ub - sig_lb) / (upper_bound - lower_bound + 1e-9),
                             sig_lb * (1 - sig_lb))

            A_lb = A_lb_new * slope[:, None]
            A_ub = A_ub_new * slope[:, None]

            intercept_lb = sig_lb - slope * lower_bound
            intercept_ub = sig_ub - slope * upper_bound

            b_lb = b_lb_new * slope + intercept_lb
            b_ub = b_ub_new * slope + intercept_ub
        else:
            lb_final = lower_bound[0]
            ub_final = upper_bound[0]

    return lb_final, ub_final

def split_points(points, delta):
    new_points = []
    new_delta = delta / 2.0
    d = points[0].shape[0]
    offset_combinations = list(itertools.product([-1, 1], repeat=d))
    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape) * new_delta
            new_points.append(x + offset)
    return new_points, new_delta

def evaluate_cell(model, x, delta):
    lb_ibp, ub_ibp = compute_phi_ibp(model, x, delta)
    gap_ibp = ub_ibp - lb_ibp

    lb_aff, ub_aff = compute_phi_affine(model, x, delta)
    gap_aff = ub_aff - lb_aff

    return gap_ibp, gap_aff

def run_doerfler_phi_comparison():
    print("=== Dörfler Refinement: Forward Bounds (Phi) IBP vs. Affine ===\n")

    model = FlexibleNet(LAYERS, activation=ACTIVATION)
    if os.path.exists(MODEL_PATH):
        print(f"Loading trained weights from {MODEL_PATH}...\n")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"WARNING: {MODEL_PATH} not found. Using random weights.\n")
    model.eval()

    domain_start, domain_end = -1.5, 1.5
    N_side = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    active_cells = []
    for x in centers_init:
        g_ibp, g_aff = evaluate_cell(model, x, delta_init)
        active_cells.append({
            'center': x,
            'delta': delta_init,
            'gap_ibp': g_ibp,
            'gap_aff': g_aff
        })

    print(f"{'Iter':<5} | {'Cells':<6} | {'Avg Gap IBP':<15} | {'Avg Gap Affine':<15} | {'Improvement (%)':<15}")
    print("-" * 65)

    for i in range(MAX_ITER + 1):
        num_cells = len(active_cells)

        avg_gap_ibp = sum(c['gap_ibp'] for c in active_cells) / num_cells
        avg_gap_aff = sum(c['gap_aff'] for c in active_cells) / num_cells

        improvement = ((avg_gap_ibp - avg_gap_aff) / avg_gap_ibp) * 100 if avg_gap_ibp > 0 else 0

        print(f"{i:<5} | {num_cells:<6} | {avg_gap_ibp:<15.6f} | {avg_gap_aff:<15.6f} | {improvement:<15.2f}")

        if i == MAX_ITER:
            break

        total_error = sum(c['gap_aff'] for c in active_cells)
        active_cells.sort(key=lambda c: c['gap_aff'], reverse=True)

        marked_set = []
        unmarked_set = []
        current_sum = 0.0
        condition_met = False

        for cell in active_cells:
            if not condition_met:
                marked_set.append(cell)
                current_sum += cell['gap_aff']
                if current_sum >= THETA * total_error:
                    condition_met = True
            else:
                unmarked_set.append(cell)

        new_cells = []
        for parent in marked_set:
            child_centers, child_delta = split_points([parent['center']], parent['delta'])
            for cx in child_centers:
                g_ibp, g_aff = evaluate_cell(model, cx, child_delta)
                new_cells.append({
                    'center': cx,
                    'delta': child_delta,
                    'gap_ibp': g_ibp,
                    'gap_aff': g_aff
                })

        active_cells = unmarked_set + new_cells

if __name__ == "__main__":
    run_doerfler_phi_comparison()