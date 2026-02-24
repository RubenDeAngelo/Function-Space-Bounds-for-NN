"""
run_independent_refinement.py

Comparison of W1p and W2p norms using INDEPENDENT Dynamic Grids with DÖRFLER MARKING.
- Grid 1: Optimized specifically to minimize W1p error (Theta-Marking).
- Grid 2: Optimized specifically to minimize W2p error (Theta-Marking).
- Static Baselines: Computed for every 4th step of the adaptive trajectory to save time.
- Theory: Adds O(N^-1) convergence lines for both W1p and W2p norms.

UPDATED: Fully Theorem-Compliant metrics, Monte Carlo Integration, and Publication Formatting.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import pi
import itertools

# --- Global Font Size Configuration ---
gloabal_size_para = 20
plt.rcParams.update({
    'font.size': gloabal_size_para,
    'axes.titlesize': gloabal_size_para,
    'axes.labelsize': gloabal_size_para,
    'xtick.labelsize': gloabal_size_para,
    'ytick.labelsize': gloabal_size_para,
    'legend.fontsize': gloabal_size_para,
    'figure.titlesize': gloabal_size_para
})

# --- Imports ---
from flex_model import FlexibleNet
from training import train_model
from w1p_bounds import compute_w1p_bounds
from w2p_bounds import compute_w2p_bounds

# --- CONFIGURATION ---
LAYERS = [1, 200, 1]
ACTIVATION = 'sigmoid'
P_NORM = 2.0
MAX_REFINE_DEPTH = 30
THETA = 0.4         # Dörfler Parameter
STATIC_STEP = 5     # Only compute static baseline every 4th iteration

FIG_DIR = f"fig_independent_refinement_{LAYERS}"
os.makedirs(FIG_DIR, exist_ok=True)

# --- UTILS ---

def split_points(points, delta):
    """Splits a list of center points into 2^d sub-intervals."""
    new_points = []
    if np.isscalar(delta):
        new_delta = delta / 2.0
    else:
         new_delta = delta / 2.0

    d = points[0].shape[0]
    offsets = [-new_delta, new_delta]
    offset_combinations = list(itertools.product(offsets, repeat=d))

    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape)
            new_points.append(x + offset)

    return new_points, new_delta

def compute_static_gap(model, domain_start, domain_end, num_cells, p=2.0, norm_type='w1p'):
    """Computes the static grid gap and quadrature for a specific cell count and norm."""
    total = domain_end - domain_start
    delta = (total / num_cells) / 2.0
    centers = np.linspace(domain_start + delta, domain_end - delta, num_cells).astype(np.float32)

    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    bounds = []
    for c in centers:
        c_reshaped = c.reshape(-1, 1) if isinstance(c, np.ndarray) and c.ndim==0 else c
        l, h = bound_fn(model, c_reshaped, delta, p=p)
        bounds.append({'low': l, 'high': h})

    gap = sum(0.5 * ((b['high'] ** p) - (b['low'] ** p)) for b in bounds)
    q_val = sum(0.5 * ((b['high'] ** p) + (b['low'] ** p)) for b in bounds)
    return gap, q_val

def compute_monte_carlo_sobolev_1d(model, num_samples=100000, p=2.0, domain=(0.0, 2*pi)):
    """Computes exact analytical Sobolev bounds via 1D Monte Carlo sampling."""
    print(f"\n--- Computing 1D Monte Carlo Ground Truth (N={num_samples}) ---")

    device = next(model.parameters()).device
    X = torch.empty(num_samples, 1, device=device).uniform_(domain[0], domain[1]).requires_grad_(True)
    volume = domain[1] - domain[0]

    # Forward pass: Phi
    Y = model(X)  # Shape: (N, 1)

    # First derivative: Jacobian (Grad)
    grad_outputs = torch.ones_like(Y)
    grad_Y = torch.autograd.grad(Y, X, grad_outputs=grad_outputs, create_graph=True)[0]  # Shape: (N, 1)

    # Second derivative: Hessian
    hess_Y = torch.autograd.grad(grad_Y, X, grad_outputs=torch.ones_like(grad_Y), retain_graph=False)[0]  # Shape: (N, 1)

    # Calculate L^p integrands per sample
    phi_p = torch.abs(Y).pow(p).sum(dim=1)
    grad_p = torch.abs(grad_Y).pow(p).sum(dim=1)
    hess_p = torch.abs(hess_Y).pow(p).sum(dim=1)

    # Combine for W1p and W2p
    w1p_integrand = phi_p + grad_p
    w2p_integrand = phi_p + grad_p + hess_p

    # Monte Carlo Integral = Volume * Average
    w1p_mc = volume * w1p_integrand.mean().item()
    w2p_mc = volume * w2p_integrand.mean().item()

    print(f"MC W1P True Value: {w1p_mc:.6f}")
    print(f"MC W2P True Value: {w2p_mc:.6f}")
    return w1p_mc, w2p_mc

def generate_localized_data(N=150):
    """Generates 1D Gaussian pulse data."""
    X = np.linspace(0, 2*pi, N)
    y = np.exp(-20 * (X - pi)**2)
    return X.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)

# --- INDEPENDENT REFINEMENT LOGIC ---

def run_doerfler_grid_for_norm(model, initial_centers, initial_delta, max_iter, theta=0.5, p=2.0, norm_type='w1p'):
    """Runs a dynamic refinement loop specifically for ONE norm type using Dörfler Marking."""
    print(f"\n--- Running Dörfler Refinement (theta={theta}, p={p}) for {norm_type.upper()} ---")

    active_cells = []
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    if np.isscalar(initial_delta):
         initial_batch = [(c.reshape(-1, 1), initial_delta) for c in initial_centers]
    else:
         initial_batch = zip(initial_centers, initial_delta)

    for x, d in initial_batch:
        lb, ub = bound_fn(model, x, d, p=p)
        # Theorem compliant metrics
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        q_val = 0.5 * ((ub ** p) + (lb ** p))
        active_cells.append({
            'center': x, 'delta': d,
            'low': lb, 'high': ub,
            'error': error_mass,
            'q_val': q_val
        })

    history = []

    for i in range(max_iter + 1):
        count = len(active_cells)
        global_gap = sum(c['error'] for c in active_cells)
        global_q = sum(c['q_val'] for c in active_cells)

        history.append({'iter': i, 'count': count, 'gap_dyn': global_gap, 'q_dyn': global_q})
        print(f"Iter {i:<3} | Cells: {count:<5} | {norm_type.upper()} Gap: {global_gap:.6f} | Q_n: {global_q:.6f}")

        if i == max_iter: break

        total_error = sum(c['error'] for c in active_cells)
        active_cells.sort(key=lambda c: c['error'], reverse=True)

        marked_set = []
        unmarked_set = []
        current_sum = 0.0
        condition_met = False

        for cell in active_cells:
            if not condition_met:
                marked_set.append(cell)
                current_sum += cell['error']
                if current_sum >= theta * total_error:
                    condition_met = True
            else:
                unmarked_set.append(cell)

        new_cells = []
        for parent in marked_set:
            child_centers, child_delta = split_points([parent['center']], parent['delta'])
            for cx in child_centers:
                cl, cu = bound_fn(model, cx, child_delta, p=p)
                # Theorem compliant metrics for children
                child_error_mass = 0.5 * ((cu ** p) - (cl ** p))
                child_q_val = 0.5 * ((cu ** p) + (cl ** p))
                new_cells.append({
                    'center': cx, 'delta': child_delta,
                    'low': cl, 'high': cu,
                    'error': child_error_mass,
                    'q_val': child_q_val
                })
        active_cells = unmarked_set + new_cells

    return history


def run_independent_experiment():
    print(f"--- Starting Independent Refinement Experiment (Theorem Compliant) ---")

    # 1. Train Model
    X_np, y_np = generate_localized_data()
    print("Training Model...")
    model = FlexibleNet(LAYERS, activation=ACTIVATION)
    train_model(model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=6000)
    model.eval()

    # Initial Grid Setup
    domain_start, domain_end = 0.0, 2 * pi
    N_init = 2
    delta_init = ((domain_end - domain_start) / N_init) / 2.0
    centers_init = np.linspace(domain_start + delta_init, domain_end - delta_init, N_init).astype(np.float32)

    # 2. Run W1 Refinement
    w1_history = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_REFINE_DEPTH, THETA, P_NORM, 'w1p')

    # 3. Run W2 Refinement
    w2_history = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_REFINE_DEPTH, THETA, P_NORM, 'w2p')

    # 4. Compute Matching Static Baselines (SUBSAMPLED)
    print(f"\n--- Computing Static Baselines (Every {STATIC_STEP}th step) ---")

    def get_subsampled_indices(total_len, step):
        indices = list(range(0, total_len, step))
        if indices[-1] != total_len - 1:
            indices.append(total_len - 1)
        return indices

    # For W1 Plot
    w1_full_counts = [h['count'] for h in w1_history]
    w1_dyn_gaps = np.array([h['gap_dyn'] for h in w1_history])
    w1_dyn_qs = np.array([h['q_dyn'] for h in w1_history])

    idx_w1 = get_subsampled_indices(len(w1_full_counts), STATIC_STEP)
    w1_counts_sub = [w1_full_counts[i] for i in idx_w1]

    w1_stat_results = [compute_static_gap(model, domain_start, domain_end, c, P_NORM, 'w1p') for c in w1_counts_sub]
    w1_stat_gaps_sub = [res[0] for res in w1_stat_results]
    w1_stat_qs_sub = [res[1] for res in w1_stat_results]

    # For W2 Plot
    w2_full_counts = [h['count'] for h in w2_history]
    w2_dyn_gaps = np.array([h['gap_dyn'] for h in w2_history])
    w2_dyn_qs = np.array([h['q_dyn'] for h in w2_history])

    idx_w2 = get_subsampled_indices(len(w2_full_counts), STATIC_STEP)
    w2_counts_sub = [w2_full_counts[i] for i in idx_w2]

    w2_stat_results = [compute_static_gap(model, domain_start, domain_end, c, P_NORM, 'w2p') for c in w2_counts_sub]
    w2_stat_gaps_sub = [res[0] for res in w2_stat_results]
    w2_stat_qs_sub = [res[1] for res in w2_stat_results]

    # Calculate Monte Carlo Ground Truth
    mc_w1p, mc_w2p = compute_monte_carlo_sobolev_1d(model, num_samples=100000, p=P_NORM, domain=(domain_start, domain_end))

    # 5. Convergence Plot: Residual Error Gap R_n(f)
    plt.figure(figsize=(16, 12))

    # Optimal Rate for 1D Interval Approx is O(N^-1)
    x_theory = np.linspace(min(w1_full_counts[0], w2_full_counts[0]), max(w1_full_counts[-1], w2_full_counts[-1]), 100)

    n0_w1, e0_w1 = w1_full_counts[0], w1_dyn_gaps[0]
    y_theory_w1 = e0_w1 * (x_theory / n0_w1) ** (-1.0)
    plt.plot(x_theory, y_theory_w1, 'k--', linewidth=1.5, label=r'Theory Smooth $O(N^{-1})$')

    n0_w2, e0_w2 = w2_full_counts[0], w2_dyn_gaps[0]
    y_theory_w2 = e0_w2 * (x_theory / n0_w2) ** (-1.0)
    plt.plot(x_theory, y_theory_w2, 'k:', linewidth=1.5)

    plt.plot(w1_full_counts, w1_dyn_gaps, '-o', color='#D55E00', linewidth=2, label=r'Dörfler $W^{1,2}$ ($\theta=' + str(THETA) + r'$)')
    plt.plot(w2_full_counts, w2_dyn_gaps, '-o', color='#009E73', linewidth=2, label=r'Dörfler $W^{2,2}$ ($\theta=' + str(THETA) + r'$)')
    plt.plot(w1_counts_sub, w1_stat_gaps_sub, '--s', color='#D55E00', alpha=0.4, label=r'Static $W^{1,2}$')
    plt.plot(w2_counts_sub, w2_stat_gaps_sub, '--s', color='#009E73', alpha=0.4, label=r'Static $W^{2,2}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title("1D Localized - Residual Convergence R_n(f)", fontsize=30)
    plt.xlabel("Number of Grid Box", fontsize=30)
    plt.ylabel("Global Residual Bound Gap", fontsize=30)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    save_path_gap = os.path.join(FIG_DIR, f"independent_residual_convergence_{LAYERS}.pdf")
    plt.tight_layout()
    plt.savefig(save_path_gap, dpi=300)
    print(f"\nResidual Convergence plot saved to {save_path_gap}")

    # 6. Convergence Plot: Integral Quadrature Q_n(f)
    plt.figure(figsize=(16, 12))

    # W1p Quadrature
    plt.plot(w1_full_counts, w1_dyn_qs, '-o', color='#D55E00', linewidth=2, label=r'Dörfler $Q_n(f)$ for $W^{1,2}$')
    plt.fill_between(w1_full_counts, w1_dyn_qs - w1_dyn_gaps, w1_dyn_qs + w1_dyn_gaps, color='#D55E00', alpha=0.2, label=r'Uncertainty $\pm R_n(f)$ ($W^{1,2}$)')

    # W2p Quadrature
    plt.plot(w2_full_counts, w2_dyn_qs, '-o', color='#009E73', linewidth=2, label=r'Dörfler $Q_n(f)$ for $W^{2,2}$')
    plt.fill_between(w2_full_counts, w2_dyn_qs - w2_dyn_gaps, w2_dyn_qs + w2_dyn_gaps, color='#009E73', alpha=0.2, label=r'Uncertainty $\pm R_n(f)$ ($W^{2,2}$)')

    # Static Baselines
    plt.plot(w1_counts_sub, w1_stat_qs_sub, '--s', color='#D55E00', alpha=0.4, label=r'Static $Q_n(f)$ ($W^{1,2}$)')
    plt.plot(w2_counts_sub, w2_stat_qs_sub, '--s', color='#009E73', alpha=0.4, label=r'Static $Q_n(f)$ ($W^{2,2}$)')

    # Monte Carlo Ground Truth
    plt.axhline(y=mc_w1p, color='black', linestyle='-.', alpha=0.8, label=r'MC True $W^{1,2}$')
    plt.axhline(y=mc_w2p, color='gray', linestyle='-.', alpha=0.8, label=r'MC True $W^{2,2}$')

    plt.xscale('log')
    plt.title("1D Localized - Integral Quadrature Convergence Q_n(f)", fontsize=30)
    plt.xlabel("Number of Grid Box", fontsize=30)
    plt.ylabel("Calculated Integral Value Q_n(f)", fontsize=30)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    save_path_integral = os.path.join(FIG_DIR, f"independent_integral_convergence_{LAYERS}.pdf")
    plt.tight_layout()
    plt.savefig(save_path_integral, dpi=300)
    print(f"Integral Convergence plot saved to {save_path_integral}")

    plt.show()

if __name__ == "__main__":
    run_independent_experiment()