"""
run_combined_convergence_experiments.py

Computes the convergence rates for W1p, W2p, and Lp norms using Dörfler marking.
Normalizes the empirical error bounds using Monte Carlo estimations and plots
the convergence against theoretical rates. Includes combined overview plots.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
import logging
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# --- Global Plotting Configuration ---
global_size_para = 25
plt.rcParams.update({
    'font.size': global_size_para,
    'axes.titlesize': global_size_para,
    'axes.labelsize': global_size_para,
    'xtick.labelsize': global_size_para,
    'ytick.labelsize': global_size_para,
    'legend.fontsize': 30,
    'figure.titlesize': global_size_para
})

# --- Imports for Sobolev (W1p, W2p) ---
from flex_model import FlexibleNet
from w1p_bounds import compute_w1p_bounds
from w2p_bounds import compute_w2p_bounds
from run_doerfler_theorem_compliant_2d import compute_monte_carlo_sobolev

# --- Imports for Lp ---
from adaptive_lp_core import (
    ReLUNetwork,
    init_many_affine_pieces,
    AdaptiveLpNorm,
    monte_carlo_lp_norm
)

# --- Configuration ---
FIG_DIR = "fig_combined_convergence"
os.makedirs(FIG_DIR, exist_ok=True)

# SPECIFIC THETAS FOR EACH METHOD
THETA_W1P = 0.25
THETA_W2P = 0.25
THETA_LP = 0.4

MAX_ITER = 30
P_NORM = 2.0
D_DIM = 2
MC_SAMPLES = 500000

# Sobolev Model Config
LAYERS = [2, 32, 32, 32, 32, 1]
ACTIVATION = 'sigmoid'
SOBOLEV_MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}.pth"

# Theoretical rho values
RHO_SOBOLEV = 0.25  # Splitting a 2D box into 4
RHO_LP = 0.50       # Bisecting the largest dimension


@contextmanager
def silence_everything(disable_logging: bool = True, disable_warnings: bool = True):
    prev_disable_level = logging.root.manager.disable
    prev_warning_filters = warnings.filters[:]
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            if disable_logging: logging.disable(logging.CRITICAL)
            if disable_warnings: warnings.simplefilter("ignore")
            yield
        finally:
            if disable_warnings: warnings.filters = prev_warning_filters
            if disable_logging: logging.disable(prev_disable_level)


# --- Utility Functions ---
def split_points(points, delta):
    new_points = []
    new_delta = delta / 2.0
    offset_combinations = list(itertools.product([-1, 1], repeat=D_DIM))
    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape) * new_delta
            new_points.append(x + offset)
    return new_points, new_delta


# --- Sobolev Run Logic ---
def run_sobolev_doerfler(model, initial_centers, initial_delta, max_iter, theta, p, norm_type):
    print(f"  -> Running {norm_type.upper()} Refinement (theta={theta})...")
    active_cells = []
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    for x in initial_centers:
        lb, ub = bound_fn(model, x, initial_delta, p=p)
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        active_cells.append({'center': x, 'delta': initial_delta, 'error': error_mass})

    sizes, gaps = [], []

    for i in range(max_iter + 1):
        global_gap = sum(c['error'] for c in active_cells)
        sizes.append(len(active_cells))
        gaps.append(global_gap)

        if i == max_iter: break

        active_cells.sort(key=lambda c: c['error'], reverse=True)
        marked_set, unmarked_set = [], []
        current_sum = 0.0

        for cell in active_cells:
            if current_sum < theta * global_gap:
                marked_set.append(cell)
                current_sum += cell['error']
            else:
                unmarked_set.append(cell)

        new_cells = []
        for parent in marked_set:
            child_centers, child_delta = split_points([parent['center']], parent['delta'])
            for cx in child_centers:
                cl, cu = bound_fn(model, cx, child_delta, p=p)
                new_cells.append({
                    'center': cx, 'delta': child_delta,
                    'error': 0.5 * ((cu ** p) - (cl ** p))
                })
        active_cells = unmarked_set + new_cells

    return {'sizes': np.array(sizes), 'gaps': np.array(gaps)}


# --- Individual Plotting Functions ---
def plot_geometric_decay(results, norm_type, theta, rho):
    gaps = results['gaps']
    iterations = np.arange(len(gaps))

    plt.figure(figsize=(10, 8))
    plt.plot(iterations, gaps, '-s', color='#85C57A', linewidth=2.5, markersize=8, label=rf'Empirical $\theta={theta}$')

    q = 1.0 - theta * (1.0 - rho)
    theory_gaps = gaps[0] * (q ** iterations)
    plt.plot(iterations, theory_gaps, '--k', linewidth=2.5, label=rf'Theory $q={q:.3f}$')

    plt.yscale('log')
    plt.title(f"Geometric Error Decay ({norm_type.upper()})", fontsize=global_size_para)
    plt.xlabel("Adaptation Iteration $k$", fontsize=global_size_para)
    plt.ylabel(r"Global Normalized Error $R_k / R_{MC}$", fontsize=global_size_para)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"{norm_type}_geometric_decay.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()


def plot_n_convergence(results, norm_type, d):
    sizes = results['sizes']
    gaps = results['gaps']

    plt.figure(figsize=(10, 8))
    plt.plot(sizes, gaps, '-o', color='#D55E00', linewidth=2.5, markersize=8, label=rf'Empirical Rate')

    n0, e0 = sizes[0], gaps[0]
    x_theory = np.linspace(n0, sizes[-1], 100)
    rate = -1.0 / d
    plt.plot(x_theory, e0 * (x_theory / n0) ** rate, 'k--', linewidth=2.5, label=rf'Reference $\mathcal{{O}}(N^{{{rate:.2f}}})$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Adaptive Convergence Rate ({norm_type.upper()})", fontsize=global_size_para)
    plt.xlabel("Number of Grid Boxes $N$", fontsize=global_size_para)
    plt.ylabel(r"Normalized Error Bound $R_k / R_{MC}$", fontsize=global_size_para)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"{norm_type}_n_convergence.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()


# --- Combined Plotting Functions ---
def plot_combined_geometric_decay(all_results):
    plt.figure(figsize=(12, 9))

    styles = {
        'w1p': {'color': '#1f77b4', 'marker': 'o', 'label': r'$W^{1,p}$'},
        'w2p': {'color': '#ff7f0e', 'marker': 's', 'label': r'$W^{2,p}$'},
        'lp': {'color': '#2ca02c', 'marker': '^', 'label': r'$L^p$'}
    }

    for norm_type, data in all_results.items():
        gaps = data['res']['gaps']
        theta = data['theta']
        rho = data['rho']
        iterations = np.arange(len(gaps))

        style = styles[norm_type]

        # Plot Empirical
        plt.plot(iterations, gaps, linestyle='-', marker=style['marker'], color=style['color'],
                 linewidth=2.5, markersize=7, label=f"Empirical {style['label']} ($\\theta={theta}$)")

        # Plot Theoretical
        q = 1.0 - theta * (1.0 - rho)
        theory_gaps = gaps[0] * (q ** iterations)
        plt.plot(iterations, theory_gaps, linestyle='--', color=style['color'],
                 linewidth=2, alpha=0.8, label=f"Theory {style['label']} ($q={q:.3f}$)")

    plt.yscale('log')
    plt.title("Combined Geometric Error Decay", fontsize=global_size_para)
    plt.xlabel("Adaptation Iteration $k$", fontsize=global_size_para)
    plt.ylabel(r"Normalized Error Bound $R_k / R_{MC}$", fontsize=global_size_para)
    plt.grid(True, which="both", alpha=0.3)

    # --- UPDATED LEGEND ---
    # Set fontsize to global_size_para (or increase it, e.g., global_size_para + 5)
    # Use markerscale to scale up the symbols inside the legend box
    plt.legend(loc='best', fontsize=global_size_para-5, markerscale=1.5)

    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "combined_geometric_decay.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()

def plot_combined_n_convergence(all_results, d):
    plt.figure(figsize=(12, 9))

    styles = {
        'w1p': {'color': '#1f77b4', 'marker': 'o', 'label': r'$W^{1,p}$'},
        'w2p': {'color': '#ff7f0e', 'marker': 's', 'label': r'$W^{2,p}$'},
        'lp':  {'color': '#2ca02c', 'marker': '^', 'label': r'$L^p$'}
    }

    max_size = 0
    for norm_type, data in all_results.items():
        sizes = data['res']['sizes']
        gaps = data['res']['gaps']
        style = styles[norm_type]

        max_size = max(max_size, np.max(sizes))

        plt.plot(sizes, gaps, linestyle='-', marker=style['marker'], color=style['color'],
                 linewidth=2.5, markersize=7, label=f"{style['label']} Rate")

    # Use W1P's first point as the anchor for the theoretical line (arbitrary choice for reference)
    base_sizes = all_results['w1p']['res']['sizes']
    base_gaps = all_results['w1p']['res']['gaps']
    n0, e0 = base_sizes[0], base_gaps[0]

    x_theory = np.linspace(n0, max_size, 100)
    rate = -1.0 / d
    plt.plot(x_theory, e0 * (x_theory / n0) ** rate, 'k--', linewidth=2.5,
             label=rf'Reference $\mathcal{{O}}(N^{{{rate:.2f}}})$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Combined Adaptive Convergence Rate", fontsize=global_size_para)
    plt.xlabel("Number of Grid Boxes $N$", fontsize=global_size_para)
    plt.ylabel(r"Normalized Error Bound $R_k / R_{MC}$", fontsize=global_size_para)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "combined_n_convergence.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()


# --- Main Execution ---
def main():
    print("=== Starting Combined Convergence Experiments ===")

    # 1. Initialize Sobolev Model
    print("\n--- Setup Sobolev Model ---")
    sobolev_model = FlexibleNet(LAYERS, activation=ACTIVATION)
    if os.path.exists(SOBOLEV_MODEL_PATH):
        sobolev_model.load_state_dict(torch.load(SOBOLEV_MODEL_PATH))
    else:
        print(f"Model {SOBOLEV_MODEL_PATH} not found. Train it first!")
        sys.exit(1)
    sobolev_model.eval()

    # Initial Grid for Sobolev
    domain_start, domain_end = -1.5, 1.5
    N_side = 2
    delta_init = ((domain_end - domain_start) / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # 2. Initialize Lp Model
    print("\n--- Setup Lp Model ---")
    lp_model = ReLUNetwork(input_dim=D_DIM, output_dim=1, width=32, depth=2, init_method="he_normal")
    init_many_affine_pieces(lp_model)
    lp_model.eval()

    # 3. Compute Monte Carlo Truths
    mc_w1p, mc_w2p = compute_monte_carlo_sobolev(sobolev_model, num_samples=MC_SAMPLES, p=P_NORM, domain=(domain_start, domain_end))

    mc_lp = monte_carlo_lp_norm(lp_model, d=D_DIM, p=P_NORM, n_samples=MC_SAMPLES)
    print(f"LP MC Truth: {mc_lp:.6f}")

    # 4. Run Experiments
    print("\n=== Running Experiments ===")

    # W1p
    res_w1p = run_sobolev_doerfler(sobolev_model, centers_init, delta_init, MAX_ITER, THETA_W1P, P_NORM, 'w1p')
    res_w1p['gaps'] = res_w1p['gaps'] / mc_w1p  # Normalize

    # W2p
    res_w2p = run_sobolev_doerfler(sobolev_model, centers_init, delta_init, MAX_ITER, THETA_W2P, P_NORM, 'w2p')
    res_w2p['gaps'] = res_w2p['gaps'] / mc_w2p  # Normalize

    # Lp
    print(f"  -> Running LP Refinement (theta={THETA_LP})...")
    lp_runner = AdaptiveLpNorm(d=D_DIM, neural_net=lp_model, is_relu=True)
    with silence_everything():
        lp_runner.norm(p=P_NORM, m=MAX_ITER, theta=THETA_LP, refiner="max", eta_type="fval")

    res_lp = {
        'sizes': np.array(lp_runner.out_adap['size_hist']),
        'gaps': np.array(lp_runner.out_adap['post_hist']) / mc_lp  # Normalize
    }

    # Structure data for combined plotting
    all_results = {
        'w1p': {'res': res_w1p, 'theta': THETA_W1P, 'rho': RHO_SOBOLEV},
        'w2p': {'res': res_w2p, 'theta': THETA_W2P, 'rho': RHO_SOBOLEV},
        'lp':  {'res': res_lp,  'theta': THETA_LP,  'rho': RHO_LP}
    }

    # 5. Generate Plots
    print("\n=== Generating Plots ===")

    # Individual Plots
    for norm_type, data in all_results.items():
        print(f"Plotting individual charts for {norm_type.upper()}...")
        plot_geometric_decay(data['res'], norm_type, data['theta'], data['rho'])
        plot_n_convergence(data['res'], norm_type, D_DIM)

    # Combined Plots
    print("Plotting combined charts...")
    plot_combined_geometric_decay(all_results)
    plot_combined_n_convergence(all_results, D_DIM)

    print(f"\nAll done! Plots saved to '{FIG_DIR}' directory.")

if __name__ == "__main__":
    main()