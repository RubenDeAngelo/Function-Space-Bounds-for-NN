"""
run_combined_doerfler_1d_experiment.py

100-Run Statistical Analysis comparing W1p and W2p norms.
Tracks Geometric Decay over Iteration steps (k), includes 95% confidence intervals.
Normalizes the empirical error bounds using individual Monte Carlo estimations for
both untrained and trained networks.
"""

import os
import sys
from contextlib import contextmanager

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import matplotlib.pyplot as plt

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
LAYERS = [1, 32, 32, 32, 1]
ACTIVATION = 'tanh'
EPOCHS = 3000
NUM_RUNS = 100
MAX_ITER = 30
THETA = 0.3
P_NORM = 2.0
JACOBIAN_METHOD = 'ibp'  # Enforce fast Interval Bound Propagation

FIG_DIR = f"fig_combined_norms_waterfall"
os.makedirs(FIG_DIR, exist_ok=True)


# --- UTILITIES ---
@contextmanager
def suppress_stdout():
    """Temporarily suppresses print statements during training."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def generate_localized_data(N=150):
    """Generates 1D Gaussian pulse data."""
    X = np.linspace(0, 2 * np.pi, N)
    y = np.exp(-20 * (X - np.pi)**2)
    return X.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


def compute_monte_carlo_sobolev_1d(model, num_samples=100000, p=2.0, domain=(0.0, 2 * np.pi)):
    """Computes exact analytical Sobolev bounds via 1D Monte Carlo sampling."""
    device = next(model.parameters()).device
    X = torch.empty(num_samples, 1, device=device).uniform_(domain[0], domain[1]).requires_grad_(True)
    volume = domain[1] - domain[0]

    Y = model(X)
    grad_outputs = torch.ones_like(Y)
    grad_Y = torch.autograd.grad(Y, X, grad_outputs=grad_outputs, create_graph=True)[0]
    hess_Y = torch.autograd.grad(grad_Y, X, grad_outputs=torch.ones_like(grad_Y), retain_graph=False)[0]

    phi_p = torch.abs(Y).pow(p).sum(dim=1)
    grad_p = torch.abs(grad_Y).pow(p).sum(dim=1)
    hess_p = torch.abs(hess_Y).pow(p).sum(dim=1)

    w1p_mc = volume * (phi_p + grad_p).mean().item()
    w2p_mc = volume * (phi_p + grad_p + hess_p).mean().item()

    return w1p_mc, w2p_mc


def split_points_1d(points, delta):
    new_points = []
    new_delta = delta / 2.0
    for x in points:
        new_points.append(x - np.array([new_delta]))
        new_points.append(x + np.array([new_delta]))
    return new_points, new_delta


def get_iteration_stats(data_list):
    """
    Stacks arrays of bounds by iteration index to calculate
    the mean and 95% confidence intervals across all runs.
    """
    max_len = max(len(d[1]) for d in data_list)

    padded_gaps = []
    for d in data_list:
        gaps = d[1]
        if len(gaps) < max_len:
            gaps = np.pad(gaps, (0, max_len - len(gaps)), mode='edge')
        padded_gaps.append(gaps)

    stacked = np.vstack(padded_gaps)

    mean_gaps = np.mean(stacked, axis=0)
    lower_ci = np.percentile(stacked, 2.5, axis=0)
    upper_ci = np.percentile(stacked, 97.5, axis=0)

    lower_ci = np.maximum(lower_ci, 1e-16)
    iterations = np.arange(max_len)

    return iterations, mean_gaps, lower_ci, upper_ci


# --- DÖRFLER REFINEMENT ---
def run_doerfler_refinement_1d(model, domain_start, domain_end, bound_fn, p=2.0, max_iter=30, theta=0.2, jac_method='ibp'):
    N_cells_init = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_cells_init) / 2.0

    centers_1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_cells_init).astype(np.float32)
    active_cells = []

    for x in centers_1d:
        cx = np.array([x])
        lb, ub = bound_fn(model, cx, delta_init, p=p, jacobian_method=jac_method)
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        active_cells.append({
            'center': cx, 'delta': delta_init,
            'error': error_mass
        })

    counts = []
    gaps = []

    for i in range(max_iter + 1):
        counts.append(len(active_cells))
        global_gap = sum(c['error'] for c in active_cells)
        gaps.append(global_gap)

        if i == max_iter: break

        total_error = global_gap
        active_cells.sort(key=lambda c: c['error'], reverse=True)

        marked_set, unmarked_set = [], []
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
            child_centers, child_delta = split_points_1d([parent['center']], parent['delta'])
            for cx in child_centers:
                cl, cu = bound_fn(model, cx, child_delta, p=p, jacobian_method=jac_method)
                new_cells.append({
                    'center': cx, 'delta': child_delta,
                    'error': 0.5 * ((cu ** p) - (cl ** p))
                })
        active_cells = unmarked_set + new_cells

    return np.array(counts), np.array(gaps)


# --- MAIN EXPERIMENT ---
def run_combined_experiment():
    domain_start, domain_end = 0.0, 2 * np.pi
    print(f"--- Starting Experiment ({NUM_RUNS} seeds) [Waterfall Metric] ---")

    results = {
        'untrained': {'w1': [], 'w2': []},
        'trained':   {'w1': [], 'w2': []}
    }

    X_train, y_train = generate_localized_data(N=150)
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for i in range(NUM_RUNS):
        print(f"Seed {i + 1}/{NUM_RUNS}...", end=" ", flush=True)

        torch.manual_seed(i)
        np.random.seed(i)
        model = FlexibleNet(LAYERS, activation=ACTIVATION)

        # --- UNTRAINED ---
        model.eval()

        # Calculate MC Truth for normalization on Untrained Network
        mc_w1_untrained, mc_w2_untrained = compute_monte_carlo_sobolev_1d(model, num_samples=50000, p=P_NORM, domain=(domain_start, domain_end))

        c_w1, g_w1 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w1p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA, jac_method=JACOBIAN_METHOD)
        results['untrained']['w1'].append((c_w1, g_w1 / mc_w1_untrained))

        c_w2, g_w2 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w2p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA, jac_method=JACOBIAN_METHOD)
        results['untrained']['w2'].append((c_w2, g_w2 / mc_w2_untrained))

        # --- TRAINING ---
        model.train()
        with suppress_stdout():
            train_model(model, X_tensor, y_tensor, epochs=EPOCHS)

        # --- TRAINED ---
        model.eval()

        # Calculate MC Truth for normalization on Trained Network
        mc_w1_trained, mc_w2_trained = compute_monte_carlo_sobolev_1d(model, num_samples=50000, p=P_NORM, domain=(domain_start, domain_end))

        c_w1, g_w1 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w1p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA, jac_method=JACOBIAN_METHOD)
        results['trained']['w1'].append((c_w1, g_w1 / mc_w1_trained))

        c_w2, g_w2 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w2p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA, jac_method=JACOBIAN_METHOD)
        results['trained']['w2'].append((c_w2, g_w2 / mc_w2_trained))

        print("Done.")

    # --- PLOTTING FUNCTION ---
    def plot_with_shading(scenario_key, title_prefix, is_normalized=False):
        plt.figure(figsize=(10, 7))

        c_w1_dyn = '#D55E00'
        c_w2_dyn = '#009E73'
        data = results[scenario_key]

        def plot_curve(data_list, color, label, marker):
            x, mean, lower, upper = get_iteration_stats(data_list)

            # Plot the mean empirical curve
            plt.plot(x, mean, linestyle='-', marker=marker, color=color, linewidth=2.5, markersize=8, label=label)

            # Confidence interval shading
            plt.fill_between(x, lower, upper, color=color, alpha=0.2)

        # Plot Curves for W1 and W2 in the same plot
        plot_curve(data['w1'], c_w1_dyn, r'Adaptive $W^{1,2}$', 'o')
        plot_curve(data['w2'], c_w2_dyn, r'Adaptive $W^{2,2}$', 's')

        plt.xscale('linear')
        plt.yscale('log')
        plt.title(rf"{title_prefix}: Geometric Decay | $\theta={THETA}$", fontsize=gloabal_size_para + 2)
        plt.xlabel("Iteration Step $k$", fontsize=gloabal_size_para)

        y_label = "Normalized Global Bound Gap" if is_normalized else "Global Bound Gap"
        plt.ylabel(y_label, fontsize=gloabal_size_para)

        plt.grid(True, which="both", alpha=0.3)
        plt.legend(loc='best', fontsize=gloabal_size_para - 4)

        path = os.path.join(FIG_DIR, f"adaptive_waterfall_{scenario_key}_{LAYERS}.pdf")
        plt.tight_layout()
        plt.savefig(path, dpi=300, format='pdf')
        print(f"Saved {path}")

    print("\nGenerating Plots...")
    # BOTH are now normalized
    plot_with_shading('untrained', 'Untrained Networks', is_normalized=True)
    plot_with_shading('trained', 'Trained Networks', is_normalized=True)

if __name__ == "__main__":
    run_combined_experiment()