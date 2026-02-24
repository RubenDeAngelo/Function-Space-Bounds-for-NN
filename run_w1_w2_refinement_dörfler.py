"""
run_combined_doerfler_1d_experiment.py

100-Run Statistical Analysis comparing W1p and W2p norms.
Includes interpolated tracking over grid cell counts, confidence intervals,
and clean, marker-free mean lines. Images saved as PDF.
"""

import os
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
LAYERS = [1, 200, 1]
ACTIVATION = 'sigmoid'
EPOCHS = 2000
NUM_RUNS = 100
MAX_ITER = 20
THETA = 0.3
P_NORM = 2.0

FIG_DIR = f"fig_combined_norms_waterfall"
os.makedirs(FIG_DIR, exist_ok=True)


# --- DATA GENERATION ---
def generate_localized_data(N=150):
    """Generates 1D Gaussian pulse data."""
    X = np.linspace(0, 2 * np.pi, N)
    y = np.exp(-20 * (X - np.pi)**2)
    return X.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)


# --- 1D UTILITIES ---
def split_points_1d(points, delta):
    new_points = []
    new_delta = delta / 2.0
    for x in points:
        new_points.append(x - np.array([new_delta]))
        new_points.append(x + np.array([new_delta]))
    return new_points, new_delta


def get_interpolated_stats(data_list, num_points=150):
    """
    Interpolates varying grid-cell counts onto a common log-spaced axis
    to calculate means and 95% confidence intervals.
    """
    all_x = np.concatenate([d[0] for d in data_list])
    min_x, max_x = np.min(all_x), np.max(all_x)

    x_common = np.logspace(np.log10(min_x), np.log10(max_x), num_points)

    interpolated_gaps = []
    for x, y in data_list:
        x_unique, idx = np.unique(x, return_index=True)
        y_unique = y[idx]
        y_interp = np.interp(x_common, x_unique, y_unique)
        interpolated_gaps.append(y_interp)

    interpolated_gaps = np.array(interpolated_gaps)

    mean_gaps = np.mean(interpolated_gaps, axis=0)
    lower_ci = np.percentile(interpolated_gaps, 2.5, axis=0)
    upper_ci = np.percentile(interpolated_gaps, 97.5, axis=0)

    lower_ci = np.maximum(lower_ci, 1e-16)

    return x_common, mean_gaps, lower_ci, upper_ci


# --- DÖRFLER REFINEMENT ---
def run_doerfler_refinement_1d(model, domain_start, domain_end, bound_fn, p=2.0, max_iter=30, theta=0.2):
    N_cells_init = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_cells_init) / 2.0

    centers_1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_cells_init).astype(np.float32)
    active_cells = []

    for x in centers_1d:
        cx = np.array([x])
        lb, ub = bound_fn(model, cx, delta_init, p=p)
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        active_cells.append({
            'center': cx, 'delta': delta_init,
            'low': lb, 'high': ub,
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
                cl, cu = bound_fn(model, cx, child_delta, p=p)
                new_cells.append({
                    'center': cx, 'delta': child_delta,
                    'low': cl, 'high': cu,
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
        c_w1, g_w1 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w1p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA)
        results['untrained']['w1'].append((c_w1, g_w1))

        c_w2, g_w2 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w2p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA)
        results['untrained']['w2'].append((c_w2, g_w2))

        # --- TRAINING ---
        model.train()
        train_model(model, X_tensor, y_tensor, epochs=EPOCHS)

        # --- TRAINED ---
        model.eval()
        c_w1, g_w1 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w1p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA)
        results['trained']['w1'].append((c_w1, g_w1))

        c_w2, g_w2 = run_doerfler_refinement_1d(model, domain_start, domain_end, compute_w2p_bounds, p=P_NORM, max_iter=MAX_ITER, theta=THETA)
        results['trained']['w2'].append((c_w2, g_w2))

        print("Done.")

    # --- PLOTTING FUNCTION ---
    def plot_with_shading(scenario_key, title_prefix):
        plt.figure(figsize=(10, 7))

        c_w1_dyn = '#D55E00' # Vermillion
        c_w2_dyn = '#009E73' # Bluish Green
        data = results[scenario_key]

        def plot_curve(data_list, color, label):
            x, mean, lower, upper = get_interpolated_stats(data_list)

            # Explicitly setting linestyle and removing markers
            plt.plot(x, mean, linestyle='-', marker='', color=color, linewidth=2.5, label=label)

            # Confidence interval shading
            plt.fill_between(x, lower, upper, color=color, alpha=0.2, label=f"95% CI ({label.split()[-1]})")

        # Plot Curves for W1 and W2 in the same plot
        plot_curve(data['w1'], c_w1_dyn, r'Adaptive $W^{1,p}$')
        plot_curve(data['w2'], c_w2_dyn, r'Adaptive $W^{2,p}$')

        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"{title_prefix}: Adaptive Refinement", fontsize=16)
        plt.xlabel("Number of Grid Cells", fontsize=14)
        plt.ylabel("Global Bound Gap", fontsize=14)
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        # UPDATED: Changed .png to .pdf
        path = os.path.join(FIG_DIR, f"adaptive_waterfall_{scenario_key}_{LAYERS}.pdf")
        plt.tight_layout()
        plt.savefig(path, dpi=300, format='pdf')
        print(f"Saved {path}")

    print("\nGenerating Plots...")
    plot_with_shading('untrained', 'Untrained Networks')
    plot_with_shading('trained', 'Trained Networks')
    plt.show()

if __name__ == "__main__":
    run_combined_experiment()