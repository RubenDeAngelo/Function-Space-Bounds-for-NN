"""
Fig. 1 (left): Combined geometric decay for W^{1,p}, W^{2,p}, and L^p norms.
Smooth-coin 2D target. Sobolev model uses tanh + weight decay; Lp model uses ReLU.
Errors normalized by individual Monte Carlo ground truths.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import logging
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr

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

from lib_nn_model import FlexibleNet
from lib_training import train_model
from lib_w1p_enclosures import compute_w1p_bounds
from lib_w2p_enclosures import compute_w2p_bounds
from exp_sobolev_2d_disk_heatmaps import compute_monte_carlo_sobolev

from lib_adaquad_lp import (
    ReLUNetwork,
    AdaptiveLpNorm,
    monte_carlo_lp_norm
)

FIG_DIR = "fig_combined_convergence"
os.makedirs(FIG_DIR, exist_ok=True)

THETA_W1P = 0.35
THETA_W2P = 0.35
THETA_LP = 0.4
JACOBIAN_METHOD = 'ibp'

MAX_ITER = 30
P_NORM = 2.0
D_DIM = 2
MC_SAMPLES = 500000

EPOCHS = 10000
N_SAMPLES = 5000
EPSILON_TRANSITION = 0.5

SOBOLEV_LAYERS = [2, 32, 32, 32, 1]
SOBOLEV_ACTIVATION = 'tanh'
WEIGHT_DECAY = 1e-3
SOBOLEV_MODEL_PATH = f"model_smooth_coin_2d_{SOBOLEV_LAYERS}_{EPSILON_TRANSITION}_wd{WEIGHT_DECAY}_{SOBOLEV_ACTIVATION}.pth"

LP_LAYERS = [2, 32, 32, 32, 1]
LP_ACTIVATION = 'relu'
LP_MODEL_PATH = f"model_smooth_coin_2d_{LP_LAYERS}_{EPSILON_TRANSITION}_wd0.0_{LP_ACTIVATION}.pth"

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

def h(t):
    return np.where(t > 0, np.exp(-1.0 / np.maximum(t, 1e-16)), 0.0)

def smooth_disk(x, y, epsilon=0.1):
    r = np.sqrt(x ** 2 + y ** 2)
    r_inner = 1.0
    r_outer = 1.0 + epsilon
    term1 = h(r_outer - r)
    term2 = h(r - r_inner)
    denom = term1 + term2
    val = np.zeros_like(r)
    val[r <= r_inner] = 1.0
    mask = (r > r_inner) & (r < r_outer)
    transition_vals = np.divide(term1, denom, where=denom != 0)
    val[mask] = transition_vals[mask]
    return val

def generate_balanced_2d_coin_data(N=3000, epsilon=0.1, transition_ratio=0.5):
    N_transition = int(N * transition_ratio)
    N_uniform = N - N_transition
    low, high = -1.5, 1.5
    X_uni = np.random.uniform(low, high, (N_uniform, 2)).astype(np.float32)
    X = X_uni
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return X, y.reshape(-1, 1).astype(np.float32)

def split_points(points, delta):
    new_points = []
    new_delta = delta / 2.0
    offset_combinations = list(itertools.product([-1, 1], repeat=D_DIM))
    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape) * new_delta
            new_points.append(x + offset)
    return new_points, new_delta

def get_flattened_layers(model):
    """Recursively searches for native Linear/ReLU layers to avoid Nested-Sequentials."""
    layers = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            layers.extend(get_flattened_layers(module))
        else:
            layers.append(module)
    return layers

def extract_and_plot_heatmap(partition, iteration, domain_start, domain_width, p, scale_factor, epsilon=0.5):
    """Extracts internal boxes and plots a heatmap of the local error bounds."""
    ids = list(partition.leaves)
    if not ids: return

    # Retrieve coordinates in the library's internal [0,1] space
    slots = torch.tensor([partition.id2slot[i] for i in ids], dtype=torch.long, device=partition.storage.device)
    cr = partition.storage.index_select(0, slots).cpu().numpy()

    # Map back to the original [-1.5, 1.5] domain
    centers = domain_start + domain_width * cr[:, :, 0]
    radii = domain_width * cr[:, :, 1]

    # Scale errors (remains in L_p^p space)
    raw_etas = np.array([partition.eta[i] for i in ids])
    etas = raw_etas * scale_factor

    fig, ax = plt.subplots(figsize=(12, 12))
    min_err, max_err = etas.min(), etas.max()
    if max_err == min_err: max_err += 1e-9

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_err, vmax=max_err)

    for i in range(len(ids)):
        cx, cy = centers[i, 0], centers[i, 1]
        rx, ry = radii[i, 0], radii[i, 1]

        x0, y0 = cx - rx, cy - ry
        width, height = 2.0 * rx, 2.0 * ry

        color = cmap(norm(etas[i]))
        rect = patches.Rectangle((x0, y0), width, height,
                                 linewidth=0.5, edgecolor='white', facecolor=color, alpha=0.9)
        ax.add_patch(rect)

    # Draw smooth disk boundaries
    circle = plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--', linewidth=2.0, label='Inner Radius')
    circle_outer = plt.Circle((0, 0), 1.0 + epsilon, color='orange', fill=False, linestyle='--', linewidth=2.0,
                              label='Outer Radius')
    ax.add_patch(circle)
    ax.add_patch(circle_outer)

    domain_end = domain_start + domain_width
    ax.set_xlim(domain_start, domain_end)
    ax.set_ylim(domain_start, domain_end)
    ax.set_title(
        rf"Adaptive Grid $L_{{{int(p)}}}^{{{int(p)}}}$ - Iteration {iteration}" + "\n" + rf"$N={len(ids)}$ Boxes",
        fontsize=global_size_para + 4)
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(rf'Local Gap ($L_{{{int(p)}}}^{{{int(p)}}}$)', fontsize=global_size_para)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "lp_heatmap_combined_run.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

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
        iterations = np.arange(len(gaps))

        style = styles[norm_type]

        plt.plot(iterations, gaps, linestyle='-', marker=style['marker'], color=style['color'],
                 linewidth=2.5, markersize=7, label=rf"{style['label']} ($\theta={theta}$)")

    plt.yscale('log')
    plt.title("Combined Geometric Error Decay", fontsize=global_size_para)
    plt.xlabel("Adaptation Iteration $k$", fontsize=global_size_para)
    plt.ylabel(r"Normalized Error Bound", fontsize=global_size_para)
    plt.grid(True, which="both", alpha=0.3)

    plt.legend(loc='best', fontsize=global_size_para-5, markerscale=1.5)
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "combined_geometric_decay.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    plt.close()

def run_sobolev_doerfler(model, initial_centers, initial_delta, max_iter, theta, p, norm_type, jac_method):
    print(f"  -> Running {norm_type.upper()} Refinement (theta={theta}, method={jac_method.upper()})...")
    active_cells = []
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    for x in initial_centers:
        lb, ub = bound_fn(model, x, initial_delta, p=p, jacobian_method=jac_method)
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        active_cells.append({'center': x, 'delta': initial_delta, 'error': error_mass})

    sizes, gaps = [], []

    for i in range(max_iter + 1):
        global_gap = sum(c['error'] for c in active_cells)
        print(f"    Iter {i:<3} | Cells: {len(active_cells):<6} | Gap: {global_gap:.6e}")
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
                cl, cu = bound_fn(model, cx, child_delta, p=p, jacobian_method=jac_method)
                new_cells.append({
                    'center': cx, 'delta': child_delta,
                    'error': 0.5 * ((cu ** p) - (cl ** p))
                })
        active_cells = unmarked_set + new_cells

    return {'sizes': np.array(sizes), 'gaps': np.array(gaps)}

def main():
    print("=== Starting Combined Convergence Experiments ===")

    X_np, y_np = generate_balanced_2d_coin_data(N=N_SAMPLES, epsilon=EPSILON_TRANSITION)

    sobolev_model = FlexibleNet(SOBOLEV_LAYERS, activation=SOBOLEV_ACTIVATION)
    if os.path.exists(SOBOLEV_MODEL_PATH):
        print(f"Loading existing Sobolev model from {SOBOLEV_MODEL_PATH}...")
        sobolev_model.load_state_dict(torch.load(SOBOLEV_MODEL_PATH))
    else:
        print(f"Training Sobolev model on {N_SAMPLES} samples with Weight Decay {WEIGHT_DECAY}...")
        train_model(sobolev_model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=EPOCHS, weight_decay=WEIGHT_DECAY)
        torch.save(sobolev_model.state_dict(), SOBOLEV_MODEL_PATH)
    sobolev_model.eval()

    lp_model = ReLUNetwork(input_dim=D_DIM, output_dim=1, width=32, depth=3, init_method="he_normal")
    if os.path.exists(LP_MODEL_PATH):
        print(f"Loading existing Lp model from {LP_MODEL_PATH}...")
        lp_model.load_state_dict(torch.load(LP_MODEL_PATH))
    else:
        print(f"Training Lp model on {N_SAMPLES} samples with NO Weight Decay...")
        train_model(lp_model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=EPOCHS, weight_decay=0.0)
        torch.save(lp_model.state_dict(), LP_MODEL_PATH)
    lp_model.eval()

    # Wrap Lp model with an affine scaling layer to map [domain_start, domain_end]^d -> [0,1]^d
    domain_start, domain_end = -1.5, 1.5
    domain_width = domain_end - domain_start
    volume_scale = domain_width ** D_DIM

    scaling_layer = nn.Linear(D_DIM, D_DIM)
    scaling_layer.weight.data = torch.eye(D_DIM) * domain_width
    scaling_layer.bias.data = torch.full((D_DIM,), float(domain_start))
    scaling_layer.requires_grad_(False)

    wrapped_lp_model = nn.Sequential(scaling_layer, *get_flattened_layers(lp_model))
    wrapped_lp_model.eval()

    print("\n--- Computing Monte Carlo Ground Truths ---")
    mc_w1p, mc_w2p = compute_monte_carlo_sobolev(sobolev_model, num_samples=MC_SAMPLES, p=P_NORM, domain=(domain_start, domain_end))

    # monte_carlo_lp_norm returns the L_p norm; convert to L_p^p integral space
    raw_mc_lp = monte_carlo_lp_norm(wrapped_lp_model, d=D_DIM, p=P_NORM, n_samples=MC_SAMPLES)
    mc_lp = (raw_mc_lp ** P_NORM) * volume_scale
    print(f"LP MC Truth (Integral): {mc_lp:.6f}")

    N_side = 2
    delta_init = (domain_width / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    res_w1p = run_sobolev_doerfler(sobolev_model, centers_init, delta_init, MAX_ITER+2, THETA_W1P, P_NORM, 'w1p', JACOBIAN_METHOD)
    res_w1p['gaps'] = res_w1p['gaps'] / mc_w1p

    res_w2p = run_sobolev_doerfler(sobolev_model, centers_init, delta_init, MAX_ITER+4, THETA_W2P, P_NORM, 'w2p', JACOBIAN_METHOD)
    res_w2p['gaps'] = res_w2p['gaps'] / mc_w2p

    print(f"  -> Running LP Refinement (theta={THETA_LP})...")
    lp_runner = AdaptiveLpNorm(d=D_DIM, neural_net=wrapped_lp_model, is_relu=True)
    with silence_everything():
        lp_runner.norm(p=P_NORM, m=MAX_ITER+2, theta=THETA_LP, refiner="max", eta_type="fval")

    # Scale post_hist from L_p norm space to L_p^p integral space, then normalize
    raw_lp_gaps = np.array(lp_runner.out_adap['post_hist']) * volume_scale
    normalized_lp_gaps = raw_lp_gaps / mc_lp

    res_lp = {
        'sizes': np.array(lp_runner.out_adap['size_hist']),
        'gaps': normalized_lp_gaps
    }

    all_results = {
        'w1p': {'res': res_w1p, 'theta': THETA_W1P},
        'w2p': {'res': res_w2p, 'theta': THETA_W2P},
        'lp':  {'res': res_lp,  'theta': THETA_LP}
    }

    plot_combined_geometric_decay(all_results)
    extract_and_plot_heatmap(lp_runner.partition_adaptive, MAX_ITER, domain_start, domain_width, P_NORM, volume_scale, epsilon=EPSILON_TRANSITION)

    print(f"\nAll done! Plots saved to '{FIG_DIR}'.")

if __name__ == "__main__":
    main()