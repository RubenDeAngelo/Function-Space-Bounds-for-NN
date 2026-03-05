"""
run_lp_bound_smooth_coin_2d_heatmaps.py

Standalone script to compute and plot the convergence rate for the Lp norm
on the 2D Smooth Coin function using the dedicated AdaptiveLpNorm library.
- Integrates affine domain scaling as a native nn.Linear layer to ALLOW the
  internal ReLU-optimizations (boxes_inside_activation_region) to work!
- Extracts and plots heatmaps dynamically from the internal library state.
- Normalizes the empirical error bound using Monte Carlo estimations.
- Plots the geometric decay over iterations (k) against the theoretical rate q^k.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from contextlib import contextmanager, redirect_stdout, redirect_stderr

# --- Global Plotting Configuration ---
global_size_para = 25
plt.rcParams.update({
    'font.size': global_size_para,
    'axes.titlesize': global_size_para,
    'axes.labelsize': global_size_para,
    'xtick.labelsize': global_size_para,
    'ytick.labelsize': global_size_para,
    'legend.fontsize': global_size_para - 4,
    'figure.titlesize': global_size_para
})

# --- Imports ---
import adaptive_lp_core  # Import module to allow monkey-patching
from flex_model import FlexibleNet
from training import train_model
from adaptive_lp_core import AdaptiveLpNorm

# --- Configuration ---
LAYERS = [2, 32, 32, 16, 1]
ACTIVATION = 'relu'  # Set to 'relu' to utilize the full speedup!
WEIGHT_DECAY = 0.0
EPOCHS = 10000
N_SAMPLES = 5000
EPSILON_TRANSITION = 0.5

P_NORM = 2.0
D_DIM = 2
MAX_ITER = 50
THETA = 0.30
REFINER = "max" # Bisection strategy
RHO = 0.50      # Theoretical error reduction for 1D bisection in 2D space
HEATMAP_STEP = 5

# Domain Scaling Config
DOMAIN_START = -1.5
DOMAIN_END = 1.5
DOMAIN_WIDTH = DOMAIN_END - DOMAIN_START       # 3.0
VOLUME = DOMAIN_WIDTH ** D_DIM                 # 3.0^2 = 9.0
SCALE_FACTOR = VOLUME ** (1.0 / P_NORM)        # 9.0^(1/2) = 3.0 (for L2 norm)

FIG_DIR = f"fig_lp_smooth_coin_heatmaps_{LAYERS}_{ACTIVATION}"
os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}_{EPSILON_TRANSITION}_wd{WEIGHT_DECAY}_{ACTIVATION}.pth"


@contextmanager
def silence_everything(disable_logging: bool = True, disable_warnings: bool = True):
    """Suppresses warnings and console spam from external bounding libraries."""
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


# --- Data Generation ---
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

def generate_balanced_2d_coin_data(N=3000, epsilon=0.1):
    low, high = DOMAIN_START, DOMAIN_END
    X = np.random.uniform(low, high, (N, 2)).astype(np.float32)
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return X, y.reshape(-1, 1).astype(np.float32)


# --- Monte Carlo Ground Truth ---
def compute_monte_carlo_lp(model, num_samples=500000, p=2.0, domain=(DOMAIN_START, DOMAIN_END)):
    print(f"\n--- Computing Monte Carlo Ground Truth (N={num_samples}) ---")
    model.eval()
    device = next(model.parameters()).device

    X = torch.empty(num_samples, 2, device=device).uniform_(domain[0], domain[1])
    volume = (domain[1] - domain[0]) ** 2

    Y = model(X)
    lp_integrand = torch.abs(Y).pow(p).sum(dim=1)

    lp_integral = volume * lp_integrand.mean().item()
    lp_mc = lp_integral ** (1.0 / p)

    print(f"MC Integral Value  : {lp_integral:.6f}")
    print(f"MC L{int(p)} True Norm : {lp_mc:.6f}")
    return lp_mc


# --- Heatmap Extraction Logic ---
def extract_and_plot_heatmap(partition, iteration):
    """Extracts internal box state from the library's Partition object and plots it."""
    ids = list(partition.leaves)
    if not ids: return

    slots = torch.tensor([partition.id2slot[i] for i in ids], dtype=torch.long, device=partition.storage.device)
    cr = partition.storage.index_select(0, slots).cpu().numpy()

    # Map back to original domain
    centers = DOMAIN_START + DOMAIN_WIDTH * cr[:, :, 0]
    radii = DOMAIN_WIDTH * cr[:, :, 1]

    # Scale errors
    etas = np.array([partition.eta[i] for i in ids]) * SCALE_FACTOR

    fig, ax = plt.subplots(figsize=(12, 12))

    min_err, max_err = etas.min(), etas.max()
    if max_err == min_err: max_err += 1e-9

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_err, vmax=max_err)

    for i in range(len(ids)):
        cx, cy = centers[i, 0], centers[i, 1]
        rx, ry = radii[i, 0], radii[i, 1]

        x0 = cx - rx
        y0 = cy - ry
        width = 2.0 * rx
        height = 2.0 * ry

        color = cmap(norm(etas[i]))
        rect = patches.Rectangle((x0, y0), width, height,
                                 linewidth=0.5, edgecolor='white', facecolor=color, alpha=0.9)
        ax.add_patch(rect)

    circle = plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--', linewidth=1.5, label='Inner Radius')
    circle_outer = plt.Circle((0, 0), 1.0 + EPSILON_TRANSITION, color='orange', fill=False, linestyle='--', linewidth=1.5, label='Outer Radius')
    ax.add_patch(circle)
    ax.add_patch(circle_outer)

    ax.set_xlim(DOMAIN_START, DOMAIN_END)
    ax.set_ylim(DOMAIN_START, DOMAIN_END)
    ax.set_title(f"L{int(P_NORM)} Adaptive Grid - Iteration {iteration}\nN={len(ids)} Boxes", fontsize=30)
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Local Gap', fontsize=30)

    save_name = f"heatmap_lp_compliant_iter_{iteration:03d}.pdf"
    save_path = os.path.join(FIG_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [Plot] Saved heatmap to {save_path}")


# --- Plotting Convergence ---
def plot_lp_geometric_decay(iterations, gaps, theta, rho):
    plt.figure(figsize=(12, 9))

    # Calculate exact q value
    q = 1.0 - theta * (1.0 - rho)
    theory_decay = gaps[0] * (q ** iterations)

    # Plot Empirical Error
    plt.plot(iterations, gaps, linestyle='-', marker='^', color='#2ca02c',
             linewidth=2.5, markersize=10, label=rf"Empirical $L^{{{int(P_NORM)}}}$")

    # Plot Theory Line
    plt.plot(iterations, theory_decay, '--k',
             linewidth=2.5, alpha=0.7, label=rf'Theory $q={q:.4f}$')

    plt.yscale('log')
    plt.title(rf"Geometric Convergence Decay ($L^{{{int(P_NORM)}}}$) | $\theta={theta}$", fontsize=global_size_para + 4)
    plt.xlabel("Iteration Step $k$", fontsize=global_size_para + 2)
    plt.ylabel(r"Normalized Error Bound $R_k / MC$", fontsize=global_size_para + 2)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"lp_geometric_decay_{LAYERS}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Lp Geometric Decay Plot to {save_path}")
    plt.close()


# --- Main Execution ---
def main():
    print("=== Starting Adaptive Lp Norm Experiment (auto_LiRPA) ===")

    # 1. Model Setup & Training
    print("\n--- 1. Setting up Model ---")
    model = FlexibleNet(LAYERS, activation=ACTIVATION)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"Training new model on {N_SAMPLES} samples...")
        X_np, y_np = generate_balanced_2d_coin_data(N=N_SAMPLES, epsilon=EPSILON_TRANSITION)
        train_model(model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=EPOCHS, weight_decay=WEIGHT_DECAY)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.")

    model.eval()

    # 2. Monte Carlo Truth
    mc_lp = compute_monte_carlo_lp(model, p=P_NORM, domain=(DOMAIN_START, DOMAIN_END))

    # 3. Setup Library and Monkey-Patching for Heatmaps
    print(f"\n--- 2. Running auto_LiRPA L{int(P_NORM)} Refinement (theta={THETA}) ---")

    # Store original method for Monkey-Patch
    original_refine_once = adaptive_lp_core.Partition.refine_once

    # Use a dictionary to track state safely across scope bounds
    tracker_state = {"iteration": 0}

    def patched_refine_once(self_part, marker, refiner="all", estimator=None):
        # Print heatmap BEFORE refinement
        if tracker_state["iteration"] % HEATMAP_STEP == 0:
            extract_and_plot_heatmap(self_part, tracker_state["iteration"])
        tracker_state["iteration"] += 1

        # Call original logic
        return original_refine_once(self_part, marker, refiner, estimator)

    # Apply Patch
    adaptive_lp_core.Partition.refine_once = patched_refine_once

    # ====================================================================
    # DER GENIALE TRICK: Skalierung als echter nn.Linear Layer
    # ====================================================================
    scaling_layer = nn.Linear(D_DIM, D_DIM)
    scaling_layer.weight.data = torch.eye(D_DIM) * DOMAIN_WIDTH     # Diagonale Matrix (3.0)
    scaling_layer.bias.data = torch.full((D_DIM,), float(DOMAIN_START)) # Verschiebungsvektor (-1.5)
    scaling_layer.requires_grad_(False) # Einfrieren

    # Extrahiere die Layer deines trainierten Modells (FlexibleNet nutzt oft self.net)
    if hasattr(model, 'net'):
        inner_layers = list(model.net.children())
    elif hasattr(model, 'layers'):
        inner_layers = list(model.layers.children())
    else:
        inner_layers = list(model.children())

    # Baue ein neues sequentielles Modell: Erst Skalierung, dann dein Netz!
    wrapped_model = nn.Sequential(scaling_layer, *inner_layers)
    wrapped_model.eval()

    # WICHTIG: Jetzt können wir is_relu aktivieren, falls es ein ReLU Netz ist.
    # Die Funktion boxes_inside_activation_region wird den scaling_layer erkennen,
    # die Boxen korrekt skalieren und dann fehlerfrei die ReLUs scannen!
    is_relu_net = (ACTIVATION.lower() == 'relu')

    lp_runner = AdaptiveLpNorm(d=D_DIM, neural_net=wrapped_model, is_relu=is_relu_net)
    # ====================================================================

    start_time = time.time()
    with silence_everything():
        lp_runner.norm(p=P_NORM, m=MAX_ITER, theta=THETA, refiner=REFINER, eta_type="fval")
    duration = time.time() - start_time

    # Plot the final heatmap after the loop finishes
    extract_and_plot_heatmap(lp_runner.partition_adaptive, tracker_state["iteration"])

    # Clean up Monkey Patch
    adaptive_lp_core.Partition.refine_once = original_refine_once

    # 4. Extract and Normalize Data
    results = lp_runner.out_adap
    sizes = np.array(results['size_hist'])

    # Scale the integration gaps back up to the full [-1.5, 1.5] volume
    raw_gaps = np.array(results['post_hist']) * SCALE_FACTOR
    final_approx = results['approx'] * SCALE_FACTOR
    final_error = results['error'] * SCALE_FACTOR

    normalized_gaps = raw_gaps / mc_lp
    iterations = np.arange(len(normalized_gaps))

    print(f"\n{'=' * 40}")
    print(f"RESULTS (Duration: {duration:.2f}s)")
    print(f"{'=' * 40}")
    print(f"Final calculated L{int(P_NORM)} Norm : {final_approx:.6f}")
    print(f"Final a-posteriori Error : {final_error:.6f}")
    print(f"MC Ground Truth          : {mc_lp:.6f}")
    print(f"Final Grid Boxes         : {sizes[-1]}")
    print(f"{'=' * 40}")

    # 5. Plotting
    print("\n--- 3. Generating Plot ---")
    plot_lp_geometric_decay(iterations, normalized_gaps, THETA, RHO)

if __name__ == "__main__":
    main()