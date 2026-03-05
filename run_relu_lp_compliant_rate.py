"""
run_relu_lp_compliant.py

Computation of Lp norms using Adaptive Refinement via auto_LiRPA (AdaptiveLpNorm).
ADAPTED FOR 2D "SMOOTH COIN" FUNCTION.

Updates:
- Matched file structure and parameter definitions to the Sobolev role-model.
- NORMALIZED: Geometric Decay plot divides error by MC Ground Truth.
- LEGEND: Shows exact calculated q-value in the plot.
- HEATMAP: Generates a visual map of the local gaps for the final adaptive grid.
- Integrates domain scaling as a native nn.Linear layer to allow ReLU optimizations.
- Applies the (1/p) root and volume scaling to extract accurate L_p norms from L_p^p integrals.
- All plots saved as PDFs.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from contextlib import contextmanager, redirect_stdout, redirect_stderr

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
from adaptive_lp_core import AdaptiveLpNorm

# Data Generation Settings
EPSILON_TRANSITION = 0.5

# Model & Training
LAYERS = [2, 32, 32, 32, 1]  # Deep ReLU architecture
weight_decay = 0.0
ACTIVATION = 'relu'          # Must be 'relu' for library optimizations
EPOCHS = 2000
N_SAMPLES = 5000
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}_{EPSILON_TRANSITION}_wd{weight_decay}_{ACTIVATION}.pth"

# --- CONFIGURATION ---
FIG_DIR = f"fig_lp_geometric_decay_{LAYERS}_{EPSILON_TRANSITION}_wd{weight_decay}_{ACTIVATION}"
os.makedirs(FIG_DIR, exist_ok=True)

# Refinement Settings
P_NORM = 2.0
MAX_ITER = 30
THETA = 0.4
RHO = 0.50            # Theoretical local error reduction factor for 1D bisection in 2D
REFINER = "max"       # Bisection strategy


@contextmanager
def silence_everything(disable_logging: bool = True, disable_warnings: bool = True):
    """Suppresses warnings and console spam from external bounding libraries."""
    prev_disable_level = logging.root.manager.disable
    prev_warning_filters = warnings.filters[:]
    with open(os.devnull, "w"), redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        try:
            if disable_logging: logging.disable(logging.CRITICAL)
            if disable_warnings: warnings.simplefilter("ignore")
            yield
        finally:
            if disable_warnings: warnings.filters = prev_warning_filters
            if disable_logging: logging.disable(prev_disable_level)


# --- SMOOTH COIN FUNCTION DEFINITION ---
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


def generate_balanced_2d_coin_data(N=5000, epsilon=0.1, transition_ratio=0.5):
    N_transition = int(N * transition_ratio)
    N_uniform = N - N_transition
    low, high = -1.5, 1.5

    # 1. Generate Uniform Data
    X_uni = np.random.uniform(low, high, (N_uniform, 2)).astype(np.float32)

    # 2. Generate Data concentrated on the transition boundary
    radii = np.random.uniform(1.0, 1.0 + epsilon, N_transition).astype(np.float32)
    angles = np.random.uniform(0, 2 * np.pi, N_transition).astype(np.float32)
    X_trans = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    # 3. Combine them
    X = np.vstack((X_uni, X_trans))

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)

    return X, y.reshape(-1, 1).astype(np.float32)


# --- MONTE CARLO INTEGRATION ---
def compute_monte_carlo_lp(model, num_samples=500000, p=2.0, domain=(-1.5, 1.5)):
    print(f"\n--- Computing Monte Carlo Ground Truth (N={num_samples}) ---")
    model.eval()
    device = next(model.parameters()).device

    X = torch.empty(num_samples, 2, device=device).uniform_(domain[0], domain[1])
    volume = (domain[1] - domain[0]) ** 2

    Y = model(X)
    lp_integrand = torch.abs(Y).pow(p).sum(dim=1)

    lp_integral = volume * lp_integrand.mean().item()
    lp_mc = lp_integral ** (1.0 / p)

    print(f"MC L{int(p)} True Value: {lp_mc:.6f}")
    return lp_mc


# --- LAYER EXTRACTION UTILITY ---
def get_flattened_layers(model):
    """Recursively searches for Linear/ReLU layers to avoid Nested-Sequentials."""
    layers = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            layers.extend(get_flattened_layers(module))
        else:
            layers.append(module)
    return layers


# --- PLOTTING FUNCTIONS ---
def plot_geometric_decay_lp(results, theta, p, rho, mc_truth):
    """
    Plots the geometric decay of the Lp norm, aligned with the role-model style.
    """
    raw_post_err = np.array(results["post_hist"])
    iterations = np.arange(len(raw_post_err))

    domain_width = 3.0 # -1.5 to 1.5
    volume = domain_width ** 2
    scale_factor = volume ** (1.0 / p)

    scaled_err = (raw_post_err ** (1.0 / p)) * scale_factor
    err_normalized = scaled_err / mc_truth

    q_integral = 1.0 - theta * (1.0 - rho)
    q_norm = q_integral ** (1.0 / p)

    theory_decay = err_normalized[0] * (q_norm ** iterations)

    plt.figure(figsize=(12, 9))

    plt.plot(iterations, err_normalized, '-o', markersize=8, color='#D55E00', linewidth=2.5, label=rf'Empirical $L^{{{int(p)}}}$')
    plt.plot(iterations, theory_decay, '--', color='#D55E00', linewidth=2.0, alpha=0.7, label=fr'Theory $q^k={q_norm:.4f}^k$')

    plt.yscale('log')
    plt.title(rf"Geometric Convergence Decay | $\theta$={theta}", fontsize=gloabal_size_para + 6)
    plt.xlabel("Iteration Step $k$", fontsize=gloabal_size_para + 2)
    plt.ylabel("Normalized Global Error Bound", fontsize=gloabal_size_para + 2)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"geometric_decay_lp_{ACTIVATION}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Lp Geometric Decay Plot to {save_path}")
    plt.close()


def extract_and_plot_heatmap(partition, iteration, domain_start, domain_width, p, scale_factor, epsilon=0.5):
    """Extrahiert die internen Boxen und zeichnet eine Heatmap der lokalen Fehler."""
    ids = list(partition.leaves)
    if not ids: return

    slots = torch.tensor([partition.id2slot[i] for i in ids], dtype=torch.long, device=partition.storage.device)
    cr = partition.storage.index_select(0, slots).cpu().numpy()

    centers = domain_start + domain_width * cr[:, :, 0]
    radii = domain_width * cr[:, :, 1]

    raw_etas = np.array([partition.eta[i] for i in ids])
    etas = (raw_etas ** (1.0 / p)) * scale_factor

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

    circle = plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--', linewidth=1.5, label='Inner Radius')
    circle_outer = plt.Circle((0, 0), 1.0 + epsilon, color='orange', fill=False, linestyle='--', linewidth=1.5, label='Outer Radius')
    ax.add_patch(circle)
    ax.add_patch(circle_outer)

    domain_end = domain_start + domain_width
    ax.set_xlim(domain_start, domain_end)
    ax.set_ylim(domain_start, domain_end)
    ax.set_title(rf"Adaptive Grid $L_{{{int(p)}}}$ - Iteration {iteration}\n$N={len(ids)}$ Boxes", fontsize=gloabal_size_para + 6)
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(rf'Local Gap ($L_{{{int(p)}}}$)', fontsize=gloabal_size_para + 2)

    save_name = f"heatmap_lp_compliant_iter_{iteration:03d}.pdf"
    save_path = os.path.join(FIG_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Heatmap Plot to {save_path}")


# --- MAIN EXPERIMENT ---
def run_experiment():
    print(f"--- Starting Geometric Decay Experiment (Lp Norm) ---")

    # 1. Generate Data & Train Model
    X_np, y_np = generate_balanced_2d_coin_data(N=N_SAMPLES, epsilon=EPSILON_TRANSITION)
    model = FlexibleNet(LAYERS, activation=ACTIVATION)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"No trained model found. Training on {len(X_np)} samples...")
        train_model(model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=EPOCHS, weight_decay=weight_decay)
        torch.save(model.state_dict(), MODEL_PATH)

    model.eval()

    # 2. Monte Carlo Truth Calculation
    mc_lp = compute_monte_carlo_lp(model, num_samples=100000, p=P_NORM, domain=(-1.5, 1.5))

    # 3. Setup Domain Scaling Wrapper
    print("\n--- Setting up Domain Scaling for auto_LiRPA ---")
    d_dim = 2
    domain_start = -1.5
    domain_width = 3.0

    scaling_layer = nn.Linear(d_dim, d_dim)
    scaling_layer.weight.data = torch.eye(d_dim) * domain_width
    scaling_layer.bias.data = torch.full((d_dim,), float(domain_start))
    scaling_layer.requires_grad_(False)

    inner_layers = get_flattened_layers(model.net if hasattr(model, 'net') else model)
    wrapped_model = nn.Sequential(scaling_layer, *inner_layers)
    wrapped_model.eval()

    # 4. Execute Adaptive Lp Norm
    print(f"\n--- Running Adaptive L{int(P_NORM)} Refinement (theta={THETA}) ---")
    lp_runner = AdaptiveLpNorm(d=d_dim, neural_net=wrapped_model, is_relu=(ACTIVATION.lower() == 'relu'))

    start_time = time.time()
    with silence_everything():
        lp_runner.norm(p=P_NORM, m=MAX_ITER, theta=THETA, refiner=REFINER, eta_type="fval")
    duration = time.time() - start_time

    results = lp_runner.out_adap

    # 5. Extraction and Output Logging
    print(f"\n{'=' * 40}")
    print(f"RESULTS (Duration: {duration:.2f}s)")
    print(f"{'=' * 40}")

    scale_factor = (domain_width ** d_dim) ** (1.0 / P_NORM)
    final_approx = (results['approx'] ** (1.0 / P_NORM)) * scale_factor
    final_error = (results['error'] ** (1.0 / P_NORM)) * scale_factor

    print(f"Final calculated L{int(P_NORM)} Norm : {final_approx:.6f}")
    print(f"Final a-posteriori Error : {final_error:.6f}")
    print(f"MC Ground Truth          : {mc_lp:.6f}")
    print(f"Final Grid Boxes         : {results['size']}")
    print(f"{'=' * 40}")

    # 6. Generate Plots
    print("\n--- Generating Plots ---")
    plot_geometric_decay_lp(results, THETA, P_NORM, RHO, mc_lp)
    extract_and_plot_heatmap(lp_runner.partition_adaptive, MAX_ITER, domain_start, domain_width, P_NORM, scale_factor, epsilon=EPSILON_TRANSITION)


if __name__ == "__main__":
    run_experiment()