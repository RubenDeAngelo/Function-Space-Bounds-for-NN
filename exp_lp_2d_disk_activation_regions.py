"""Fig. 2 + Fig. 6: Lp norm on the smooth-coin 2D domain with activation region heatmaps (ReLU network)."""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
import matplotlib.patches as patches

global_font_size = 20
plt.rcParams.update({
    'font.size': global_font_size,
    'axes.titlesize': global_font_size,
    'axes.labelsize': global_font_size,
    'xtick.labelsize': global_font_size,
    'ytick.labelsize': global_font_size,
    'legend.fontsize': global_font_size,
    'figure.titlesize': global_font_size
})

from lib_adaquad_lp import (
    ReLUNetwork,
    AdaptiveLpNorm,
    monte_carlo_lp_norm
)

@contextmanager
def silence_everything(disable_logging: bool = True, disable_warnings: bool = True):
    """Suppresses warnings and console spam from external bounding libraries."""
    prev_disable_level = logging.root.manager.disable
    prev_warning_filters = warnings.filters[:]

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            if disable_logging:
                logging.disable(logging.CRITICAL)
            if disable_warnings:
                warnings.simplefilter("ignore")
            yield
        finally:
            if disable_warnings:
                warnings.filters = prev_warning_filters
            if disable_logging:
                logging.disable(prev_disable_level)

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

def generate_disk_data(n_samples=5000, domain_start=-1.5, domain_end=1.5, epsilon=0.5):
    X = np.random.uniform(domain_start, domain_end, (n_samples, 2)).astype(np.float32)
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return torch.from_numpy(X), torch.from_numpy(y.reshape(-1, 1))

def train_model(model, X_train, y_train, epochs=2000, lr=1e-3, weight_decay=0.0):
    print(f"Training model for {epochs} epochs...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.6f}")

    model.eval()
    print("Training complete.")

def get_flattened_layers(model):
    """Recursively searches for native Linear/ReLU layers to avoid Nested-Sequentials."""
    layers = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            layers.extend(get_flattened_layers(module))
        else:
            layers.append(module)
    return layers

def extract_and_plot_heatmap(partition, iteration, domain_start, domain_width, p, scale_factor, mc_truth, epsilon=0.5):
    """Extracts internal boxes and plots a heatmap of the local error bounds normalized by MC Truth."""
    ids = list(partition.leaves)
    if not ids: return

    # Retrieve coordinates in the library's internal [0,1] space
    slots = torch.tensor([partition.id2slot[i] for i in ids], dtype=torch.long, device=partition.storage.device)
    cr = partition.storage.index_select(0, slots).cpu().numpy()

    # Map back to the original [-1.5, 1.5] domain
    centers = domain_start + domain_width * cr[:, :, 0]
    radii = domain_width * cr[:, :, 1]

    # Scale errors to L_p^p space and normalize with global MC Truth
    raw_etas = np.array([partition.eta[i] for i in ids])
    etas = (raw_etas * scale_factor) / mc_truth

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
        rf"Normalized Adaptive Grid $L_{{{int(p)}}}^{{{int(p)}}}$ - Iteration {iteration}" + "\n" + rf"$N={len(ids)}$ Boxes",
        fontsize=global_font_size + 4)
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(rf'Normalized Local Gap ($L_{{{int(p)}}}^{{{int(p)}}}$)', fontsize=global_font_size)

    plt.tight_layout()
    file_name = f"lp_heatmap_iter_{iteration:02d}.pdf"
    plt.savefig(file_name, dpi=300)
    plt.close()
    print(f"Saved: {file_name}")

def plot_normalized_convergence(results, mc_truth, theta, p):
    """Plots the geometric convergence rate of the global bound gap normalized by MC truth."""
    post_err = np.array(results["post_hist"])
    iterations = np.arange(len(post_err))

    # Normalize (Error and mc_truth are both in L_p^p space)
    normalized_gaps = post_err / mc_truth

    plt.figure(figsize=(12, 9))

    # Empirical Line
    plt.plot(iterations, normalized_gaps, '-o', markersize=8, color='#D55E00', linewidth=2.5,
             label=rf'Empirical Bound Gap ($L_{{{int(p)}}}^{{{int(p)}}}$)')

    plt.yscale('log')
    plt.title(rf"Geometric Convergence Decay | $\theta$={theta}", fontsize=global_font_size + 4)
    plt.xlabel("Iteration Step $k$", fontsize=global_font_size)
    plt.ylabel("Normalized Global Error Bound", fontsize=global_font_size)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.savefig("lp_norm_normalized_decay.pdf", dpi=300)
    print("Saved: lp_norm_normalized_decay.pdf")
    plt.close()

def main():
    d = 2
    p = 2.0
    m = 30
    theta = 0.5
    refiner = "max"

    domain_start = -1.5
    domain_end = 1.5
    domain_width = domain_end - domain_start
    volume = domain_width ** d

    # Scale factor represents the volume to map L_p^p accurately
    scale_factor = volume

    print("--- 1. Network Setup & Training ---")
    net = ReLUNetwork(input_dim=d, output_dim=1, width=200, depth=1, init_method="he_normal")

    X_train, y_train = generate_disk_data(n_samples=5000, domain_start=domain_start, domain_end=domain_end, epsilon=0.5)
    train_model(net, X_train, y_train, epochs=4000, lr=1e-3)

    scaling_layer = nn.Linear(d, d)
    scaling_layer.weight.data = torch.eye(d) * domain_width
    scaling_layer.bias.data = torch.full((d,), float(domain_start))
    scaling_layer.requires_grad_(False)

    inner_layers = get_flattened_layers(net)
    wrapped_model = nn.Sequential(scaling_layer, *inner_layers)
    wrapped_model.eval()

    print("\n--- 2. Monte Carlo Validation ---")
    raw_mc_truth = monte_carlo_lp_norm(wrapped_model, d=d, p=p, n_samples=500_000)
    mc_truth = (raw_mc_truth ** p) * scale_factor
    print(f"Monte Carlo Ground Truth     : {mc_truth:.6f}")

    lp = AdaptiveLpNorm(d=d, neural_net=wrapped_model, is_relu=True)

    print(f"\n--- 3. Starting adaptive L_{int(p)} Bound Computation (m={m}, theta={theta}) ---")
    start_time = time.time()

    with silence_everything():
        out = lp.norm(p=p, m=m, theta=theta, refiner=refiner, eta_type="fval")

    duration = time.time() - start_time
    results = lp.out_adap

    # Calculate final corrected L_p^p numbers
    raw_post_err = np.array(results["post_hist"])
    results["post_hist"] = raw_post_err * scale_factor
    final_approx = (results['approx'] ** p) * scale_factor
    final_error = results['error'] * scale_factor

    print(f"\n{'=' * 40}")
    print(f"RESULTS (Duration: {duration:.2f} seconds)")
    print(f"{'=' * 40}")
    print(f"Final Calculated L_{int(p)}^{int(p)} Integral : {final_approx:.6f}")
    print(f"Final a-posteriori Error Gap : {final_error:.6f}")
    print(f"Total Number of Grid Boxes   : {results['size']}")
    print(f"{'=' * 40}\n")

    print("--- 4. Generating Plots ---")
    plot_normalized_convergence(results, mc_truth, theta, p)

    # Generate a heatmap snapshot every 5 iterations
    print("\nGenerating intermediate heatmaps...")
    for it in range(5, m + 1, 5):
        lp_step = AdaptiveLpNorm(d=d, neural_net=wrapped_model, is_relu=True)
        with silence_everything():
            lp_step.norm(p=p, m=it, theta=theta, refiner=refiner, eta_type="fval")

        extract_and_plot_heatmap(
            partition=lp_step.partition_adaptive,
            iteration=it,
            domain_start=domain_start,
            domain_width=domain_width,
            p=p,
            scale_factor=scale_factor,
            mc_truth=mc_truth,
            epsilon=0.5
        )

if __name__ == "__main__":
    main()