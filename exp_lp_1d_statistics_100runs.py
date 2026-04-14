"""Fig. 5: 100-run statistical analysis of the Lp norm geometric decay (1D Gaussian pulse, wide vs deep ReLU)."""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt

global_font_size = 20
plt.rcParams.update({
    'font.size': global_font_size,
    'axes.titlesize': global_font_size,
    'axes.labelsize': global_font_size,
    'xtick.labelsize': global_font_size,
    'ytick.labelsize': global_font_size,
    'legend.fontsize': global_font_size - 4,
    'figure.titlesize': global_font_size
})

from lib_adaquad_lp import (
    ReLUNetwork,
    AdaptiveLpNorm,
    monte_carlo_lp_norm
)

NUM_RUNS = 100
EPOCHS = 3000
D_DIM = 1
P_NORM = 2.0
MAX_ITER = 30
THETA = 0.3
REFINER = "max"

DOMAIN_START = 0.0
DOMAIN_END = 2 * np.pi
DOMAIN_WIDTH = DOMAIN_END - DOMAIN_START
VOLUME_SCALE = DOMAIN_WIDTH ** D_DIM

FIG_DIR = f"fig_lp_statistical_combined_waterfall"
os.makedirs(FIG_DIR, exist_ok=True)

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

def generate_localized_1d_data(n_samples=150, domain_start=0.0, domain_end=2 * np.pi):
    """Generates the 1D Gaussian pulse data."""
    X = np.linspace(domain_start, domain_end, n_samples)
    y = np.exp(-20 * (X - np.pi)**2)
    return torch.from_numpy(X.reshape(-1, 1).astype(np.float32)), torch.from_numpy(y.reshape(-1, 1).astype(np.float32))

def train_model(model, X_train, y_train, epochs=2000, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    model.eval()

def get_flattened_layers(model):
    layers = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            layers.extend(get_flattened_layers(module))
        else:
            layers.append(module)
    return layers

def get_iteration_stats(data_list):
    """Pads unequal iteration histories and calculates percentiles."""
    max_len = max(len(arr) for arr in data_list)
    padded = []
    for arr in data_list:
        if len(arr) < max_len:
            # Pad early stopping with the final calculated bound gap
            arr = np.pad(arr, (0, max_len - len(arr)), mode='edge')
        padded.append(arr)

    stacked = np.vstack(padded)
    mean_vals = np.mean(stacked, axis=0)
    lower_ci = np.percentile(stacked, 2.5, axis=0)
    upper_ci = np.percentile(stacked, 97.5, axis=0)

    lower_ci = np.maximum(lower_ci, 1e-16)  # Prevent negative values on log scale
    iterations = np.arange(max_len)

    return iterations, mean_vals, lower_ci, upper_ci

def plot_combined_with_shading(data_dict, title_prefix, filename_suffix, is_normalized=True):
    plt.figure(figsize=(12, 9))

    color_deep = '#D55E00'  # Vermillion
    color_wide = '#009E73'  # Bluish Green

    def plot_curve(data_list, color, label_prefix, marker):
        if not data_list: return
        x, mean, lower, upper = get_iteration_stats(data_list)

        # Plot Mean Line
        plt.plot(x, mean, linestyle='-', marker=marker, color=color, linewidth=2.5, markersize=8,
                 label=rf"{label_prefix} ($L_{{{int(P_NORM)}}}^{{{int(P_NORM)}}}$)")

        # Plot Confidence Interval
        plt.fill_between(x, lower, upper, color=color, alpha=0.2)

    # Plot both architectures
    plot_curve(data_dict['deep'], color_deep, 'Deep Arch.', 'o')
    plot_curve(data_dict['wide'], color_wide, 'Wide Arch.', 's')

    plt.yscale('log')
    plt.title(rf"{title_prefix}: Geometric Decay | $\theta={THETA}$", fontsize=global_font_size + 4)
    plt.xlabel("Iteration Step $k$", fontsize=global_font_size)

    y_label = "Normalized Global Error" if is_normalized else "Global Error"
    plt.ylabel(y_label, fontsize=global_font_size)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    path = os.path.join(FIG_DIR, f"lp_adaptive_waterfall_combined_{filename_suffix}.pdf")
    plt.savefig(path, dpi=300, format='pdf')
    print(f"Saved {path}")
    plt.close()

def main():
    print(f"--- Starting {NUM_RUNS}-Run Statistical Experiment (L_p^p Integral) ---")

    X_train, y_train = generate_localized_1d_data(n_samples=150, domain_start=DOMAIN_START, domain_end=DOMAIN_END)

    results_untrained = {'deep': [], 'wide': []}
    results_trained = {'deep': [], 'wide': []}

    architectures = [
        ('deep', 32, 3),  # width=32, depth=3
        ('wide', 200, 1)  # width=200, depth=1
    ]

    for i in range(NUM_RUNS):
        print(f"\nEvaluating Seed {i + 1}/{NUM_RUNS}...")

        for arch_name, width, depth in architectures:
            print(f"  -> Processing {arch_name.upper()} architecture...", end=" ", flush=True)

            # 1. Reproducibility
            torch.manual_seed(i)
            np.random.seed(i)

            # 2. Network Setup & Wrapper
            net = ReLUNetwork(input_dim=D_DIM, output_dim=1, width=width, depth=depth, init_method="he_normal")

            scaling_layer = nn.Linear(D_DIM, D_DIM)
            scaling_layer.weight.data = torch.eye(D_DIM) * DOMAIN_WIDTH
            scaling_layer.bias.data = torch.full((D_DIM,), float(DOMAIN_START))
            scaling_layer.requires_grad_(False)

            wrapped_model = nn.Sequential(scaling_layer, *get_flattened_layers(net))
            wrapped_model.eval()

            # Calculate MC Truth for untrained, then transform to L_p^p space
            raw_mc_truth_untrained = monte_carlo_lp_norm(wrapped_model, d=D_DIM, p=P_NORM, n_samples=100_000)
            mc_truth_untrained = (raw_mc_truth_untrained ** P_NORM) * VOLUME_SCALE

            lp_untrained = AdaptiveLpNorm(d=D_DIM, neural_net=wrapped_model, is_relu=True)
            with silence_everything():
                lp_untrained.norm(p=P_NORM, m=MAX_ITER, theta=THETA, refiner=REFINER, eta_type="fval")

            # Extract, scale to L_p^p Volume, and normalize
            raw_untrained = np.array(lp_untrained.out_adap["post_hist"]) * VOLUME_SCALE
            normalized_untrained = raw_untrained / mc_truth_untrained
            results_untrained[arch_name].append(normalized_untrained)

            with suppress_stdout():
                train_model(net, X_train, y_train, epochs=EPOCHS, lr=1e-3)

            wrapped_model.eval()

            # Calculate MC Truth (returns L_p norm), then transform to L_p^p space
            raw_mc_truth = monte_carlo_lp_norm(wrapped_model, d=D_DIM, p=P_NORM, n_samples=100_000)
            mc_truth = (raw_mc_truth ** P_NORM) * VOLUME_SCALE

            lp_trained = AdaptiveLpNorm(d=D_DIM, neural_net=wrapped_model, is_relu=True)
            with silence_everything():
                lp_trained.norm(p=P_NORM, m=MAX_ITER, theta=THETA, refiner=REFINER, eta_type="fval")

            # Extract, scale to L_p^p, and normalize
            raw_trained = np.array(lp_trained.out_adap["post_hist"]) * VOLUME_SCALE
            normalized_trained = raw_trained / mc_truth
            results_trained[arch_name].append(normalized_trained)

            print("Done.")

    # 3. Generate Plots
    print("\nGenerating Statistical Plots...")
    # Both sets are now fully normalized
    plot_combined_with_shading(results_untrained, 'Untrained Networks', 'untrained', is_normalized=True)
    plot_combined_with_shading(results_trained, 'Trained Networks', 'trained', is_normalized=True)
    print("Experiment Complete.")

if __name__ == "__main__":
    main()