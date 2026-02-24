"""
plot_cross_section_derivatives_uniform.py

Generates a comparison of 1D cross-sections (y=0) and their 1st and 2nd derivatives
for a Deep vs. Wide neural network approximating the 2D Smooth Coin function.
Trained strictly on UNIFORM sampling.
Outputs the results as two separate PDFs.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Global Font Size Configuration ---
global_size_para = 20
plt.rcParams.update({
    'font.size': global_size_para,
    'axes.titlesize': global_size_para + 2,
    'axes.labelsize': global_size_para,
    'xtick.labelsize': global_size_para - 2,
    'ytick.labelsize': global_size_para - 2,
    'legend.fontsize': global_size_para - 2,
    'figure.titlesize': global_size_para + 4
})

# --- Imports (Assumed to be in the same directory) ---
from flex_model import FlexibleNet
from training import train_model

# --- CONFIGURATION ---
ARCH_DEEP = [2, 32, 32, 32, 32, 1]
ARCH_WIDE = [2, 200, 1]
ACTIVATION = 'sigmoid'
EPOCHS = 6000
N_SAMPLES = 5000

# Independent Learning Rates
LR_DEEP = 1e-3
LR_WIDE = 5e-2

EPSILON_TRANSITION = 0.2

FIG_DIR = "fig_cross_sections"
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_PATH_DEEP = f"model_smooth_coin_2d_DEEP_UNIFORM.pth"
MODEL_PATH_WIDE = f"model_smooth_coin_2d_WIDE_UNIFORM.pth"


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

# --- UNIFORM SAMPLING FUNCTION ---
def generate_uniform_2d_coin_data(N=3000, epsilon=0.1):
    low, high = -1.5, 1.5
    X = np.random.uniform(low, high, (N, 2)).astype(np.float32)
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return X, y.reshape(-1, 1).astype(np.float32)


# --- DERIVATIVE COMPUTATION ---
def compute_cross_section_derivatives(model, x_range=(-1.6, 1.6), points=1000):
    """
    Evaluates the network at y=0 over x_range and computes 1st and 2nd derivatives w.r.t x.
    """
    model.eval()
    x_val = torch.linspace(x_range[0], x_range[1], points).view(-1, 1)
    y_val = torch.zeros_like(x_val)

    # Combine to (N, 2) and require gradients for autograd
    X_test = torch.cat([x_val, y_val], dim=1).requires_grad_(True)

    # Forward Pass: f(x, 0)
    out = model(X_test)

    # First Derivative: df/dx
    grad_out = torch.ones_like(out)
    jacobian = torch.autograd.grad(out, X_test, grad_outputs=grad_out, create_graph=True)[0]
    df_dx = jacobian[:, 0:1]  # Extract only the x-derivative

    # Second Derivative: d^2f/dx^2
    grad_df_dx = torch.ones_like(df_dx)
    hessian_x = torch.autograd.grad(df_dx, X_test, grad_outputs=grad_df_dx, retain_graph=False)[0]
    d2f_dx2 = hessian_x[:, 0:1] # Extract x-derivative of the x-derivative

    # Return as numpy arrays for plotting
    x_np = x_val.detach().numpy().flatten()
    out_np = out.detach().numpy().flatten()
    df_dx_np = df_dx.detach().numpy().flatten()
    d2f_dx2_np = d2f_dx2.detach().numpy().flatten()

    return x_np, out_np, df_dx_np, d2f_dx2_np


# --- PLOTTING LOGIC ---
def create_architecture_plot(x, f_true, f_pred, df, d2f, arch_name, save_filename):
    """Helper function to generate a 3-subplot figure for a specific architecture."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    def format_ax(ax, title=None, ylabel=None):
        if title: ax.set_title(title)
        if ylabel: ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)

    # --- TOP: Cross-Section ---
    axes[0].plot(x, f_true, '--', color='grey', linewidth=2, label='Ground Truth')
    axes[0].plot(x, f_pred, '-', color='blue', linewidth=2, label='NN Output')
    format_ax(axes[0], title=f"Network Cross-Section (y=0) - {arch_name}", ylabel="f(x, 0)")
    axes[0].legend()

    # --- MIDDLE: 1st Derivative ---
    axes[1].plot(x, df, '-', color='red', linewidth=2, label=r'$\partial f/\partial x$')
    format_ax(axes[1], ylabel="1st Deriv")
    axes[1].legend()

    # --- BOTTOM: 2nd Derivative ---
    axes[2].plot(x, d2f, '-', color='forestgreen', linewidth=2, label=r'$\partial^2 f/\partial x^2$')
    format_ax(axes[2], ylabel="2nd Deriv")
    axes[2].legend()
    axes[2].set_xlabel("x")

    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, save_filename)
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"Plot saved successfully to: {save_path}")
    plt.close()


def run_and_plot():
    print("--- Preparing Data & Models (Uniform Sampling) ---")
    X_np, y_np = generate_uniform_2d_coin_data(N=N_SAMPLES, epsilon=EPSILON_TRANSITION)
    X_tensor = torch.from_numpy(X_np)
    y_tensor = torch.from_numpy(y_np)

    # Load or Train Deep Model
    model_deep = FlexibleNet(ARCH_DEEP, activation=ACTIVATION)
    if os.path.exists(MODEL_PATH_DEEP):
        model_deep.load_state_dict(torch.load(MODEL_PATH_DEEP))
    else:
        print("Training Deep Model...")
        train_model(model_deep, X_tensor, y_tensor, epochs=EPOCHS, lr=LR_DEEP)
        torch.save(model_deep.state_dict(), MODEL_PATH_DEEP)

    # Load or Train Wide Model
    model_wide = FlexibleNet(ARCH_WIDE, activation=ACTIVATION)
    if os.path.exists(MODEL_PATH_WIDE):
        model_wide.load_state_dict(torch.load(MODEL_PATH_WIDE))
    else:
        print("Training Wide Model...")
        train_model(model_wide, X_tensor, y_tensor, epochs=EPOCHS, lr=LR_WIDE)
        torch.save(model_wide.state_dict(), MODEL_PATH_WIDE)

    print("--- Computing Derivatives ---")
    x_d, f_d, df_d, d2f_d = compute_cross_section_derivatives(model_deep)
    x_w, f_w, df_w, d2f_w = compute_cross_section_derivatives(model_wide)

    # True function for overlay
    f_true = smooth_disk(x_d, np.zeros_like(x_d), epsilon=EPSILON_TRANSITION)

    print("--- Generating Plots ---")
    create_architecture_plot(x_d, f_true, f_d, df_d, d2f_d, "Deep Architecture", "architecture_derivative_deep_uniform.pdf")
    create_architecture_plot(x_w, f_true, f_w, df_w, d2f_w, "Wide Architecture", "architecture_derivative_wide_uniform.pdf")

if __name__ == "__main__":
    run_and_plot()