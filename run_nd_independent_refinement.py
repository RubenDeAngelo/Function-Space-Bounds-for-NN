"""
run_nd_independent_refinement.py

Multi-dimensional comparison of W1p and W2p norms using
INDEPENDENT Dynamic Grids with DÖRFLER MARKING.
Extends the 1D AMR logic to d-dimensions.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import pi
import itertools

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# --- Global Font Size Configuration ---
global_size_para = 20
plt.rcParams.update({
    'font.size': global_size_para,
    'axes.titlesize': global_size_para,
    'axes.labelsize': global_size_para,
    'xtick.labelsize': global_size_para,
    'ytick.labelsize': global_size_para,
    'legend.fontsize': global_size_para,
    'figure.titlesize': global_size_para
})

# --- Imports (Assuming these exist in your directory) ---
from flex_model import FlexibleNet
from training import train_model
from w1p_bounds import compute_w1p_bounds
from w2p_bounds import compute_w2p_bounds

# --- CONFIGURATION ---
D_IN = 1  # Input Dimension (d)
D_OUT = 30  # Output Dimension
LAYERS = [D_IN, 32, 32, 32, 32, D_OUT]
ACTIVATION = 'tanh'
P_NORM = 2.0
MAX_ITER = 25  # Reduced for d-dimensions to prevent combinatorial explosion
THETA = 0.4  # Dörfler Parameter
RHO = 0.50  # Theoretical reduction factor base
JACOBIAN_METHOD = 'ibp'

FIG_DIR = f"fig_nd_refinement_{LAYERS}"
os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- UTILS ---

def generate_multivariate_data(d_in, d_out, N=500):
    """
    Generates d-dimensional Gaussian pulses using biased sampling to overcome
    the curse of dimensionality. Automatically adjusts ratio based on dimensions.
    """
    # 1. Determine heuristic peak ratio based on dimensions
    if d_in <= 1:
        peak_ratio = 0.0
    elif d_in == 2:
        peak_ratio = 0.20
    elif d_in == 3:
        peak_ratio = 0.50
    elif d_in == 4:
        peak_ratio = 0.75
    else:
        peak_ratio = 0.90

    N_peak = int(N * peak_ratio)
    N_uniform = N - N_peak

    # 2. Sample Uniform Background (Exploration)
    X_uniform = torch.rand(N_uniform, d_in) * 2 * pi

    # 3. Sample Biased Peak (Exploitation)
    X_peak = torch.randn(N_peak, d_in) * 0.8 + pi
    X_peak = torch.clamp(X_peak, 0.0, 2 * pi)

    # Combine and shuffle
    X = torch.cat([X_uniform, X_peak], dim=0)
    shuffle_idx = torch.randperm(N)
    X = X[shuffle_idx]

    # 4. Generate Labels with Dimensional Scaling
    dist_sq = torch.sum((X - pi)**2, dim=1, keepdim=True)

    # Scale the variance factor by d_in to prevent immediate underflow to 0.0 in high D
    variance_factor = 2.0 * d_in
    y = torch.exp(- dist_sq / variance_factor).repeat(1, d_out)

    return X.numpy().astype(np.float32), y.numpy().astype(np.float32)


def plot_monte_carlo_convergence(model, max_samples=500000, step=5000000, p=2.0, domain=(0.0, 2 * pi)):
    """Computes and plots ND analytical Sobolev bounds via Monte Carlo sampling incrementally."""
    print(f"\n--- Computing {D_IN}D Monte Carlo Ground Truth Convergence (Up to N={max_samples}) ---")

    w1_means = []
    w2_means = []
    sample_counts = list(range(step, max_samples + 1, step))

    w1p_sum = 0.0
    w2p_sum = 0.0
    total_samples = 0
    volume = (domain[1] - domain[0]) ** D_IN

    for n in sample_counts:
        X = torch.empty(step, D_IN, device=device).uniform_(domain[0], domain[1]).requires_grad_(True)
        Y = model(X)

        # Compute W0p (phi_p)
        phi_p = torch.abs(Y).pow(p).sum(dim=1)

        # Compute W1p (Jacobian Frobenius Norm approximation)
        grad_p = torch.zeros(step, device=device)
        for i in range(D_OUT):
            grad_outputs = torch.zeros_like(Y)
            grad_outputs[:, i] = 1.0
            grad_Y = torch.autograd.grad(Y, X, grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
            grad_p += torch.abs(grad_Y).pow(p).sum(dim=1)

        # Compute W2p (Hessian approximation)
        hess_p = torch.zeros(step, device=device)
        for i in range(D_OUT):
            grad_outputs = torch.zeros_like(Y)
            grad_outputs[:, i] = 1.0
            grad_Y = torch.autograd.grad(Y, X, grad_outputs=grad_outputs, create_graph=True)[0]

            for j in range(D_IN):
                grad2_outputs = torch.zeros_like(grad_Y)
                grad2_outputs[:, j] = 1.0
                hess_Y = torch.autograd.grad(grad_Y, X, grad_outputs=grad2_outputs, retain_graph=True)[0]
                hess_p += torch.abs(hess_Y).pow(p).sum(dim=1)

        w1p_integrand = phi_p + grad_p
        w2p_integrand = phi_p + grad_p + hess_p

        w1p_sum += w1p_integrand.sum().item()
        w2p_sum += w2p_integrand.sum().item()
        total_samples += step

        w1_means.append(volume * (w1p_sum / total_samples))
        w2_means.append(volume * (w2p_sum / total_samples))

        #print(f"MC Samples: {total_samples:<6} | W1P: {w1_means[-1]:.6f} | W2P: {w2_means[-1]:.6f}")

    # --- PLOTTING MC CONVERGENCE ---
    plt.figure(figsize=(12, 9))
    plt.plot(sample_counts, w1_means, '-', color='#D55E00', linewidth=2.5, label=r'$W^{1,2}$ MC Estimate')
    plt.plot(sample_counts, w2_means, '-', color='#009E73', linewidth=2.5, label=r'$W^{2,2}$ MC Estimate')

    plt.title(f"{D_IN}D Monte Carlo Convergence", fontsize=global_size_para + 6)
    plt.xlabel("Number of Samples", fontsize=global_size_para + 2)
    plt.ylabel("Sobolev Norm Estimate", fontsize=global_size_para + 2)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, f"monte_carlo_convergence_{D_IN}D_{LAYERS}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved MC Convergence Plot to {save_path}")
    plt.close()

    print(f"\nFinal MC W1P True Value: {w1_means[-1]:.6f}")
    print(f"Final MC W2P True Value: {w2_means[-1]:.6f}")
    return w1_means[-1], w2_means[-1]


# --- ND REFINEMENT LOGIC ---

def process_cell_bounds(model, center, delta, p, norm_type, jac_method):
    """Wrapper to safely call external bound functions and calculate mass."""
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    # THE FIX: Keep center as a NumPy array (do not convert to torch.tensor)
    # The underlying bounds modules (network_ops.py) use NumPy for the forward pass!
    cx_input = np.array(center, dtype=np.float32)

    # Call the external bound function
    lb, ub = bound_fn(model, cx_input, delta, p=p, jacobian_method=jac_method)

    # Ensure lb, ub are scalars (whether they return as tensors, arrays, or floats)
    if hasattr(lb, 'item'): lb = lb.item()
    if hasattr(ub, 'item'): ub = ub.item()

    # THE FIX: The external bounds modules already integrate the volume of the cell internally.
    # We do NOT multiply by volume here, matching the correct 1D mathematical logic.
    error_mass = 0.5 * ((ub ** p) - (lb ** p))
    q_val = 0.5 * ((ub ** p) + (lb ** p))

    return lb, ub, error_mass, q_val

def run_nd_doerfler_grid(model, domain_start, domain_end, max_iter, theta=0.5, p=2.0, norm_type='w1p',
                         jac_method='ibp'):
    print(f"\n--- Running {D_IN}D Dörfler Refinement (theta={theta}, p={p}, method={jac_method}) for {norm_type.upper()} ---")

    active_cells = []

    # Initial Macro-Cell
    initial_delta = (domain_end - domain_start) / 2.0
    initial_center = np.array([domain_start + initial_delta] * D_IN)

    lb, ub, err, q = process_cell_bounds(model, initial_center, initial_delta, p, norm_type, jac_method)

    active_cells.append({
        'center': initial_center, 'delta': initial_delta,
        'low': lb, 'high': ub, 'error': err, 'q_val': q
    })

    history = []

    for i in range(max_iter + 1):
        count = len(active_cells)
        global_gap = sum(c['error'] for c in active_cells)
        global_q = sum(c['q_val'] for c in active_cells)

        history.append({'iter': i, 'count': count, 'gap_dyn': global_gap, 'q_dyn': global_q})

        # UPDATED: Log formatting to match the target spacing and float precision exactly
        print(f"Iter {i:<2}  | Cells: {count:<5}  | {norm_type.upper()} Gap: {global_gap:.6f} | Q_n: {global_q:.6f}")

        if i == max_iter: break

        active_cells.sort(key=lambda c: c['error'], reverse=True)
        marked_set, unmarked_set = [], []
        current_sum = 0.0
        cutoff = theta * global_gap

        for cell in active_cells:
            if current_sum < cutoff:
                marked_set.append(cell)
                current_sum += cell['error']
            else:
                unmarked_set.append(cell)

        # Stop if no cells are marked (numerical precision limit reached)
        if not marked_set:
            break

        new_cells = []
        # Split all marked cells into 2^d children
        for parent in marked_set:
            new_delta = parent['delta'] / 2.0
            offsets = list(itertools.product([-new_delta, new_delta], repeat=D_IN))

            for offset in offsets:
                child_center = parent['center'] + np.array(offset)
                cl, cu, cerr, cq = process_cell_bounds(model, child_center, new_delta, p, norm_type, jac_method)

                new_cells.append({
                    'center': child_center, 'delta': new_delta,
                    'low': cl, 'high': cu, 'error': cerr, 'q_val': cq
                })

        active_cells = unmarked_set + new_cells

    return history


# --- PLOTTING ---

def plot_geometric_decay_combined(w1_hist, w2_hist, theta, mc_w1, mc_w2):
    iter_w1 = np.array([h['iter'] for h in w1_hist])
    err_w1 = np.array([h['gap_dyn'] for h in w1_hist]) / mc_w1

    iter_w2 = np.array([h['iter'] for h in w2_hist])
    err_w2 = np.array([h['gap_dyn'] for h in w2_hist]) / mc_w2

    plt.figure(figsize=(12, 9))
    plt.plot(iter_w1, err_w1, '-o', markersize=8, color='#D55E00', linewidth=2.5, label=r'Empirical $W^{1,2}$')
    plt.plot(iter_w2, err_w2, '-s', markersize=8, color='#009E73', linewidth=2.5, label=r'Empirical $W^{2,2}$')

    plt.yscale('log')
    plt.title(rf"{D_IN}D Geometric Convergence Decay | $\theta$={theta}", fontsize=global_size_para + 6)
    plt.xlabel("Iteration Step $k$", fontsize=global_size_para + 2)
    plt.ylabel("Normalized Global Error Bound", fontsize=global_size_para + 2)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    save_path = os.path.join(FIG_DIR, f"geometric_decay_combined_{D_IN}D_{LAYERS}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Combined ND Geometric Decay Plot to {save_path}")
    plt.close()


# --- MAIN EXPERIMENT ---

def run_nd_independent_experiment():
    print(f"--- Starting {D_IN}D Independent Refinement Experiment ---")

    X_np, y_np = generate_multivariate_data(D_IN, D_OUT)

    print("Training Model...")
    model = FlexibleNet(LAYERS, activation=ACTIVATION).to(device)
    train_model(model, torch.from_numpy(X_np).to(device), torch.from_numpy(y_np).to(device), epochs=10000)
    model.eval()

    domain_start, domain_end = 0.0, 2 * pi

    # Calculate ND Monte Carlo Ground Truth AND Plot Convergence
    mc_w1p, mc_w2p = plot_monte_carlo_convergence(model, max_samples=500000, step=10000, p=P_NORM, domain=(domain_start, domain_end))

    # Run Refinements
    w1_history = run_nd_doerfler_grid(model, domain_start, domain_end, MAX_ITER, THETA, P_NORM, 'w1p', JACOBIAN_METHOD)
    w2_history = run_nd_doerfler_grid(model, domain_start, domain_end, MAX_ITER, THETA, P_NORM, 'w2p', JACOBIAN_METHOD)

    # Plot
    print("\n--- Generating ND Combined Geometric Decay Plot ---")
    plot_geometric_decay_combined(w1_history, w2_history, THETA, mc_w1p, mc_w2p)


if __name__ == "__main__":
    run_nd_independent_experiment()