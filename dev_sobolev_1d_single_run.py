"""
run_independent_refinement.py

Comparison of W1p and W2p norms using INDEPENDENT Dynamic Grids with DÖRFLER MARKING.
- Grid 1: Optimized specifically to minimize W1p error (Theta-Marking).
- Grid 2: Optimized specifically to minimize W2p error (Theta-Marking).
- Plots the combined Geometric Decay against the theoretical rate q^k.
- Errors are normalized by the Monte Carlo ground truth.

UPDATED: Includes an analytical plot to prove why deep networks
trained on MSE fail to capture higher-order derivatives (W2p).
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import matplotlib.pyplot as plt
from math import pi
import itertools

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

from lib_nn_model import FlexibleNet
from lib_training import train_model
from lib_w1p_enclosures import compute_w1p_bounds
from lib_w2p_enclosures import compute_w2p_bounds

LAYERS = [1, 32, 32, 32, 32, 1]
ACTIVATION = 'tanh'
P_NORM = 2.0
MAX_ITER = 30
THETA = 0.4         # Dörfler Parameter
RHO = 0.50          # Theoretical reduction factor for 1D
JACOBIAN_METHOD = 'ibp'  # Interval Bound Propagation

FIG_DIR = f"fig_independent_refinement_{LAYERS}"
os.makedirs(FIG_DIR, exist_ok=True)

def split_points(points, delta):
    """Splits a list of center points into 2^d sub-intervals."""
    new_points = []
    if np.isscalar(delta):
        new_delta = delta / 2.0
    else:
         new_delta = delta / 2.0

    d = points[0].shape[0]
    offsets = [-new_delta, new_delta]
    offset_combinations = list(itertools.product(offsets, repeat=d))

    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape)
            new_points.append(x + offset)

    return new_points, new_delta

def compute_monte_carlo_sobolev_1d(model, num_samples=100000, p=2.0, domain=(0.0, 2*pi)):
    """Computes exact analytical Sobolev bounds via 1D Monte Carlo sampling."""
    print(f"\n--- Computing 1D Monte Carlo Ground Truth (N={num_samples}) ---")

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

    w1p_integrand = phi_p + grad_p
    w2p_integrand = phi_p + grad_p + hess_p

    w1p_mc = volume * w1p_integrand.mean().item()
    w2p_mc = volume * w2p_integrand.mean().item()

    print(f"MC W1P True Value: {w1p_mc:.6f}")
    print(f"MC W2P True Value: {w2p_mc:.6f}")
    return w1p_mc, w2p_mc

def generate_localized_data(N=150):
    """Generates 1D Gaussian pulse data."""
    X = np.linspace(0, 2*pi, N)
    y = np.exp(-20 * (X - pi)**2)
    return X.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32)

def plot_network_derivatives(model, domain_start, domain_end):
    """
    Plots the 0th, 1st, and 2nd derivatives of the learned network
    against the true mathematical function to visualize structural failure.
    """
    print("\n--- Generating Network Fit vs True Derivatives Plot ---")
    device = next(model.parameters()).device

    # Generate dense test points
    X_test = torch.linspace(domain_start, domain_end, 500, device=device).reshape(-1, 1).requires_grad_(True)

    # Neural Network Predictions
    Y_pred = model(X_test)
    grad_outputs = torch.ones_like(Y_pred)
    grad_Y = torch.autograd.grad(Y_pred, X_test, grad_outputs=grad_outputs, create_graph=True)[0]
    hess_Y = torch.autograd.grad(grad_Y, X_test, grad_outputs=torch.ones_like(grad_Y), retain_graph=False)[0]

    X_np = X_test.detach().cpu().numpy().flatten()
    y_pred_np = Y_pred.detach().cpu().numpy().flatten()
    dy_pred_np = grad_Y.detach().cpu().numpy().flatten()
    ddy_pred_np = hess_Y.detach().cpu().numpy().flatten()

    # Analytical Truth: f(x) = exp(-20 * (x - pi)^2)
    y_true = np.exp(-20 * (X_np - pi)**2)
    dy_true = -40 * (X_np - pi) * y_true
    ddy_true = 40 * (40 * (X_np - pi)**2 - 1) * y_true

    plt.figure(figsize=(24, 7))

    # 0th Derivative (Function Value)
    plt.subplot(1, 3, 1)
    plt.plot(X_np, y_true, 'k--', linewidth=4, label='True $f(x)$')
    plt.plot(X_np, y_pred_np, 'r-', linewidth=2.5, alpha=0.8, label='NN Fit')
    plt.title("Function Value (0th Order)", fontsize=global_size_para + 2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=global_size_para - 2)

    # 1st Derivative (Jacobian)
    plt.subplot(1, 3, 2)
    plt.plot(X_np, dy_true, 'k--', linewidth=4, label="True $f'(x)$")
    plt.plot(X_np, dy_pred_np, 'g-', linewidth=2.5, alpha=0.8, label="NN $f'(x)$")
    plt.title("First Derivative (1st Order)", fontsize=global_size_para + 2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=global_size_para - 2)

    # 2nd Derivative (Hessian)
    plt.subplot(1, 3, 3)
    plt.plot(X_np, ddy_true, 'k--', linewidth=4, label="True $f''(x)$")
    plt.plot(X_np, ddy_pred_np, 'b-', linewidth=2.5, alpha=0.8, label="NN $f''(x)$")
    plt.title("Second Derivative (2nd Order)", fontsize=global_size_para + 2)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=global_size_para - 2)

    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, f"network_derivative_fit_{LAYERS}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Network Fit plot to {save_path}")
    plt.close()

def run_doerfler_grid_for_norm(model, initial_centers, initial_delta, max_iter, theta=0.5, p=2.0, norm_type='w1p', jac_method='ibp'):
    print(f"\n--- Running Dörfler Refinement (theta={theta}, p={p}, method={jac_method.upper()}) for {norm_type.upper()} ---")

    active_cells = []
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    if np.isscalar(initial_delta):
         initial_batch = [(c.reshape(-1, 1), initial_delta) for c in initial_centers]
    else:
         initial_batch = zip(initial_centers, initial_delta)

    for x, d in initial_batch:
        lb, ub = bound_fn(model, x, d, p=p, jacobian_method=jac_method)
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        q_val = 0.5 * ((ub ** p) + (lb ** p))
        active_cells.append({
            'center': x, 'delta': d,
            'low': lb, 'high': ub,
            'error': error_mass,
            'q_val': q_val
        })

    history = []

    for i in range(max_iter + 1):
        count = len(active_cells)
        global_gap = sum(c['error'] for c in active_cells)
        global_q = sum(c['q_val'] for c in active_cells)

        history.append({'iter': i, 'count': count, 'gap_dyn': global_gap, 'q_dyn': global_q})
        print(f"Iter {i:<3} | Boxes: {count:<5} | {norm_type.upper()} Gap: {global_gap:.6f} | Q_n: {global_q:.6f}")

        if i == max_iter: break

        total_error = sum(c['error'] for c in active_cells)
        active_cells.sort(key=lambda c: c['error'], reverse=True)

        marked_set = []
        unmarked_set = []
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
            child_centers, child_delta = split_points([parent['center']], parent['delta'])
            for cx in child_centers:
                cl, cu = bound_fn(model, cx, child_delta, p=p, jacobian_method=jac_method)
                child_error_mass = 0.5 * ((cu ** p) - (cl ** p))
                child_q_val = 0.5 * ((cu ** p) + (cl ** p))
                new_cells.append({
                    'center': cx, 'delta': child_delta,
                    'low': cl, 'high': cu,
                    'error': child_error_mass,
                    'q_val': child_q_val
                })
        active_cells = unmarked_set + new_cells

    return history

def plot_geometric_decay_combined(w1_hist, w2_hist, theta, rho, mc_w1, mc_w2):
    iter_w1 = np.array([h['iter'] for h in w1_hist])
    err_w1 = np.array([h['gap_dyn'] for h in w1_hist]) / mc_w1

    iter_w2 = np.array([h['iter'] for h in w2_hist])
    err_w2 = np.array([h['gap_dyn'] for h in w2_hist]) / mc_w2

    q = 1.0 - theta * (1.0 - rho)
    theory_w1 = err_w1[0] * (q ** iter_w1)
    theory_w2 = err_w2[0] * (q ** iter_w2)

    plt.figure(figsize=(12, 9))
    plt.plot(iter_w1, err_w1, '-o', markersize=8, color='#D55E00', linewidth=2.5, label=r'Empirical $W^{1,2}$')
    plt.plot(iter_w1, theory_w1, '--', color='#D55E00', linewidth=2.0, alpha=0.7, label=rf'Theory $q={q:.4f}$')
    plt.plot(iter_w2, err_w2, '-s', markersize=8, color='#009E73', linewidth=2.5, label=r'Empirical $W^{2,2}$')
    plt.plot(iter_w2, theory_w2, ':', color='#009E73', linewidth=2.0, alpha=0.7, label=rf'Theory $q={q:.4f}$')

    plt.yscale('log')
    plt.title(rf"Geometric Convergence Decay | $\theta$={theta}", fontsize=global_size_para + 6)
    plt.xlabel("Iteration Step $k$", fontsize=global_size_para + 2)
    plt.ylabel("Normalized Global Error Bound", fontsize=global_size_para + 2)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"geometric_decay_combined_{LAYERS}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Combined Geometric Decay Plot to {save_path}")
    plt.close()

def run_independent_experiment():
    print(f"--- Starting Independent Refinement Experiment (Theorem Compliant) ---")

    X_np, y_np = generate_localized_data()
    print("Training Model...")
    model = FlexibleNet(LAYERS, activation=ACTIVATION)
    train_model(model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=6000)
    model.eval()

    domain_start, domain_end = 0.0, 2 * pi
    N_init = 2
    delta_init = ((domain_end - domain_start) / N_init) / 2.0
    centers_init = np.linspace(domain_start + delta_init, domain_end - delta_init, N_init).astype(np.float32)

    # 1. Plot the actual network fit vs mathematical truth
    plot_network_derivatives(model, domain_start, domain_end)

    # 2. Calculate Monte Carlo Ground Truth
    mc_w1p, mc_w2p = compute_monte_carlo_sobolev_1d(model, num_samples=100000, p=P_NORM, domain=(domain_start, domain_end))

    # 3. Run W1 Refinement (Using fast IBP)
    w1_history = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w1p', JACOBIAN_METHOD)

    # 4. Run W2 Refinement (Using fast IBP)
    w2_history = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w2p', JACOBIAN_METHOD)

    # 5. Plot Combined Geometric Decay
    print("\n--- Generating Combined Geometric Decay Plot ---")
    plot_geometric_decay_combined(w1_history, w2_history, THETA, RHO, mc_w1p, mc_w2p)

if __name__ == "__main__":
    run_independent_experiment()