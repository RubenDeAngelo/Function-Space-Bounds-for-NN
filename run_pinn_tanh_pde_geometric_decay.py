"""
run_pinn_tanh_pde_geometric_decay.py

Tracks and plots the Geometric Decay and Local Error Reduction
for the Tanh Wavefront PDE using Adaptive Dörfler Marking.
Compliant with the Energy Norm Bound Reduction framework.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import itertools

# --- Global Font Size Configuration ---
global_size_para = 16
plt.rcParams.update({
    'font.size': global_size_para,
    'axes.titlesize': global_size_para + 2,
    'axes.labelsize': global_size_para,
    'xtick.labelsize': global_size_para - 2,
    'ytick.labelsize': global_size_para - 2,
    'legend.fontsize': global_size_para - 2,
    'figure.titlesize': global_size_para + 4
})

# --- Import from your library ---
from flex_model import extract_numpy_params
from network_ops import forward_pass, backward_pass
from activations import (
    compute_sigma_jac_intervals_all_layers,
    compute_sigma_hess_intervals_all_layers
)
from jacobian_bounds import compute_jac_bounds, compute_deltas, compute_omega
from hessian_bounds import compute_hessian_bounds_only

# --- CONFIGURATION ---
LAYERS = [2, 32, 32, 32, 32, 1]
ACTIVATION = 'tanh'
EPOCHS = 6000

# PDE Parameters
K_PARAM = 5.0  # Steepness parameter for z = k(x + y - 1)

FIG_DIR = f"fig_pinn_tanh_pde_geometric_decay"
os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PATH = f"model_pinn_tanh_pde_{LAYERS}.pth"

# Dörfler Parameters
THETA = 0.32
MAX_ITER = 30  # Increased for geometric decay visualization
RHO = 0.25  # Theoretical local error reduction factor (1/4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- MODEL DEFINITION ---
class PINNNet(nn.Module):
    def __init__(self, layer_sizes, activation='tanh'):
        super().__init__()
        self.activation_name = activation.lower()
        self.activation_fn = nn.Sigmoid() if self.activation_name == 'sigmoid' else nn.Tanh()
        layers_list = []
        for i in range(len(layer_sizes) - 2):
            layers_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers_list.append(self.activation_fn)
        layers_list.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)


# --- HELPERS ---
def get_forcing_interval(x_c, y_c, delta, k=K_PARAM):
    """
    Rigorous interval bounds [f_min, f_max] for f(z) = -4k^2(1 - tanh^2(z))tanh(z)
    where z = k(x + y - 1).
    """
    z_min = k * (x_c - delta + y_c - delta - 1.0)
    z_max = k * (x_c + delta + y_c + delta - 1.0)

    def f_val(z):
        t = np.tanh(z)
        return -4.0 * (k ** 2) * (1.0 - t ** 2) * t

    vals = [f_val(z_min), f_val(z_max)]

    # Critical points of f(z) occur where tanh^2(z) = 1/3
    z_c1 = np.arctanh(1.0 / np.sqrt(3.0))
    z_c2 = -np.arctanh(1.0 / np.sqrt(3.0))

    if z_min < z_c1 < z_max:
        vals.append(f_val(z_c1))
    if z_min < z_c2 < z_max:
        vals.append(f_val(z_c2))

    return min(vals), max(vals)


# --- CORE BOUNDING LOGIC ---
def compute_cell_metrics(model, weights, xc, d):
    """
    Computes theorem-compliant metrics for a single cell.
    Returns:
      error_gap: R_res(f, K)
      q_val:     Q_n(f, K)
    """
    x_in = xc.reshape(1, -1)
    x_l, z_l = forward_pass(model, x_in.T)
    delta_prime_jac, _, _ = compute_deltas(model, x_l, d)
    omega_all = compute_omega(len(weights), weights, delta_prime_jac)
    J_all = backward_pass(model, z_l)
    alpha_all, beta_all = compute_sigma_jac_intervals_all_layers(model, z_l, omega_all)
    gamma_all, eta_all = compute_sigma_hess_intervals_all_layers(model, z_l, omega_all)
    jac_bound_up, jac_bound_down = compute_jac_bounds(weights, alpha_all, beta_all, 1)

    H_low, H_high = compute_hessian_bounds_only(
        model, xc, d, jacobian_bound_method='ibp',
        entries_to_compute=[(0, 0), (1, 1)],
        weights=weights, x_l=x_l, z_l=z_l,
        delta_prime_jac=delta_prime_jac, omega_all=omega_all, J_all=J_all,
        alpha_all=alpha_all, beta_all=beta_all, gamma_all=gamma_all, eta_all=eta_all,
        jac_bound_up=jac_bound_up, jac_bound_down=jac_bound_down
    )

    # 1. Extract Differential Operator Bounds
    D_under = (H_low[0, 0, 0] + H_low[0, 1, 1]).item()
    D_over = (H_high[0, 0, 0] + H_high[0, 1, 1]).item()

    # 2. Extract Forcing Term Bounds
    f_min, f_max = get_forcing_interval(xc[0], xc[1], d)

    # 3. Upper Bound Calculation
    E_upper = max(abs(D_under - f_max) ** 2, abs(D_over - f_min) ** 2)

    # 4. Lower Bound Calculation (Piecewise logic)
    if f_max < D_under:
        E_lower = (D_under - f_max) ** 2
    elif D_under <= f_min and f_max <= D_over:
        E_lower = 0.0
    elif f_min > D_over:
        E_lower = (D_over - f_min) ** 2
    else:
        E_lower = 0.0

    vol = (2.0 * d) ** 2

    # Theorem Compliant Metrics
    r_upper = vol * E_upper
    r_lower = vol * E_lower

    error_gap = r_upper - r_lower
    q_val = 0.5 * (r_upper + r_lower)

    return error_gap, q_val


# --- DÖRFLER REFINEMENT & SPLIT TRACKING LOGIC ---
def split_cell_2d(cell):
    c, d = cell['center'], cell['delta']
    new_d = d / 2.0
    offsets = list(itertools.product([-new_d, new_d], repeat=2))
    return [{'center': c + np.array(off, dtype=np.float32), 'delta': new_d} for off in offsets]


def run_pinn_doerfler_refinement_tracked(model, max_iter=40, theta=0.3):
    model.eval()
    weights, _ = extract_numpy_params(model)

    active_cells = [{'center': np.array(c, dtype=np.float32), 'delta': 0.25}
                    for c in itertools.product([0.25, 0.75], repeat=2)]

    history = []
    split_data = []  # To store (parent_error, sum_children_error)

    print(f"\n--- Starting Adaptive Refinement (Theta={theta}) ---")

    for iteration in range(max_iter + 1):
        # Calculation Step
        for cell in active_cells:
            if 'error_gap' not in cell:
                err_gap, q = compute_cell_metrics(model, weights, cell['center'], cell['delta'])
                cell['error_gap'] = err_gap
                cell['q_val'] = q

        total_error_gap = sum(c['error_gap'] for c in active_cells)

        history.append({
            'iter': iteration,
            'count': len(active_cells),
            'gap_dyn': total_error_gap
        })

        print(f"Iter {iteration:<3}: Cells={len(active_cells):<5}, Gap={total_error_gap:.6f}")

        if iteration == max_iter: break

        # Marking & Refining Step
        active_cells.sort(key=lambda c: c['error_gap'], reverse=True)
        target = theta * total_error_gap
        current_sum = 0.0
        split_idx = 0

        for i, cell in enumerate(active_cells):
            current_sum += cell['error_gap']
            if current_sum >= target:
                split_idx = i + 1
                break

        if split_idx == 0: split_idx = 1

        new_cells = []
        unmarked_set = active_cells[split_idx:]

        for parent in active_cells[:split_idx]:
            children_for_this_parent = split_cell_2d(parent)

            # Evaluate children immediately to track local error reduction
            for child in children_for_this_parent:
                err_gap, q = compute_cell_metrics(model, weights, child['center'], child['delta'])
                child['error_gap'] = err_gap
                child['q_val'] = q

            new_cells.extend(children_for_this_parent)

            # Extract Local Error Reduction Data
            sum_child_error = sum(c['error_gap'] for c in children_for_this_parent)
            split_data.append((parent['error_gap'], sum_child_error))

        active_cells = unmarked_set + new_cells

    return history, np.array(split_data)


# --- PLOTTING FUNCTIONS ---
def plot_geometric_decay(history, theta, rho):
    iterations = np.array([h['iter'] for h in history])
    empirical_gaps = np.array([h['gap_dyn'] for h in history])

    # Calculate Theoretical Line: R_k = R_0 * (1 - theta * (1 - rho))^k
    q = 1.0 - theta * (1.0 - rho)
    R_0 = empirical_gaps[0]
    theory_gaps = R_0 * (q ** iterations)

    plt.figure(figsize=(10, 8))
    plt.plot(iterations, empirical_gaps, '-o', markersize=8, color='#4A7BC7', linewidth=2.5, label='Empirical Error')

    theory_label = r'Theory $(1 - ' + str(
        int((1 - rho) * 4)) + r'\theta/4)^k$' if rho == 0.25 else f'Theory $q={q:.3f}$'
    plt.plot(iterations, theory_gaps, '--k', linewidth=2.5, label=theory_label)

    plt.yscale('log')
    plt.title("Geometric Convergence Decay (Residual Bound)", fontsize=20)
    plt.xlabel("Iteration Step $k$", fontsize=18)
    plt.ylabel("Global Residual Error Bound $R_k^{\mathrm{res}}(f)$", fontsize=18)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "geometric_decay_pinn_tanh.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Geometric Decay Plot to {save_path}")
    plt.close()


def plot_local_error_reduction(split_data, rho):
    parent_errors = split_data[:, 0]
    children_errors = split_data[:, 1]

    plt.figure(figsize=(10, 8))

    # Plot Scatter
    plt.scatter(parent_errors, children_errors, color='#85C57A', alpha=0.5, s=30, label='Splits')

    # Plot Theoretical Line: y = rho * x
    min_err = min(np.min(parent_errors), np.min(children_errors))
    max_err = max(np.max(parent_errors), np.max(children_errors))

    x_theory = np.logspace(np.log10(min_err / 10), np.log10(max_err * 10), 100)
    y_theory = rho * x_theory

    theory_label = r'Theory $y = 1/4 x$' if rho == 0.25 else f'Theory $y = {rho} x$'
    plt.plot(x_theory, y_theory, '--k', linewidth=2.5, label=theory_label)

    # Red line for y = x
    plt.plot(x_theory, x_theory, '-r', linewidth=2.0, label=r'$y = x$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(r"Local Error Reduction ($R_K \to \sum R_{K'}$)", fontsize=20)
    plt.xlabel(r"Parent Error $R(K)$", fontsize=18)
    plt.ylabel(r"Sum Children Error $\sum R_{K'}$", fontsize=18)

    plt.xlim(min_err / 5, max_err * 5)
    plt.ylim(min_err / 5, max_err * 5)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "local_error_reduction_pinn_tanh.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Local Error Reduction Plot to {save_path}")
    plt.close()


# --- MAIN EXPERIMENT ---
def run_experiment():
    print("--- 1. Setting up Model & Training ---")
    torch.manual_seed(42)
    np.random.seed(42)
    model = PINNNet(LAYERS, activation=ACTIVATION).to(device)

    # MODEL LOAD / SAVE LOGIC
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing trained model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"No trained model found. Starting training for {EPOCHS} epochs...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            raw_data = torch.rand((1000, 2), device=device)
            x_in = raw_data[:, 0:1].requires_grad_(True)
            y_in = raw_data[:, 1:2].requires_grad_(True)
            xy = torch.cat([x_in, y_in], dim=1)

            u = model(xy)

            d_xy = grad(u.sum(), [x_in, y_in], create_graph=True)
            u_x, u_y = d_xy[0], d_xy[1]
            u_xx = grad(u_x.sum(), x_in, create_graph=True)[0]
            u_yy = grad(u_y.sum(), y_in, create_graph=True)[0]

            z = K_PARAM * (x_in + y_in - 1.0)
            tanh_z = torch.tanh(z)
            f = -4.0 * (K_PARAM ** 2) * (1.0 - tanh_z ** 2) * tanh_z

            loss_res = nn.MSELoss()(u_xx + u_yy, f)

            # Boundary Loss
            x_bnd = torch.rand((200, 1), device=device)
            y_bnd_0 = torch.zeros_like(x_bnd)
            y_bnd_1 = torch.ones_like(x_bnd)

            y_bnd = torch.rand((200, 1), device=device)
            x_bnd_0 = torch.zeros_like(y_bnd)
            x_bnd_1 = torch.ones_like(y_bnd)

            xy_bnd = torch.cat([
                torch.cat([x_bnd, y_bnd_0], dim=1),
                torch.cat([x_bnd, y_bnd_1], dim=1),
                torch.cat([x_bnd_0, y_bnd], dim=1),
                torch.cat([x_bnd_1, y_bnd], dim=1)
            ], dim=0)

            z_bnd = K_PARAM * (xy_bnd[:, 0:1] + xy_bnd[:, 1:2] - 1.0)
            u_target = torch.tanh(z_bnd)

            u_pred_bnd = model(xy_bnd)
            loss_bnd = nn.MSELoss()(u_pred_bnd, u_target)

            loss = loss_res + loss_bnd
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"  Epoch {epoch}, Total Loss: {loss.item():.6f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.")

    print("\n--- 2. Analysis ---")
    hist, splits = run_pinn_doerfler_refinement_tracked(model, MAX_ITER, THETA)

    print("\n--- 3. Generating Plots ---")
    plot_geometric_decay(hist, THETA, RHO)
    plot_local_error_reduction(splits, RHO)


if __name__ == "__main__":
    run_experiment()