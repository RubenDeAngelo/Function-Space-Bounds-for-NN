"""
run_pinn_sin_pde_geometric_decay.py

Tracks and plots the Geometric Decay and Local Error Reduction
for the Sin PDE: Delta u = -(x^2 + y^2) * sin(xy) using Adaptive Dörfler Marking.
Compliant with the Energy Norm Bound Reduction framework.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

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

from lib_nn_model import extract_numpy_params
from lib_nn_interval_ops import forward_pass, backward_pass
from lib_activation_enclosures import (
    compute_sigma_jac_intervals_all_layers,
    compute_sigma_hess_intervals_all_layers
)
from lib_jacobian_enclosures import compute_jac_bounds, compute_deltas, compute_omega
from lib_hessian_enclosures import compute_hessian_bounds_only

LAYERS = [2, 32, 32, 32, 1]
ACTIVATION = 'tanh'
WEIGHT_DECAY = 1e-4
EPOCHS = 6000

FIG_DIR = f"fig_pinn_sin_pde_geometric_{LAYERS}_wd{WEIGHT_DECAY}_{ACTIVATION}"
os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PATH = f"model_pinn_sin_pde_{LAYERS}_wd{WEIGHT_DECAY}_{ACTIVATION}.pth"

# Dörfler Parameters
THETA = 0.3
MAX_ITER = 30
RHO = 0.25  # Theoretical local error reduction factor (1/4)
HEATMAP_STEP = 5

# Toggles for features
PLOT_LOCAL_REDUCTION = True
NORMALIZE_ERROR = False  # Set to False to plot absolute errors and skip MC computation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_forcing_interval(x_c, y_c, delta):
    """
    Rigorous interval bounds [f_min, f_max] for f(x,y) = -(x^2 + y^2) * sin(xy).
    Because x,y in [0,1], x^2+y^2 and sin(xy) are strictly monotonically increasing.
    Therefore, the extrema lie exactly at the cell corners.
    """
    x_min = max(0.0, x_c - delta)
    x_max = x_c + delta
    y_min = max(0.0, y_c - delta)
    y_max = y_c + delta

    # A = x^2 + y^2
    a_min = x_min ** 2 + y_min ** 2
    a_max = x_max ** 2 + y_max ** 2

    # B = sin(xy)
    b_min = np.sin(x_min * y_min)
    b_max = np.sin(x_max * y_max)

    # f = - A * B
    f_max = -(a_min * b_min)  # Smallest subtracted magnitude = Maximum
    f_min = -(a_max * b_max)  # Largest subtracted magnitude = Minimum

    return f_min, f_max

def compute_mc_residual(model, num_samples=100000):
    """Computes the true integrated squared residual using Monte Carlo integration."""
    print(f"\n--- Computing Monte Carlo Ground Truth Residual (N={num_samples}) ---")
    model.eval()

    # Domain is [0, 1] x [0, 1], volume = 1.0
    raw_data = torch.rand((num_samples, 2), device=device)
    x_in = raw_data[:, 0:1].requires_grad_(True)
    y_in = raw_data[:, 1:2].requires_grad_(True)

    xy = torch.cat([x_in, y_in], dim=1)
    u = model(xy)

    d_xy = grad(u.sum(), [x_in, y_in], create_graph=True)
    u_x, u_y = d_xy[0], d_xy[1]

    u_xx = grad(u_x.sum(), x_in, create_graph=True)[0]
    u_yy = grad(u_y.sum(), y_in, create_graph=True)[0]

    # Forcing term: -(x^2 + y^2) * sin(xy)
    f = -(x_in ** 2 + y_in ** 2) * torch.sin(x_in * y_in)

    res_sq = (u_xx + u_yy - f) ** 2
    mc_integral = res_sq.mean().item() * 1.0  # Volume is 1.0

    print(f"MC Residual True Value: {mc_integral:.8f}")
    return mc_integral

def plot_pinn_heatmap(active_cells, iteration):
    """Generates a heatmap of the adaptive grid colored by local error gap."""
    fig, ax = plt.subplots(figsize=(10, 10))

    errors = [c['error_gap'] for c in active_cells]

    if len(errors) > 0:
        min_err, max_err = min(errors), max(errors)
        if max_err == min_err:
            max_err += 1e-9
    else:
        min_err, max_err = 0, 1

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_err, vmax=max_err)

    for cell, err in zip(active_cells, errors):
        cx, cy = cell['center'][0], cell['center'][1]
        delta = cell['delta']
        width = 2.0 * delta
        x0 = cx - delta
        y0 = cy - delta

        color = cmap(norm(err))
        rect = patches.Rectangle((x0, y0), width, width,
                                 linewidth=0.5, edgecolor='white', facecolor=color, alpha=0.9)
        ax.add_patch(rect)

    # PINN Domain is [0, 1] x [0, 1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"PINN Adaptive Grid - Iteration {iteration}\nN={len(active_cells)} Cells", fontsize=20)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Local Residual Error Gap $R^{\mathrm{res}}(f, K)$', fontsize=16)

    save_name = f"pinn_heatmap_iter_{iteration:03d}.pdf"
    save_path = os.path.join(FIG_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, format='pdf')
    plt.close()
    print(f"    [Plot] Saved heatmap to {save_path}")

def compute_cell_metrics(model, weights, xc, d):
    """
    Computes theorem-compliant metrics for a single cell.
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

    D_under = (H_low[0, 0, 0] + H_low[0, 1, 1]).item()
    D_over = (H_high[0, 0, 0] + H_high[0, 1, 1]).item()

    # Extract Forcing Term Bounds using the new Sin PDE function
    f_min, f_max = get_forcing_interval(xc[0], xc[1], d)

    # Upper Bound Calculation
    E_upper = max(abs(D_under - f_max) ** 2, abs(D_over - f_min) ** 2)

    # Lower Bound Calculation
    if f_max < D_under:
        E_lower = (D_under - f_max) ** 2
    elif D_under <= f_min and f_max <= D_over:
        E_lower = 0.0
    elif f_min > D_over:
        E_lower = (D_over - f_min) ** 2
    else:
        E_lower = 0.0

    vol = (2.0 * d) ** 2

    r_upper = vol * E_upper
    r_lower = vol * E_lower

    error_gap = r_upper - r_lower
    q_val = 0.5 * (r_upper + r_lower)

    return error_gap, q_val

def split_cell_2d(cell):
    c, d = cell['center'], cell['delta']
    new_d = d / 2.0
    offsets = list(itertools.product([-new_d, new_d], repeat=2))
    return [{'center': c + np.array(off, dtype=np.float32), 'delta': new_d} for off in offsets]

def run_pinn_doerfler_refinement_tracked(model, max_iter=40, theta=0.3, track_splits=True):
    model.eval()
    weights, _ = extract_numpy_params(model)

    # Initial Coarse Grid for [0,1]x[0,1]
    active_cells = [{'center': np.array(c, dtype=np.float32), 'delta': 0.25}
                    for c in itertools.product([0.25, 0.75], repeat=2)]

    history = []
    split_data = []

    print(f"\n  Starting Adaptive Refinement (Theta={theta})...")

    for iteration in range(max_iter + 1):
        # 1. Calculation Step
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

        print(f"    Iter {iteration:<3}: Cells={len(active_cells):<5}, Gap={total_error_gap:.6e}")

        if iteration % HEATMAP_STEP == 0 or iteration == max_iter:
            plot_pinn_heatmap(active_cells, iteration)

        if iteration == max_iter: break

        # 2. Marking Step
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

        # 3. Refining Step
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

            if track_splits:
                sum_child_error = sum(c['error_gap'] for c in children_for_this_parent)
                split_data.append((parent['error_gap'], sum_child_error))

        active_cells = unmarked_set + new_cells

    return history, np.array(split_data) if track_splits else np.array([])

def plot_geometric_decay(history, theta, rho, mc_val=None, normalize=True):
    iterations = np.array([h['iter'] for h in history])
    empirical_gaps = np.array([h['gap_dyn'] for h in history])

    # Conditional Normalization
    if normalize and mc_val is not None:
        empirical_gaps = empirical_gaps / mc_val
        y_label = r"Normalized Residual Bound Gap"
    else:
        y_label = r"Global Residual Error Bound"

    # Calculate Theoretical Line: R_k = R_0 * (1 - theta * (1 - rho))^k
    q = 1.0 - theta * (1.0 - rho)
    R_0 = empirical_gaps[0]
    theory_gaps = R_0 * (q ** iterations)

    plt.figure(figsize=(10, 8))
    plt.plot(iterations, empirical_gaps, '-o', markersize=8, color='#4A7BC7', linewidth=2.5, label='Empirical Error')

    # Explicitly print the actual q value in the label
    theory_label = rf'Theory $q={q:.4f}$'
    plt.plot(iterations, theory_gaps, '--k', linewidth=2.5, label=theory_label)

    plt.yscale('log')
    plt.title(rf"Geometric Convergence Decay | $\theta$={theta}", fontsize=global_size_para + 2)
    plt.xlabel("Iteration Step $k$", fontsize=global_size_para)
    plt.ylabel(y_label, fontsize=global_size_para)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "geometric_decay_pinn_sin.pdf")
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

    # Protect against min_err exactly 0 for log10
    min_err = max(min_err, 1e-16)

    x_theory = np.logspace(np.log10(min_err / 10), np.log10(max_err * 10), 100)
    y_theory = rho * x_theory

    theory_label = rf'Theory $y = {rho:.2f} x$'
    plt.plot(x_theory, y_theory, '--k', linewidth=2.5, label=theory_label)

    # Red line for y = x
    plt.plot(x_theory, x_theory, '-r', linewidth=2.0, label=r'$y = x$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(r"Local Error Reduction", fontsize=global_size_para + 2)
    plt.xlabel(r"Parent Error", fontsize=global_size_para)
    plt.ylabel(r"Sum Children Error", fontsize=global_size_para)

    plt.xlim(min_err / 5, max_err * 5)
    plt.ylim(min_err / 5, max_err * 5)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, "local_error_reduction_pinn_sin.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Local Error Reduction Plot to {save_path}")
    plt.close()

def run_experiment():
    print("--- 1. Setting up Model & Training ---")
    torch.manual_seed(42)
    np.random.seed(42)
    model = PINNNet(LAYERS, activation=ACTIVATION).to(device)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing trained model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"No trained model found. Starting lib_training for {EPOCHS} epochs...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            # Domain is [0,1]x[0,1]
            raw_data = torch.rand((1000, 2), device=device)
            x_in = raw_data[:, 0:1].requires_grad_(True)
            y_in = raw_data[:, 1:2].requires_grad_(True)
            xy = torch.cat([x_in, y_in], dim=1)

            u = model(xy)

            # Residual Loss calculation
            d_xy = grad(u.sum(), [x_in, y_in], create_graph=True)
            u_x, u_y = d_xy[0], d_xy[1]
            u_xx = grad(u_x.sum(), x_in, create_graph=True)[0]
            u_yy = grad(u_y.sum(), y_in, create_graph=True)[0]

            # New Forcing term
            f = -(x_in ** 2 + y_in ** 2) * torch.sin(x_in * y_in)

            loss_res = nn.MSELoss()(u_xx + u_yy, f)

            # Boundary Loss calculation (u*(x,y) = sin(xy))
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

            # Target boundary values mapped to sin(xy)
            u_target = torch.sin(xy_bnd[:, 0:1] * xy_bnd[:, 1:2])

            u_pred_bnd = model(xy_bnd)
            loss_bnd = nn.MSELoss()(u_pred_bnd, u_target)

            loss = loss_res + loss_bnd
            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"  Epoch {epoch}, Total Loss: {loss.item():.6f}, Res Loss: {loss_res.item():.6f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.")

    mc_truth = None
    if NORMALIZE_ERROR:
        mc_truth = compute_mc_residual(model)
    else:
        print("\n--- 2. Skipping Monte Carlo Truth (Normalization Disabled) ---")

    print("\n--- 3. Analysis ---")
    hist, splits = run_pinn_doerfler_refinement_tracked(model, MAX_ITER, THETA, track_splits=PLOT_LOCAL_REDUCTION)

    print("\n--- 4. Generating Plots ---")
    plot_geometric_decay(hist, THETA, RHO, mc_val=mc_truth, normalize=NORMALIZE_ERROR)
    if PLOT_LOCAL_REDUCTION and len(splits) > 0:
        plot_local_error_reduction(splits, RHO)

if __name__ == "__main__":
    run_experiment()