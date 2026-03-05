"""
run_pinn_tanh_pde_theorem_compliant.py

Calculation of Interior Residual Bounds using Adaptive Dörfler Marking.
Compliant with the Energy Norm Bound Reduction framework.
Configured for the Tanh PDE: Delta u = -4k^2(1 - tanh^2(z))tanh(z)
Includes Heatmaps visualizing the adaptive grid refinement.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

# --- Global Font Size Configuration ---
global_size_para = 20
plt.rcParams.update({
    'font.size': global_size_para,
    'axes.titlesize': global_size_para ,
    'axes.labelsize': global_size_para,
    'xtick.labelsize': global_size_para,
    'ytick.labelsize': global_size_para,
    'legend.fontsize': global_size_para,
    'figure.titlesize': global_size_para
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
ACTIVATION = 'sigmoid'
EPOCHS = 6000

# PDE Parameters
K_PARAM = 5.0  # Steepness parameter for z = k(x + y - 1)

FIG_DIR = f"fig_pinn_tanh_pde_compliant"
os.makedirs(FIG_DIR, exist_ok=True)
MODEL_PATH = f"model_pinn_tanh_pde_{LAYERS}.pth"

# Dörfler Parameters
THETA = 0.3
MAX_ITER = 30
STATIC_STEP = 5
HEATMAP_STEP = 5

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
    # 1. Determine the interval of z for the current cell
    z_min = k * (x_c - delta + y_c - delta - 1.0)
    z_max = k * (x_c + delta + y_c + delta - 1.0)

    def f_val(z):
        t = np.tanh(z)
        return -4.0 * (k ** 2) * (1.0 - t ** 2) * t

    # Evaluate at the interval boundaries
    vals = [f_val(z_min), f_val(z_max)]

    # 2. Add critical points of f(z) to the evaluation pool
    # The derivative of f(z) w.r.t z roots at tanh^2(z) = 1/3
    z_c1 = np.arctanh(1.0 / np.sqrt(3.0))
    z_c2 = -np.arctanh(1.0 / np.sqrt(3.0))

    if z_min < z_c1 < z_max:
        vals.append(f_val(z_c1))
    if z_min < z_c2 < z_max:
        vals.append(f_val(z_c2))

    return min(vals), max(vals)


def compute_mc_residual(model, num_samples=100000, k=K_PARAM):
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

    # Forcing term
    z = k * (x_in + y_in - 1.0)
    tanh_z = torch.tanh(z)
    f = -4.0 * (k ** 2) * (1.0 - tanh_z ** 2) * tanh_z

    res_sq = (u_xx + u_yy - f) ** 2
    mc_integral = res_sq.mean().item() * 1.0  # Volume is 1.0

    print(f"MC Residual True Value: {mc_integral:.8f}")
    return mc_integral


# --- PLOTTING HELPER (HEATMAP) ---
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

    # 1. Extract Differential Operator Bounds (D_under and D_over)
    D_under = (H_low[0, 0, 0] + H_low[0, 1, 1]).item()
    D_over = (H_high[0, 0, 0] + H_high[0, 1, 1]).item()

    # 2. Extract Forcing Term Bounds using the new Tanh function
    f_min, f_max = get_forcing_interval(xc[0], xc[1], d)

    # 3. Upper Bound Calculation (Theorem Compliant)
    E_upper = max(abs(D_under - f_max) ** 2, abs(D_over - f_min) ** 2)

    # 4. Lower Bound Calculation (Theorem Compliant Piecewise Logic)
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


# --- STATIC GRID BOUNDER ---
def compute_static_grid_bound(model, target_N):
    model.eval()
    weights, _ = extract_numpy_params(model)
    k = int(np.round(np.sqrt(target_N)))
    if k < 1: k = 1
    real_N = k * k
    delta = 1.0 / (2.0 * k)
    centers_1d = np.linspace(delta, 1.0 - delta, k)

    total_error_gap = 0.0
    total_q_val = 0.0

    grid_points = itertools.product(centers_1d, repeat=2)

    for x, y in grid_points:
        xc = np.array([x, y], dtype=np.float32)
        err_gap, q = compute_cell_metrics(model, weights, xc, delta)
        total_error_gap += err_gap
        total_q_val += q

    return real_N, total_error_gap, total_q_val


# --- DÖRFLER REFINEMENT LOGIC ---
def split_cell_2d(cell):
    c, d = cell['center'], cell['delta']
    new_d = d / 2.0
    offsets = list(itertools.product([-new_d, new_d], repeat=2))
    return [{'center': c + np.array(off, dtype=np.float32), 'delta': new_d} for off in offsets]


def run_pinn_doerfler_refinement(model, max_iter=10, theta=0.5):
    model.eval()
    weights, _ = extract_numpy_params(model)

    # Initial Coarse Grid for [0,1]x[0,1]
    active_cells = [{'center': np.array(c, dtype=np.float32), 'delta': 0.25}
                    for c in itertools.product([0.25, 0.75], repeat=2)]

    history = []
    print(f"\n  Starting Adaptive Refinement (Theta={theta})...")

    for iteration in range(max_iter + 1):
        for cell in active_cells:
            if 'error_gap' not in cell:
                err_gap, q = compute_cell_metrics(model, weights, cell['center'], cell['delta'])
                cell['error_gap'] = err_gap
                cell['q_val'] = q

        total_error_gap = sum(c['error_gap'] for c in active_cells)
        total_q_val = sum(c['q_val'] for c in active_cells)

        history.append({
            'count': len(active_cells),
            'gap_dyn': total_error_gap,
            'q_dyn': total_q_val
        })

        print(
            f"    Iter {iteration:<3}: Cells={len(active_cells):<5}, Gap={total_error_gap:.6f}, Q_n={total_q_val:.6f}")

        if iteration % HEATMAP_STEP == 0 or iteration == max_iter:
            plot_pinn_heatmap(active_cells, iteration)

        if iteration == max_iter: break

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
        for parent in active_cells[:split_idx]:
            new_cells.extend(split_cell_2d(parent))

        active_cells = active_cells[split_idx:] + new_cells

    return history


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
            z = K_PARAM * (x_in + y_in - 1.0)
            tanh_z = torch.tanh(z)
            f = -4.0 * (K_PARAM ** 2) * (1.0 - tanh_z ** 2) * tanh_z

            loss_res = nn.MSELoss()(u_xx + u_yy, f)

            # Boundary Loss calculation (u*(x,y) = tanh(z))
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

            # Target boundary values mapped to z
            z_bnd = K_PARAM * (xy_bnd[:, 0:1] + xy_bnd[:, 1:2] - 1.0)
            u_target = torch.tanh(z_bnd)

            u_pred_bnd = model(xy_bnd)
            loss_bnd = nn.MSELoss()(u_pred_bnd, u_target)

            # Total Loss
            loss = loss_res + loss_bnd

            loss.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                print(f"  Epoch {epoch}, Total Loss: {loss.item():.6f}, Res Loss: {loss_res.item():.6f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.")

    print("\n--- 2. Analysis ---")
    mc_truth = compute_mc_residual(model, k=K_PARAM)

    dyn_history = run_pinn_doerfler_refinement(model, MAX_ITER, THETA)

    dyn_counts = [h['count'] for h in dyn_history]
    dyn_gaps = np.array([h['gap_dyn'] for h in dyn_history])
    dyn_qs = np.array([h['q_dyn'] for h in dyn_history])

    print(f"\n  Computing Static Baselines (Every {STATIC_STEP} steps)...")
    idxs = list(range(0, len(dyn_counts), STATIC_STEP))
    if idxs[-1] != len(dyn_counts) - 1: idxs.append(len(dyn_counts) - 1)

    stat_counts = []
    stat_gaps = []
    stat_qs = []

    for i in idxs:
        target_n = dyn_counts[i]
        real_n, s_gap, s_q = compute_static_grid_bound(model, target_n)
        stat_counts.append(real_n)
        stat_gaps.append(s_gap)
        stat_qs.append(s_q)

    # --- Plotting ---
    print("\n--- 3. Generating Plots ---")

    # Normalize gaps for plotting
    dyn_gaps_norm = dyn_gaps / mc_truth
    stat_gaps_norm = np.array(stat_gaps) / mc_truth

    # 1. Residual Convergence Plot
    plt.figure(figsize=(12, 9))
    n0, e0 = dyn_counts[0], dyn_gaps_norm[0]
    x_theory = np.linspace(n0, dyn_counts[-1], 100)
    plt.plot(x_theory, e0 * (x_theory / n0) ** (-1), 'k--', linewidth=2, label=r'Optimal Rate $O(N^{-1})$')

    plt.plot(dyn_counts, dyn_gaps_norm, '-o', color='#D55E00', linewidth=2.5, markersize=8,
             label=rf'Dörfler Adaptive ($\theta={THETA}$)')
    plt.plot(stat_counts, stat_gaps_norm, '--s', color='#D55E00', alpha=0.5, linewidth=2, label='Static Grid')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Grid Boxes (N)')
    plt.ylabel(r'Normalized Residual Bound Gap $R^{\mathrm{res}}_n(f) / MC$')
    plt.title('PINN Interior Residual Bound Convergence (Tanh PDE)')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    p1_path = os.path.join(FIG_DIR, "pinn_residual_convergence_tanh_pde.pdf")
    plt.savefig(p1_path, dpi=300, format='pdf')
    print(f"Saved Residual Convergence to {p1_path}")
    plt.close()

    # 2. Integral Quadrature Plot
    plt.figure(figsize=(12, 9))
    plt.plot(dyn_counts, dyn_qs, '-o', color='#D55E00', linewidth=2.5, markersize=8, label=r'Adaptive $Q_n(f)$')

    plt.fill_between(dyn_counts,
                     dyn_qs - 0.5 * dyn_gaps,
                     dyn_qs + 0.5 * dyn_gaps,
                     color='#D55E00', alpha=0.2,
                     label=r'Uncertainty $\pm \frac{1}{2} R_n^{\mathrm{res}}(f)$')

    plt.plot(stat_counts, stat_qs, '--s', color='#D55E00', alpha=0.5, linewidth=2, label=r'Static $Q_n(f)$')
    plt.axhline(y=mc_truth, color='black', linestyle='-.', linewidth=2, alpha=0.8, label='MC True Residual')

    plt.xscale('log')
    plt.xlabel('Number of Grid Boxes (N)')
    plt.ylabel(r'Calculated Residual Energy $Q_n(f)$')
    plt.title('PINN Residual Quadrature Convergence (Tanh PDE)')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    p2_path = os.path.join(FIG_DIR, "pinn_quadrature_convergence_tanh_pde.pdf")
    plt.savefig(p2_path, dpi=300, format='pdf')
    print(f"Saved Quadrature Convergence to {p2_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment()