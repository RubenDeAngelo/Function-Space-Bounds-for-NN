"""
run_doerfler_theorem_compliant_2d.py

Comparison of W1p and W2p norms using Dynamic Refinement with DÖRFLER MARKING.
ADAPTED FOR 2D "SMOOTH COIN" FUNCTION.
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

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

# --- Imports (Assumed to be in the same directory) ---
from flex_model import FlexibleNet
from training import train_model
from w1p_bounds import compute_w1p_bounds
from w2p_bounds import compute_w2p_bounds

# Data Generation Settings
EPSILON_TRANSITION = 0.5


# Model & Training
LAYERS = [2, 32, 32, 32, 1]
weight_decay = 1e-4
ACTIVATION = 'tanh'
EPOCHS = 10000
N_SAMPLES = 5000
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}_{EPSILON_TRANSITION}_wd{weight_decay}_{ACTIVATION}.pth"

# --- CONFIGURATION ---
FIG_DIR = f"fig_doerfler_heatmaps_compliant_{LAYERS}_{EPSILON_TRANSITION}_wd{weight_decay}_{ACTIVATION}"
os.makedirs(FIG_DIR, exist_ok=True)

# Refinement Settings
P_NORM = 2.0
MAX_ITER = 3
THETA = 0.35
JACOBIAN_METHOD = 'ibp'  # Options: 'recurjac' or 'ibp'


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


def compute_monte_carlo_sobolev(model, num_samples=100000, p=2.0, domain=(-1.5, 1.5)):
    print(f"\n--- Computing Monte Carlo Ground Truth (N={num_samples}) ---")

    device = next(model.parameters()).device
    X = torch.empty(num_samples, 2, device=device).uniform_(domain[0], domain[1]).requires_grad_(True)

    volume = (domain[1] - domain[0]) ** 2

    Y = model(X)

    grad_outputs = torch.ones_like(Y)
    grad_Y = torch.autograd.grad(Y, X, grad_outputs=grad_outputs, create_graph=True)[0]

    hess_x1 = torch.autograd.grad(grad_Y[:, 0], X, grad_outputs=torch.ones_like(grad_Y[:, 0]), retain_graph=True)[0]
    hess_x2 = torch.autograd.grad(grad_Y[:, 1], X, grad_outputs=torch.ones_like(grad_Y[:, 1]), retain_graph=False)[0]

    phi_p = torch.abs(Y).pow(p).sum(dim=1)
    grad_p = torch.abs(grad_Y).pow(p).sum(dim=1)
    hess_p = torch.abs(hess_x1).pow(p).sum(dim=1) + torch.abs(hess_x2).pow(p).sum(dim=1)

    w1p_integrand = phi_p + grad_p
    w2p_integrand = phi_p + grad_p + hess_p

    w1p_mc = volume * w1p_integrand.mean().item()
    w2p_mc = volume * w2p_integrand.mean().item()

    print(f"MC W1P True Value: {w1p_mc:.6f}")
    print(f"MC W2P True Value: {w2p_mc:.6f}")
    return w1p_mc, w2p_mc


def generate_2d_coin_data(N=3000, epsilon=0.1):
    low, high = -1.5, 1.5
    X = np.random.uniform(low, high, (N, 2)).astype(np.float32)
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return X, y.reshape(-1, 1).astype(np.float32)


# --- PLOTTING UTILS ---

def plot_grid_heatmap(active_cells, iteration, norm_type, p=2.0, domain=(-1.5, 1.5)):
    fig, ax = plt.subplots(figsize=(12, 12))

    visual_errors = [c['error'] for c in active_cells]

    if len(visual_errors) > 0:
        min_err, max_err = min(visual_errors), max(visual_errors)
        if max_err == min_err: max_err += 1e-9
    else:
        min_err, max_err = 0, 1

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min_err, vmax=max_err)

    for cell, v_err in zip(active_cells, visual_errors):
        cx, cy = cell['center'][0], cell['center'][1]
        delta = cell['delta']
        width = 2.0 * delta
        x0 = cx - delta
        y0 = cy - delta
        color = cmap(norm(v_err))
        rect = patches.Rectangle((x0, y0), width, width,
                                 linewidth=0.5, edgecolor='white', facecolor=color, alpha=0.9)
        ax.add_patch(rect)

    circle = plt.Circle((0, 0), 1.0, color='red', fill=False, linestyle='--', linewidth=1.5, label='Inner Radius')
    circle_outer = plt.Circle((0, 0), 1.0 + EPSILON_TRANSITION, color='orange', fill=False, linestyle='--',
                              linewidth=1.5, label='Outer Radius')
    ax.add_patch(circle)
    ax.add_patch(circle_outer)

    ax.set_xlim(domain[0], domain[1])
    ax.set_ylim(domain[0], domain[1])
    ax.set_title(f"{norm_type.upper()} Adaptive Grid - Iteration {iteration}\nN={len(active_cells)} Boxes", fontsize=30)
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Local Gap', fontsize=30)

    save_name = f"heatmap_theorem_compliant_{norm_type}_iter_{iteration:03d}.pdf"
    save_path = os.path.join(FIG_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    [Plot] Saved heatmap to {save_path}")


# --- UTILITIES ---

def split_points(points, delta):
    new_points = []
    new_delta = delta / 2.0
    d = points[0].shape[0]
    offset_combinations = list(itertools.product([-1, 1], repeat=d))
    for x in points:
        for factors in offset_combinations:
            offset = np.array(factors).reshape(x.shape) * new_delta
            new_points.append(x + offset)
    return new_points, new_delta


# --- DÖRFLER REFINEMENT LOGIC ---
def run_doerfler_grid_for_norm(model, initial_centers, initial_delta, max_iter, theta=0.5, p=2.0, norm_type='w1p', jac_method='recurjac'):
    print(f"\n--- Running Dörfler Refinement (theta={theta}, p={p}, method={jac_method}) for {norm_type.upper()} ---")

    active_cells = []
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    if np.isscalar(initial_delta):
        initial_batch = [(c, initial_delta) for c in initial_centers]
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
        print(f"Iter {i:<3} | Cells: {count:<5} | {norm_type.upper()} Gap: {global_gap:.6f} | Q_n: {global_q:.6f}")

        if i % 5 == 0 or i == max_iter:
            plot_grid_heatmap(active_cells, i, norm_type, p=p)

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

# --- MAIN EXPERIMENT ---

def run_experiment():
    print(f"--- Starting Dörfler Refinement Experiment (Theorem Compliant) ---")

    X_np, y_np = generate_2d_coin_data(N=N_SAMPLES, epsilon=EPSILON_TRANSITION)
    model = FlexibleNet(LAYERS, activation=ACTIVATION)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"No trained model found. Training on {len(X_np)} samples...")
        train_model(model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=EPOCHS, weight_decay=weight_decay)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}.")

    model.eval()

    domain_start, domain_end = -1.5, 1.5
    N_side = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # Reiche JACOBIAN_METHOD als Parameter durch
    w1_history = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w1p', JACOBIAN_METHOD)
    w2_history = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w2p', JACOBIAN_METHOD)

    print("\n--- Computing Static Baselines (Subsampled) ---")
    w1_counts = [h['count'] for h in w1_history]
    w1_dyn_gaps = np.array([h['gap_dyn'] for h in w1_history])
    w1_dyn_qs = np.array([h['q_dyn'] for h in w1_history])

    w2_counts = [h['count'] for h in w2_history]
    w2_dyn_gaps = np.array([h['gap_dyn'] for h in w2_history])
    w2_dyn_qs = np.array([h['q_dyn'] for h in w2_history])

    mc_w1p, mc_w2p = compute_monte_carlo_sobolev(model, num_samples=100000, p=P_NORM, domain=(domain_start, domain_end))

    # --- PLOTS ---
    plt.figure(figsize=(16, 12))

    # --- Theory O(N^-1) anchored EXACTLY at first data point ---

    n0_w1 = w1_counts[0]
    e0_w1 = w1_dyn_gaps[0]
    y_theory_w1 = e0_w1 * (np.array(w1_counts) / n0_w1) ** (-1)

    n0_w2 = w2_counts[0]
    e0_w2 = w2_dyn_gaps[0]
    y_theory_w2 = e0_w2 * (np.array(w2_counts) / n0_w2) ** (-1)

    plt.plot(w1_counts, y_theory_w1 / mc_w1p, 'k--',
             linewidth=1.5, label=r'Theory $O(N^{-1})$')

    plt.plot(w2_counts, y_theory_w2 / mc_w2p, 'k:',
             linewidth=1.5)

    plt.plot(w1_counts, w1_dyn_gaps/mc_w1p, '-o', color='#D55E00', linewidth=2,
             label=r'Dörfler $W^{1,2}$ ($\theta=' + str(THETA) + r'$)')
    plt.plot(w2_counts, w2_dyn_gaps/mc_w2p, '-o', color='#009E73', linewidth=2,
             label=r'Dörfler $W^{2,2}$ ($\theta=' + str(THETA) + r'$)')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"2D Coin - Residual Convergence R_n(f)\nMethod: {JACOBIAN_METHOD.upper()}", fontsize=30)
    plt.xlabel("Number of Grid Box", fontsize=30)
    plt.ylabel("Global Residual Bound Gap", fontsize=30)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    save_path_gap = os.path.join(FIG_DIR, f"doerfler_residual_convergence_{LAYERS}_{JACOBIAN_METHOD}.pdf")
    plt.tight_layout()
    plt.savefig(save_path_gap, dpi=300)
    print(f"\nResidual Convergence plot saved to {save_path_gap}")

    plt.figure(figsize=(16, 12))

    plt.plot(w1_counts, w1_dyn_qs, '-o', color='#D55E00', linewidth=2, label=r'Dörfler $Q_n(f)$ for $W^{1,2}$')
    plt.fill_between(w1_counts, w1_dyn_qs - w1_dyn_gaps, w1_dyn_qs + w1_dyn_gaps, color='#D55E00', alpha=0.2,
                     label=r'Uncertainty $\pm R_n(f)$ ($W^{1,2}$)')

    plt.plot(w2_counts, w2_dyn_qs, '-o', color='#009E73', linewidth=2, label=r'Dörfler $Q_n(f)$ for $W^{2,2}$')
    plt.fill_between(w2_counts, w2_dyn_qs - w2_dyn_gaps, w2_dyn_qs + w2_dyn_gaps, color='#009E73', alpha=0.2,
                     label=r'Uncertainty $\pm R_n(f)$ ($W^{2,2}$)')

    plt.axhline(y=mc_w1p, color='black', linestyle='-.', alpha=0.8, label=r'MC True $W^{1,2}$')
    plt.axhline(y=mc_w2p, color='gray', linestyle='-.', alpha=0.8, label=r'MC True $W^{2,2}$')

    plt.xscale('log')
    plt.title(f"2D Coin - Integral Quadrature Convergence Q_n(f)\nMethod: {JACOBIAN_METHOD.upper()}", fontsize=30)
    plt.xlabel("Number of Grid Box", fontsize=30)
    plt.ylabel("Calculated Integral Value Q_n(f)", fontsize=30)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    save_path_integral = os.path.join(FIG_DIR, f"doerfler_integral_convergence_{LAYERS}_{JACOBIAN_METHOD}.pdf")
    plt.tight_layout()
    plt.savefig(save_path_integral, dpi=300)
    print(f"Integral Convergence plot saved to {save_path_integral}")

    plt.show()

if __name__ == "__main__":
    run_experiment()