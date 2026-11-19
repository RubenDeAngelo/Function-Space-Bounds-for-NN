"""Fig. 7: Geometric decay of W^{1,p} and W^{2,p} bounds on the smooth-coin 2D domain with local error reduction scatter."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools

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

from lib_nn_model import FlexibleNet
from lib_training import train_model
from lib_w1p_enclosures import compute_w1p_bounds
from lib_w2p_enclosures import compute_w2p_bounds

# Data Generation Settings
EPSILON_TRANSITION = 0.5

# Model & Training
LAYERS = [2, 32, 32, 32, 1]
weight_decay = 0.001
ACTIVATION = 'tanh'
EPOCHS = 10000
N_SAMPLES = 5000
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}_{EPSILON_TRANSITION}_wd{weight_decay}_{ACTIVATION}.pth"

FIG_DIR = f"fig_doerfler_geometric_decay_{LAYERS}_{EPSILON_TRANSITION}_wd{weight_decay}_{ACTIVATION}"
os.makedirs(FIG_DIR, exist_ok=True)

# Refinement Settings
P_NORM = 2.0
MAX_ITER = 30
THETA = 0.35
RHO = 0.25     # Theoretical local error reduction factor (1/4) - used only for scatter plot reference
JACOBIAN_METHOD = 'ibp'  # Options: 'recurjac' or 'ibp'

# Set to True to track and render slow local error reduction scatter plots.
# Set to False to save memory and runtime.
PLOT_LOCAL_REDUCTION = True

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

def generate_balanced_2d_coin_data(N=3000, epsilon=0.1, transition_ratio=0.5):
    # Legacy driver: use exactly N uniform samples; no transition oversampling.
    N_uniform = N
    low, high = -1.5, 1.5
    X_uni = np.random.uniform(low, high, (N_uniform, 2)).astype(np.float32)
    X = X_uni
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return X, y.reshape(-1, 1).astype(np.float32)

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

def run_doerfler_grid_for_norm(model, initial_centers, initial_delta, max_iter, theta=0.5, p=2.0, norm_type='w1p', jac_method='recurjac', track_splits=False):
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
        active_cells.append({
            'center': x, 'delta': d,
            'low': lb, 'high': ub,
            'error': error_mass
        })

    history = []
    split_data = []

    for i in range(max_iter + 1):
        count = len(active_cells)
        global_gap = sum(c['error'] for c in active_cells)

        history.append({'iter': i, 'count': count, 'gap_dyn': global_gap})
        print(f"Iter {i:<3} | Cells: {count:<5} | {norm_type.upper()} Error: {global_gap:.6f}")

        if i == max_iter: break

        total_error = global_gap
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
            children_for_this_parent = []

            for cx in child_centers:
                cl, cu = bound_fn(model, cx, child_delta, p=p, jacobian_method=jac_method)
                child_error_mass = 0.5 * ((cu ** p) - (cl ** p))
                children_for_this_parent.append({
                    'center': cx, 'delta': child_delta,
                    'low': cl, 'high': cu,
                    'error': child_error_mass
                })

            new_cells.extend(children_for_this_parent)

            if track_splits:
                sum_child_error = sum(c['error'] for c in children_for_this_parent)
                split_data.append((parent['error'], sum_child_error))

        active_cells = unmarked_set + new_cells

    return history, np.array(split_data) if track_splits else np.array([])

def plot_geometric_decay_combined(w1_hist, w2_hist, theta, mc_w1, mc_w2):
    """
    Plots the geometric decay of BOTH W1p and W2p norms in a single chart.
    (Theory lines removed)
    """
    # Extract data for W1p
    iter_w1 = np.array([h['iter'] for h in w1_hist])
    err_w1 = np.array([h['gap_dyn'] for h in w1_hist]) / mc_w1

    # Extract data for W2p
    iter_w2 = np.array([h['iter'] for h in w2_hist])
    err_w2 = np.array([h['gap_dyn'] for h in w2_hist]) / mc_w2

    plt.figure(figsize=(12, 9))

    # Plot W1p
    plt.plot(iter_w1, err_w1, '-o', markersize=8, color='#D55E00', linewidth=2.5, label=r'Empirical $W^{1,2}$')

    # Plot W2p
    plt.plot(iter_w2, err_w2, '-s', markersize=8, color='#009E73', linewidth=2.5, label=r'Empirical $W^{2,2}$')

    plt.yscale('log')
    plt.title(rf"Geometric Convergence Decay | $\theta$={theta}", fontsize=26)
    plt.xlabel("Iteration Step $k$", fontsize=22)
    plt.ylabel("Normalized Global Error", fontsize=22)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"geometric_decay_combined_{JACOBIAN_METHOD}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Combined Geometric Decay Plot to {save_path}")
    plt.close()

def plot_local_error_reduction(split_data, rho, norm_type):
    if len(split_data) == 0: return # Safety check

    parent_errors = split_data[:, 0]
    children_errors = split_data[:, 1]

    plt.figure(figsize=(10, 8))

    plt.scatter(parent_errors, children_errors, color='#85C57A', alpha=0.5, s=30, label='Splits')

    min_err = min(np.min(parent_errors), np.min(children_errors))
    max_err = max(np.max(parent_errors), np.max(children_errors))

    x_theory = np.logspace(np.log10(min_err/10), np.log10(max_err*10), 100)
    y_theory = rho * x_theory

    theory_label = f'Reference $y = {rho:.2f} x$'
    plt.plot(x_theory, y_theory, '--k', linewidth=2.5, label=theory_label)

    plt.plot(x_theory, x_theory, '-r', linewidth=2.0, label=r'$y = x$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Local Error Reduction", fontsize=22)
    plt.xlabel(r"Parent Error", fontsize=22)
    plt.ylabel(r"Sum Children Error", fontsize=22)

    plt.xlim(min_err/5, max_err*5)
    plt.ylim(min_err/5, max_err*5)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"local_error_reduction_{norm_type}_{LAYERS}_{JACOBIAN_METHOD}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Local Error Reduction Plot to {save_path}")
    plt.close()

def run_experiment():
    print(f"--- Starting Geometric Decay & Error Reduction Experiment ---")

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

    # Pre-calculate Monte Carlo truth for normalization
    mc_w1p, mc_w2p = compute_monte_carlo_sobolev(model, num_samples=100000, p=P_NORM, domain=(-1.5, 1.5))

    domain_start, domain_end = -1.5, 1.5
    N_side = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # Pass PLOT_LOCAL_REDUCTION toggle down to refinement function
    w1_hist, w1_splits = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w1p', JACOBIAN_METHOD, track_splits=PLOT_LOCAL_REDUCTION)
    w2_hist, w2_splits = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w2p', JACOBIAN_METHOD, track_splits=PLOT_LOCAL_REDUCTION)

    # Generate Combined Geometric Plot (RHO removed from arguments)
    print("\n--- Generating Combined Geometric Decay Plot ---")
    plot_geometric_decay_combined(w1_hist, w2_hist, THETA, mc_w1p, mc_w2p)

    # Optional Local Reduction Plots
    if PLOT_LOCAL_REDUCTION:
        print("\n--- Generating Local Reduction Plots ---")
        if len(w1_splits) > 0:
            plot_local_error_reduction(w1_splits, RHO, 'w1p')
        if len(w2_splits) > 0:
            plot_local_error_reduction(w2_splits, RHO, 'w2p')

if __name__ == "__main__":
    run_experiment()