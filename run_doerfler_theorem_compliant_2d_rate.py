"""
run_doerfler_theorem_compliant_2d.py

Comparison of W1p and W2p norms using Dynamic Refinement with DÖRFLER MARKING.
ADAPTED FOR 2D "SMOOTH COIN" FUNCTION.

Updates:
- Tracks Local Error Reduction during splits (Parent vs Sum of Children).
- Plots Geometric Decay against theoretical q = 1 - theta*(1 - rho).
- Plots Local Error Reduction scatter against theoretical y = rho * x AND a y = x reference line.
- All plots saved as PDFs.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

# Model & Training
LAYERS = [2, 32, 32, 32, 32, 1]
ACTIVATION = 'sigmoid'
EPOCHS = 6000
N_SAMPLES = 5000
MODEL_PATH = f"model_smooth_coin_2d_{LAYERS}.pth"

# --- CONFIGURATION ---
FIG_DIR = f"fig_doerfler_geometric_decay_{LAYERS}"
os.makedirs(FIG_DIR, exist_ok=True)

# Data Generation Settings
EPSILON_TRANSITION = 0.5
TRANSITION_SAMPLE_RATIO = 0.1

# Refinement Settings
P_NORM = 2.0
MAX_ITER = 30  # Increased to 40 to match your reference plot
THETA = 0.2
RHO = 0.25     # Theoretical local error reduction factor (1/4)


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

def generate_balanced_2d_coin_data(N=3000, epsilon=0.1, transition_ratio=0.5):
    N_transition = int(N * transition_ratio)
    N_uniform = N - N_transition
    low, high = -1.5, 1.5
    X_uni = np.random.uniform(low, high, (N_uniform, 2)).astype(np.float32)
    if N_transition > 0:
        theta = np.random.uniform(0, 2 * np.pi, N_transition)
        padding = 0.05
        r_min = 1.0 - padding
        r_max = 1.0 + epsilon + padding
        r_focused = np.random.uniform(r_min, r_max, N_transition)
        x_focus = r_focused * np.cos(theta)
        y_focus = r_focused * np.sin(theta)
        X_focus = np.stack([x_focus, y_focus], axis=1).astype(np.float32)
        X = np.vstack([X_uni, X_focus])
    else:
        X = X_uni
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = smooth_disk(X[:, 0], X[:, 1], epsilon=epsilon)
    return X, y.reshape(-1, 1).astype(np.float32)


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
def run_doerfler_grid_for_norm(model, initial_centers, initial_delta, max_iter, theta=0.5, p=2.0, norm_type='w1p'):
    print(f"\n--- Running Dörfler Refinement (theta={theta}, p={p}) for {norm_type.upper()} ---")

    active_cells = []
    bound_fn = compute_w1p_bounds if norm_type == 'w1p' else compute_w2p_bounds

    if np.isscalar(initial_delta):
        initial_batch = [(c, initial_delta) for c in initial_centers]
    else:
        initial_batch = zip(initial_centers, initial_delta)

    for x, d in initial_batch:
        lb, ub = bound_fn(model, x, d, p=p)
        error_mass = 0.5 * ((ub ** p) - (lb ** p))
        active_cells.append({
            'center': x, 'delta': d,
            'low': lb, 'high': ub,
            'error': error_mass
        })

    history = []
    split_data = []  # To store (parent_error, sum_children_error)

    for i in range(max_iter + 1):
        count = len(active_cells)
        global_gap = sum(c['error'] for c in active_cells)

        history.append({'iter': i, 'count': count, 'gap_dyn': global_gap})
        print(f"Iter {i:<3} | Cells: {count:<5} | {norm_type.upper()} Gap: {global_gap:.6f}")

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
                cl, cu = bound_fn(model, cx, child_delta, p=p)
                child_error_mass = 0.5 * ((cu ** p) - (cl ** p))
                children_for_this_parent.append({
                    'center': cx, 'delta': child_delta,
                    'low': cl, 'high': cu,
                    'error': child_error_mass
                })

            new_cells.extend(children_for_this_parent)

            # Extract Local Error Reduction Data
            sum_child_error = sum(c['error'] for c in children_for_this_parent)
            split_data.append((parent['error'], sum_child_error))

        active_cells = unmarked_set + new_cells

    return history, np.array(split_data)


# --- PLOTTING FUNCTIONS ---
def plot_geometric_decay(history, theta, rho, norm_type):
    iterations = np.array([h['iter'] for h in history])
    empirical_gaps = np.array([h['gap_dyn'] for h in history])

    # Calculate Theoretical Line: R_k = R_0 * (1 - theta * (1 - rho))^k
    q = 1.0 - theta * (1.0 - rho)
    R_0 = empirical_gaps[0]
    theory_gaps = R_0 * (q ** iterations)

    plt.figure(figsize=(10, 8))
    plt.plot(iterations, empirical_gaps, '-o', markersize=8, color='#4A7BC7', linewidth=2.5, label='Empirical Error')

    theory_label = r'Theory $(1 - ' + str(int((1-rho)*4)) + r'\theta/4)^k$' if rho == 0.25 else f'Theory $q={q:.3f}$'
    plt.plot(iterations, theory_gaps, '--k', linewidth=2.5, label=theory_label)

    plt.yscale('log')
    plt.title(f"Geometric Convergence Decay ({norm_type.upper()})", fontsize=24)
    plt.xlabel("Iteration Step $k$", fontsize=20)
    plt.ylabel("Global Error Bound $R_k$", fontsize=20)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"geometric_decay_{norm_type}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Geometric Decay Plot to {save_path}")
    plt.close()


def plot_local_error_reduction(split_data, rho, norm_type):
    parent_errors = split_data[:, 0]
    children_errors = split_data[:, 1]

    plt.figure(figsize=(10, 8))

    # Plot Scatter
    plt.scatter(parent_errors, children_errors, color='#85C57A', alpha=0.5, s=30, label='Splits')

    # Plot Theoretical Line: y = rho * x
    min_err = min(np.min(parent_errors), np.min(children_errors))
    max_err = max(np.max(parent_errors), np.max(children_errors))

    # Extend line slightly past min/max for aesthetics
    x_theory = np.logspace(np.log10(min_err/10), np.log10(max_err*10), 100)
    y_theory = rho * x_theory

    theory_label = r'Theory $y = 1/4 x$' if rho == 0.25 else f'Theory $y = {rho} x$'
    plt.plot(x_theory, y_theory, '--k', linewidth=2.5, label=theory_label)

    # ADDED: Red line for y = x
    plt.plot(x_theory, x_theory, '-r', linewidth=2.0, label=r'$y = x$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Local Error Reduction ($R_K \\to \\sum R_{{K'}}$) ({norm_type.upper()})", fontsize=22)
    plt.xlabel(r"Parent Error $R(K)$", fontsize=20)
    plt.ylabel(r"Sum Children Error $\sum R_{K'}$", fontsize=20)

    # Set limits to tightly bound the scattered data + theory line
    plt.xlim(min_err/5, max_err*5)
    plt.ylim(min_err/5, max_err*5)

    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(FIG_DIR, f"local_error_reduction_{norm_type}_{LAYERS}.pdf")
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Saved Local Error Reduction Plot to {save_path}")
    plt.close()


# --- MAIN EXPERIMENT ---
def run_experiment():
    print(f"--- Starting Geometric Decay & Error Reduction Experiment ---")

    # Load or Train Model
    X_np, y_np = generate_balanced_2d_coin_data(N=N_SAMPLES, epsilon=EPSILON_TRANSITION,
                                                transition_ratio=TRANSITION_SAMPLE_RATIO)
    model = FlexibleNet(LAYERS, activation=ACTIVATION)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"No trained model found. Training on {len(X_np)} samples...")
        train_model(model, torch.from_numpy(X_np), torch.from_numpy(y_np), epochs=EPOCHS)
        torch.save(model.state_dict(), MODEL_PATH)

    model.eval()

    # Initial Grid Setup
    domain_start, domain_end = -1.5, 1.5
    N_side = 2
    total_width = domain_end - domain_start
    delta_init = (total_width / N_side) / 2.0
    c1d = np.linspace(domain_start + delta_init, domain_end - delta_init, N_side).astype(np.float32)
    gx, gy = np.meshgrid(c1d, c1d)
    centers_init = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # Run Refinements
    w1_hist, w1_splits = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w1p')
    w2_hist, w2_splits = run_doerfler_grid_for_norm(model, centers_init, delta_init, MAX_ITER, THETA, P_NORM, 'w2p')

    # Generate Plots
    print("\n--- Generating W1p Plots ---")
    plot_geometric_decay(w1_hist, THETA, RHO, 'w1p')
    plot_local_error_reduction(w1_splits, RHO, 'w1p')

    print("\n--- Generating W2p Plots ---")
    plot_geometric_decay(w2_hist, THETA, RHO, 'w2p')
    plot_local_error_reduction(w2_splits, RHO, 'w2p')

if __name__ == "__main__":
    run_experiment()