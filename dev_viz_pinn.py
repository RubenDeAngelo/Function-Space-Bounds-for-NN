"""Dev: visualization helpers for PINN dynamic vs static grid comparisons."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_comparison_heatmap(dyn_cells, stat_cells, dyn_N, stat_N, iteration, save_dir):
    """
    Plots Dynamic Grid (Left) vs Static Grid (Right) with shared colorbar.
    """
    # 1. Setup Figure (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # 2. Determine Shared Color Scale
    # We find the global min and max across both grids to ensure colors mean the same thing
    dyn_errs = [c['sq_error'] for c in dyn_cells]
    stat_errs = [c['sq_error'] for c in stat_cells]

    # Handle edge case where grid might be empty or errors are all zero
    if not dyn_errs: dyn_errs = [0]
    if not stat_errs: stat_errs = [0]

    vmin = min(min(dyn_errs), min(stat_errs))
    vmax = max(max(dyn_errs), max(stat_errs))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # 3. Helper to draw on a specific axis
    def draw_grid(ax, cells, title):
        for c in cells:
            # Rectangle xy is the bottom-left corner
            # c['center'] is the midpoint, so subtract delta
            xy = c['center'] - c['delta']
            width = 2 * c['delta']
            err = c['sq_error']

            # Draw rectangle
            # linewidth=0.0 prevents borders from dominating the image at high N
            rect = patches.Rectangle(
                xy, width, width,
                linewidth=0.1, edgecolor='lightgrey',
                facecolor=cmap(norm(err))
            )
            ax.add_patch(rect)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # 4. Plot Both Sides
    draw_grid(axes[0], dyn_cells, f"Dynamic Grid, N={dyn_N}")
    draw_grid(axes[1], stat_cells, f"Static Grid, N={stat_N}")

    # 5. Add Shared Colorbar at the Bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Orientation horizontal, placed at bottom
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label('Upper Energy Norm Bound')

    # 6. Save
    filename = f"comparison_iter_{iteration}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved comparison plot: {save_path}")

def plot_convergence(t_c, t_b, t_stat_c, t_stat_b, k_stiffness, theta, save_dir):
    """
    Plots the convergence rates (Error vs N) on a log-log scale.
    """
    plt.figure(figsize=(10, 7))

    # Theory Line (Anchored to Trained Start)
    if len(t_c) > 0:
        n0, e0 = t_c[0], t_b[0]
        x_theory = np.linspace(n0, t_c[-1], 100)
        plt.plot(x_theory, e0 * (x_theory / n0) ** (-0.5), 'k--', label=r'Optimal Rate $O(N^{-0.5})$')

    # Data Plots
    plt.plot(t_c, t_b, '-o', color='blue', linewidth=2, label='Dörfler Refinement')
    plt.plot(t_stat_c, t_stat_b, '--s', color='orange', alpha=0.8, label='Static Uniform')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Grid Cells (N)')
    plt.ylabel(r'Certified Residual Bound ($\|r\|_{L^2}$)')
    plt.title(f'Stiff Tanh Diagonal (k={k_stiffness})\nDörfler $\\theta={theta}$ vs Static')
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout

    save_path = os.path.join(save_dir, "pinn_convergence.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nConvergence plot saved to {save_path}")
    plt.show()