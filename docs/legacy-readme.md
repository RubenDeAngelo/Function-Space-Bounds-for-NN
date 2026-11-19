# Historical README

This document describes an earlier paper version. Use the root README for the current reproducibility snapshot.

# Certified and Accurate Computation of Function Space Norms of Deep Neural Networks

Code for the paper:

> **Certified and accurate computation of function space norms of deep neural networks**  
> Johannes Gründler, Moritz Maibaum, Philipp Petersen  
> arXiv:2603.06431 (2026)

## Overview

This repository implements the **AdaQuad** framework for computing certified lower and upper bounds on Lebesgue and Sobolev norms of feedforward neural networks. The approach combines:

- **Interval arithmetic enclosures** for neural network function values, Jacobians, and Hessians on axis-aligned boxes (Algorithms 6, 7, 8);
- **Dörfler adaptive marking** (Algorithm 4) to concentrate refinement where the local error is largest;
- **Hölder refinement** (Algorithm 5) and **uniform bisection** as complementary refinement strategies;
- **Gauss–Legendre quadrature** with an exact-integration shortcut for ReLU networks on affine-linear boxes (Proposition 4.15 / Remark 4.16).

The output of each run is a certified interval `[Q_n - η_n, Q_n + η_n]` containing the true norm, with `η_n → 0` at a guaranteed geometric rate (Theorem 4.1, Proposition 3.7). This yields certified computation of L^p, W^{1,p}, and W^{2,p} norms, as well as PINN interior residuals.

For the full mathematical treatment — convergence proofs, interval enclosure constructions, and a detailed discussion of related work — please refer to the paper.

---

## Repository Structure

### Core Library (`lib_*.py`)

| File | Contents |
|---|---|
| `lib_nn_model.py` | `FlexibleNet`: configurable feedforward network (ReLU / tanh / sigmoid). Parameter extraction utilities. |
| `lib_nn_interval_ops.py` | Layer-wise interval arithmetic forward and backward passes (Algorithms 6–8 substrate). |
| `lib_activation_enclosures.py` | Interval enclosures Σ, Σ′, Σ″ for σ, σ′, σ″ (tanh, ReLU, sigmoid) required by the bounding algorithms. |
| `lib_jacobian_enclosures.py` | Certified Jacobian bounds via IBP (Algorithm 7) and RecurJac. |
| `lib_hessian_enclosures.py` | Certified Hessian bounds via vectorized IBP with optional Numba JIT (Algorithm 8). |
| `lib_lp_enclosures.py` | Local L^p enclosures for function values, gradient, and Hessian components. |
| `lib_w1p_enclosures.py` | W^{1,p} norm enclosure combining φ-L^p and gradient-L^p bounds (Corollary 4.22). |
| `lib_w2p_enclosures.py` | W^{2,p} norm enclosure combining φ-L^p, gradient-L^p, and Hessian-L^p bounds (Corollary 4.27). |
| `lib_adaquad_lp.py` | Full AdaQuad pipeline for the L^p case (Algorithm 3): `ReLUNetwork`, activation-region check (Proposition 4.15), `AdaptiveLpNorm` with Dörfler marking and Gauss–Legendre quadrature. |
| `lib_eta_bounds.py` | Utilities for assembling global error bound η_n from local contributions η_K. |
| `lib_training.py` | Adam training loop wrapper shared across experiments. |
| `lib_utils.py` | Miscellaneous helpers (domain grids, smooth-disk target, etc.). |

### Experiments (`exp_*.py`) and Figure Mapping

Each experiment script is self-contained and saves its figures to a dedicated `fig_*/` directory that it creates automatically.  Networks are trained on first run and cached as `.pth` files; subsequent runs load the cached weights.

| Script | Paper figures |
|---|---|
| `exp_combined_lp_sobolev_convergence.py` | **Figure 1 (left)** — combined geometric decay of the certified bound gap for L^p, W^{1,p}, W^{2,p} on the 2D smooth-disk target. Also produces the **Figure 1 (right)** adaptive-partition heatmap for the L^p run. |
| `exp_sobolev_1d_statistics_100runs.py` | **Figures 3 & 4** — mean normalized bound gap and 95 % CI over 100 random seeds for W^{1,2} and W^{2,2}, for both untrained (Fig. 3) and trained (Fig. 4) tanh networks. Run once with `LAYERS = [1, 32, 32, 32, 1]` (deep) and once with `LAYERS = [1, 200, 1]` (wide). Target: 1D Gaussian peak (Eq. 47). |
| `exp_lp_1d_statistics_100runs.py` | **Figure 5** — analogous 100-run statistics for the L^p bound gap, comparing deep and wide ReLU architectures on the same 1D Gaussian-peak target. |
| `exp_lp_2d_disk_activation_regions.py` | **Figure 6** — adaptive L^2 refinement heatmaps on the 2D smooth-disk domain (deep and wide ReLU architectures, θ = 0.5), demonstrating the activation-region exact-integration shortcut. |
| `exp_sobolev_2d_disk_convergence.py` | **Figure 7** — geometric convergence of W^{1,p} and W^{2,p} bound gaps on the 2D smooth-disk domain (single run, deep and wide tanh architectures), plus local bound-gap reduction scatter plots verifying Remark 4.2. |
| `exp_sobolev_2d_disk_heatmaps.py` | **Figure 8** — adaptive-grid heatmaps of the local W^{1,p} and W^{2,p} bound gaps for both architectures at iteration 25 of the 2D smooth-disk experiment. Also provides `compute_monte_carlo_sobolev` used by `exp_combined_lp_sobolev_convergence.py`. |
| `exp_pinn_2d_energy_norm.py` | **Figure 9** — certified bound gap convergence and local reduction for the PINN energy-norm residual of the tanh-wavefront elliptic PDE (Eqs. 49–50), including the adaptive-grid heatmap. |

---

## Running the Experiments

Each script is run directly with Python. For example:

```bash
python exp_combined_lp_sobolev_convergence.py
python exp_sobolev_1d_statistics_100runs.py   # edit LAYERS inside for deep vs. wide
python exp_lp_1d_statistics_100runs.py
python exp_lp_2d_disk_activation_regions.py
python exp_sobolev_2d_disk_convergence.py
python exp_sobolev_2d_disk_heatmaps.py
python exp_pinn_2d_energy_norm.py
```

Output PDFs are written to the corresponding `fig_*/` directory.  Key hyperparameters (architecture, θ, number of iterations, weight decay) are set as module-level constants at the top of each script.

---

## Citation

If you use this code, please cite:

```bibtex
@article{grundler2026certified,
  title={Certified and accurate computation of function space norms of deep neural networks},
  author={Gr{\"u}ndler, Johannes and Maibaum, Moritz and Petersen, Philipp},
  journal={arXiv preprint arXiv:2603.06431},
  year={2026}
}
```
