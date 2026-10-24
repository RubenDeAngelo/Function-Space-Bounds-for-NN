# Certified adaptive integration for neural-network function-space norms

Local revision code for Johannes Gründler, Moritz Maibaum and Philipp Petersen.
The manuscript is in the parent paper directory, `../../LaTeX/main.tex` relative
to this checkout. This checkout remains an independent Git repository.

## Install the verified workflow

Use Python 3.13 and a CPU environment matching `requirements-verified-lock.txt`:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-verified-lock.txt
.venv/bin/python -m unittest discover -s tests -v
```

The lockfile records the tested macOS environment, not a guarantee of identical
training on all platforms. MPFR/GMP versions, Python/platform, source hashes and
installed packages are recorded for each suite. Stored binary checkpoint values
are exact real coefficients of the network being enclosed. Ordinary training and
plotting arithmetic does not enter certificates.

## Current Figure 1: preserve the agreed deep checkpoint

```sh
.venv/bin/python exp_verified_two_bumps.py --reuse results/two-bumps-deep-l2 --output results/figure1-replay --norm L2 --theta 0.35 --reference-slope --max-leaves 4096
.venv/bin/python validate_records.py results/figure1-replay
.venv/bin/python render_verified.py results/figure1-replay
```

The saved historical run is `results/two-bumps-deep-l2/`. Its target is unscaled,
its architecture is `[2,32,32,32,1]`, and theta is 7/20. The left panel shows the
full gap for the squared-L2 integral, not a rooted norm gap. Rooted endpoints
are exported separately. The N^(-1/2) triangle is a theoretical reference.
Output directories must be new. `--tolerance` sets an absolute full integral-gap
tolerance; zero disables a positive tolerance target. Budget failure is not
accuracy success. Preserve the saved checkpoint when reproducing the paper;
training again is a different operation.

## Reproducible replacement studies

```sh
.venv/bin/python exp_verified_studies.py run --config experiments/paper.json --output results/validated-paper-suite-v1
.venv/bin/python exp_verified_studies.py replay --input results/validated-paper-suite-v1 --output results/paper-suite-replay
.venv/bin/python validate_records.py results/validated-paper-suite-v1
.venv/bin/python render_verified.py results/validated-paper-suite-v1
```

The first command trains and certifies; replay loads saved networks and certifies
without training; render reads records only. A suite records config.json,
experiments.json, models with checkpoints and exact coefficients, training data,
per-policy histories/partitions, validation.json, summary.json, environment.json,
a complete source snapshot, and figures. The PINN samples new points per epoch;
its independent CPU generator seed, first batch and final generator state are
saved. All sampled diagnostics are approximate, not uniform error certificates.

`experiments/paper.json` specifies a compact campaign: three paired 1D
untrained/trained tanh seeds, one disk checkpoint for both H1/H2 experiments,
and one elliptic interior-residual checkpoint. Individual seed traces are shown;
there are no confidence intervals or population/architecture claims. The 1D
Gaussian is on [0,1] and is explicitly a new experiment, not a reconstruction of
the old [0,2*pi] study. Disk epsilon is 0.5, with 5000 uniform saved points.
All certified studies use p=2 and count each mixed derivative once.

Configuration fields choose architecture, budgets, training and arithmetic
settings. Targets/domains are the documented fixed study definitions; unsupported
changes are rejected. This runner does not yet implement arbitrary PDEs,
noninteger powers, or validated ReLU activation-region integration.

## Numerical implementation

- `verified_arithmetic.py`: explicitly directed MPFR scalar arithmetic, powers,
  roots and stable tanh derivative ranges.
- `verified_network.py`: interval forward AD of supported affine/tanh networks,
  derivative order 0/1/2 and positive integer p.
- `verified_integration.py`: exact rational dyadic geometry, ancestor range
  intersections, volume-weighted integral bounds and uniform/Dörfler refinement.
- `verified_residual.py`: validated forcing and squared interior residual for
  Delta u = 2*tanh''(x+y-1).
- `experiment_records.py`: source/environment snapshots and exact record checks.

See `VERIFIED_ARITHMETIC.md` for the inclusion argument and limits. Record
validation and numerical tests support correct use; no independent formal
verification or new backend guarantee is claimed.

## Legacy material

The old `exp_sobolev_*`, `exp_lp_*`, `exp_pinn_2d_energy_norm.py`,
`exp_combined_lp_sobolev_convergence.py` and `lib_*` bound routines remain for
historical inspection. They are not the validated paper workflow. Some require
Numba or auto_LiRPA, which are intentionally not dependencies of the verified
workflow. They do not establish end-to-end finite-precision certificates.
Legacy statistics plot empirical percentile ranges, not confidence intervals;
legacy ReLU indicators are not full integral gaps. The former Figure 1 mapping
and old algorithm numbering must not be used for the revised paper.

Confirmed legacy disk defects have been corrected for future exploratory use:
N requested uniform points are returned, heatmap/convergence defaults agree on
epsilon=0.5 and weight decay=0.001, static x-coordinates use actual k*k cells,
and reference labels no longer assert an N^-1 rate in 2D or universal rho=0.25.
These changes do not establish the provenance of old graphics or certify them.

## Artifact delivery

`results/` is Git-ignored. A source clone alone does not contain the saved paper
networks and numerical results. Supply the result archive and checksum manifest
alongside the source. `REPRODUCIBILITY.md` maps the current experiments; older
entries record the historical development and are not current certification
claims for archived figures.
