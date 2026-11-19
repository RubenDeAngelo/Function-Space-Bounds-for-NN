# Certified adaptive integration of ReLU network norms

Submission reproducibility snapshot for Johannes Gründler, Moritz Maibaum and
Philipp Petersen. Prepared 19 November 2026 from the manuscript dated 12 November
2026. This is supporting material for a planned ACOM submission, not an accepted
article or a claim that peer review has been completed.

**Version:** use the immutable commit linked in the manuscript's availability
statement. This branch is `acom-submission-2026-11-19`. Download its complete ZIP
or clone it; unlike the historical repository, this snapshot includes the saved
paper models and exact numerical records. It requires no separate local project.

## Start here

```sh
git clone --branch acom-submission-2026-11-19 https://github.com/RubenDeAngelo/Function-Space-Bounds-for-NN.git
cd Function-Space-Bounds-for-NN
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-replay.txt
.venv/bin/python release/verify.py
.venv/bin/python release/reproduce.py --checks
```

`verify.py` verifies the released file hashes and checks that the software has no
Mothercode imports or dependency. `reproduce.py` first creates a separate working
copy under `.release-work/`, then runs checks/replays there; the published records
are never overwritten. `--checks` rebuilds the C++
kernels with your installed GMP/MPFR and includes 40 1D hybrid replays,
the 16 original trained certificates, 80 randomized kernel comparisons, four
1024-leaf PDE replays, independent checks and exact terminal-record audits.
The trained and large-budget full histories are supplied and audited; the default
checks do not rerun every largest-budget trajectory.

Python 3.13 was used for validation. Replay needs no training and no PyTorch.
Retraining is a distinct operation and may change the model: the stored exact
coefficients define the network whose integral is enclosed. The old complete
training environment lock is preserved inside `Code/Function-Space-Bounds-for-NN/`.

## Native libraries and building figures

The relative-width kernel needs a C++17 compiler and GMP/GMPXX. The smooth-PDE
kernel and corner verifier additionally need MPFR. Install these libraries with
your system package manager. `CXX`, `GMP_PREFIX` and `MPFR_PREFIX` select a compiler
and nonstandard installation prefixes. There is no required `/Users/...` path.

```sh
.venv/bin/python release/reproduce.py --build-kernels --checks
.venv/bin/python release/reproduce.py --render
.venv/bin/python release/reproduce.py --build-paper --paper-source /path/to/private/LaTeX
```

Rendering regenerates tables and current plots from saved exact records. The optional
paper build uses your separately supplied manuscript source and also needs TeX
Live, `latexmk`, BibTeX and its manuscript packages. Both PDFs are built locally
in the working copy; the draft manuscript is not distributed in this code release. See `release/VALIDATION.md`
for the exact scope and outcomes of validation performed for this release.

## File and experiment map

The executable paper code is in `Code/Function-Space-Bounds-for-NN/` (abbreviated
`C` below); `LaTeX/` holds figure-generation assets and numerical tables. The manuscript
text and manuscript PDFs remain in the separate paper project.

| Paper material | Frozen input | Reproduction |
|---|---|---|
| Figure 1; matched large-disk table | `C/experiments/smoothed_disk_figure1/` | Its `validate.py`, `run_experiment.py`, `test_fast.py`; current line panel via slope renderer |
| Figure 2; 1D controls; Appendix A hat table | `C/results/paper-originality-v1/` | `C/experiments/hybrid_submission/replay.py --suite controls`; independent `validate.py` and `test_study.py` in the frozen package |
| Figure 3; 2D/3D and trained controls | `C/results/paper-geometry-v1/` | Hybrid geometry replay; independent polygon validator and tests in the frozen package |
| Figure 4; relative-width/work/endpoint tables | `C/experiments/relative_width_20261111/` | `kernel.cpp`, `prepare_validate.py`, `test_kernel.py`, `run_suite.py`, `summarize.py` |
| Figure 5; corner energy/boundary table | `C/experiments/pde_revision_20261112/corner_source/` and `runs/corner-energy-boundary.csv` | Preserved source/checkpoint, verifier hashes, provenance, replay history and validation comparison |
| Figure 6; two tanh PDE residuals | `C/experiments/pde_revision_20261112/` | `validate_residual.py`, `replay.py` (1024 leaves), `replay.py --full` (65536 leaves) |
| All active convergence reference triangles | `C/experiments/slope_triangles_20261112/` | Current `render.py` and independent `validate.py` |
| Training provenance and historical model inputs | `C/results/paper-maxgap-v2/`, `C/results/validated-paper-suite-v1/` | Saved coefficients, checkpoints, training arrays, environments and source snapshots |
| Supplement tables | `LaTeX/tables/` | `--render`; compile the separately supplied supplement source with `--build-paper --paper-source ...` |

The experiment READMEs and original manifests are preserved as
historical provenance (apart from the documented source-only validator adaptation). Drafting notes and historical manuscript bodies are omitted as recorded in
`release/EXCLUDED_MANUSCRIPT_MATERIAL.json`. Some older READMEs describe earlier paper stages, older figures, local
installation paths or the formerly deferred release; this README and the release
wrapper supersede those entry-point instructions for the current snapshot.

## Scope of the evidence

The comparisons use a specified local integral rule and dyadic refinement policy.
The principal ReLU results distinguish exact rational integration from MPFR-based
finite-precision fallback. Integral width, rooted norm width, full-grid leaf count
and actual cell evaluations are different quantities. All sixteen extended
trained-policy runs reach relative full integral width at most one percent; earlier
fixed-budget failures remain included. All four tanh residual runs exhaust their
65536-leaf caps. The corner computation uses its saved joint polar partition
sequence; it is not a J- or boundary-residual-maxgap experiment. No PDE
solution-error bound or formal software verification is claimed.

Software is MIT licensed; the manuscript retains the authors' copyright. See
`LICENSE`, `RIGHTS.md`, `release/PROVENANCE.json` and `release/SHA256.json`.
