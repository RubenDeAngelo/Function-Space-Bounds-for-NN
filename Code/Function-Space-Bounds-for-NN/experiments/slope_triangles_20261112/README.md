# Theorem reference triangles, 12 November 2026

Presentation-only revision. This package is the current renderer for all active
convergence figures. It supersedes the older plot entry points for this version
of the manuscript; the earlier implementations and frozen numerical packages
remain unchanged.

There are 13 active convergence panels with 20 theorem-reference triangles:

| Panels | Maxgap | Uniform | Source |
| --- | --- | --- | --- |
| Large deep disk, squared L2 | N^-2 | N^-1 | thm:deep_relu_separation (4.8) |
| 2D hinge, squared L2 | N^-2 | N^-1 | thm:deep_relu_separation (4.8) |
| 2D hinge, squared full H1 | N^-1 | N^-1/2 | thm:deep_relu_separation (4.8) |
| Deep/shallow disk, squared L2 | N^-2 | N^-1 | thm:deep_relu_separation (4.8) |
| Deep/shallow disk, squared full H1 | N^-1 | N^-1/2 | thm:deep_relu_separation (4.8) |
| 1D hinge, squared full H1 | 2^-N | N^-1 | thm:spline_separation (4.6) |
| 1D signed three-hinge spline, squared full H1 | 2^(-N/3) | N^-1 | thm:spline_separation (4.6) |
| Each tanh interior squared PDE-residual integral | N^-1/2 | N^-1/2 | thm: dyadic rate (4.1); eq: uniform rate (6) |

The two one-dimensional cases each have separate maxgap semilog and uniform
log-log panels. All other convergence panels use log-log axes. Shared neutral
triangles represent the common tanh benchmark. Blue and orange triangles identify
maxgap and uniform rates on the ReLU panels. The triangles are prescribed
reference slopes; no rate is estimated from data and no constant in a theorem
bound is asserted. Finite-precision captions explicitly identify the reference
as an ideal-arithmetic rate and exclude a guarantee below a rounding floor.
The smooth case has Lipschitz-width interval extensions in ideal arithmetic:
the fixed tanh networks, analytic forcing and coefficients have bounded
intermediate magnitudes and Lipschitz-width constituent extensions on the box.

There is no applicable rate theorem for the saved joint polar J/B2 refinement
sequence or the fixed-network hybrid L2 hat trajectories. Accordingly, the corner
example retains its network surface and existing certificate table. The hat
comparison uses a new exact-record table with actual leaf/evaluation counts,
terminal integral enclosures and stopping statuses. Neither the range-only
Appendix A propositions nor the 1D H1 theorem is used as a rate for hybrid L2
hat curves. Old plot assets and raw histories are retained but no longer included.
No PDE solution-error or energy-to-error estimate is introduced.

## Reproduction

From this directory, using the code repository virtual environment:

```sh
../../.venv/bin/python -B render.py
../../.venv/bin/python -B validate.py
```

Then from the paper's LaTeX directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=images disk_figure_revised.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The first LaTeX build reconstructs Figure 1 from the updated line panel and the
unchanged partition artwork. The supplement is unchanged.

`rate_guides.py` creates triangles in data coordinates. Power-law vertices
satisfy Delta log G / Delta log N = -alpha. Exponential vertices on linear-N,
log-gap axes satisfy Delta log2 G / Delta N = -1/m. Axis limits remain fixed
when drawing guides. Placement was checked visually to keep guides separate
from curves, legends and labels.

`validation/plot_metadata.json` records every panel's data, theorem mapping,
triangle vertices and hashes of all 18 source record files. `validate.py`
independently checks the prescribed rates and triangle slopes, all input hashes,
the exact four-row hat table, and all 56 previously frozen PDE-package files.
No training, integration or checkpoint is rerun or altered for this revision.
Numerical endpoints, caps, work counts and stopping outcomes remain unchanged.

The original 34-page 12 November manuscript and complete pre-edit LaTeX tree
are retained in Past Versions/12_11_2026_before_slope_triangles. Baseline hashes
are in baseline.json. Source, environment, final artifacts and validation hashes
are recorded in manifest.json. Final PDF layout/build checks are in
validation/pdf_qa.json.
