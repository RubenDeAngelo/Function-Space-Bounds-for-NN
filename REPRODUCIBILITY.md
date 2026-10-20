# Verified revision experiment registry

This file describes new runs on the local `revision/certified-integration`
branch. It does not resolve the provenance or validation of the archived paper
figures. See `VERIFIED_ARITHMETIC.md` for the operation-level inclusion argument
and supported scope.

## Two-bump network and Figure 1 candidate

- Domain: `[-1,1]^2`.
- Target: `exp(-((x+.4)^2+(y+.3)^2)/(2*.18^2)) + .7*exp(-((x-.4)^2+(y-.35)^2)/(2*.09^2))`.
- Network: tanh, `[2,32,32,32,1]`, affine output, float64 checkpoint parameters.
- Seed: `20261020`; CPU, one PyTorch thread, deterministic algorithms enabled.
- Data: 2048 uniform samples plus 1024 rejection-sampled local normal samples
  around each bump (standard deviation twice the respective bump width).
- Training: 3000 full-batch Adam steps, learning rate 0.001, followed by LBFGS
  with maximum 250 iterations and strong-Wolfe line search. No weight decay.
- Quantity: integral of `sum_{|alpha|<=2} |D^alpha Phi|^2`, mixed derivative once.
  The integral is the square of the manuscript's W^{2,2} norm of the *network*.
- Arithmetic: gmpy2 2.3.2, MPFR 4.2.2, explicit directed contexts, 128 bits;
  exact rational checkpoint coefficients, geometry and marking thresholds.
- Refinement: all-coordinate dyadic splitting from one box. Adaptive theta is
  exact rational 7/20; uniform refinement takes complete levels.
- Tolerance: zero. These runs stop at their leaf budgets, not tolerance success.
- Original code baseline: `887a4c6852c61d05b8174ba46bb173909821da96`.
- Implementation: local commits `34ee43f` (pilot) and `89ef0f4` (larger comparison).
  Each manifest also records full source hashes, Git status, Python/platform and
  the complete installed dependency list. Checkpoint hashes are in training.json.

Commands executed from the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -u exp_verified_two_bumps.py --output results/two-bumps-pilot --max-leaves 64
.venv/bin/python -u exp_verified_two_bumps.py --reuse results/two-bumps-pilot --output results/two-bumps-figure1 --max-leaves 4096
```

### Observed results

| Run | Policy | Leaves | Enclosure evaluations | Integration seconds | Full integral gap (approx.) | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Pilot | Adaptive | 49 | 65 | 1.82 | 707480.36 | leaf budget exhausted |
| Pilot | Uniform | 64 | 85 | 2.37 | 565517.39 | leaf budget exhausted |
| Figure 1 candidate | Adaptive | 3142 | 4189 | 118.60 | 5884.59 | leaf budget exhausted |
| Figure 1 candidate | Uniform | 4096 | 5461 | 155.21 | 9825.69 | leaf budget exhausted |

Adaptive refinement does not show an advantage at every resolution. The larger
run has a smaller gap with fewer leaves/evaluations; its final partition does
not have exactly the same cost as uniform. The figure states actual leaf counts
and plots both complete recorded histories. Timings are single-run observations
on this host and exclude training/rendering; they are not repeated timing estimates.

Safe outward decimal summaries of the final enclosures (the JSON records contain
the exact rational endpoints):

- Adaptive squared norm integral: `[38.9506, 5923.545]`; norm: `[6.24104, 76.96457]`.
- Uniform squared norm integral: `[0.54764, 9826.240]`; norm: `[0.74003, 99.12740]`.

The intervals remain wide. This figure is an implementation/behavior candidate,
not evidence of a tight norm computation at this budget. Further work should
evaluate sharper local enclosures and larger budgets, with the same saved
network, before selecting a final publication experiment. Merely increasing
arithmetic precision does not address dependency overestimation.

The ordinary 161x161 fit diagnostic has MSE `4.4697824e-6` and maximum absolute
error `0.01081051` (rounded up). These grid errors are numerical diagnostics,
not certified uniform approximation bounds or Sobolev-error bounds.

### Artifacts and validation

Each result directory contains the checkpoint, exact portable network.json,
training samples, fit grid, training settings/metrics, source/environment manifest,
both policy histories and final partitions, vector figure1.svg, figure1.png,
and fit_diagnostics.png. Exact certificate endpoints are rational strings;
floating conversions used for rendering are never reused for enclosure work.

Validation completed:

- 15 unit tests pass, including exact arithmetic oracles, cancellation, tanh
  extrema/saturation, near-zero roots, exponent-range failures, context
  independence, multi-layer derivatives, mixed-index conventions, checkpoint
  extraction, tiny-box geometry, analytic affine and tanh integrals, budgets
  and stagnation.
- Independent checks of exported rational records pass: exact tree-partition
  coverage; local volume products; global lower/upper sums; outward norm roots;
  gap bounds; cumulative tree-evaluation counters.
- Reuse checks checkpoint/training-data hashes and compares every checkpoint
  coefficient with the exact portable record.
- On 12 sampled final adaptive cells, recomputed 256-bit integrand ranges nest
  inside the 128-bit ranges, with essentially unchanged displayed widths.
- Candidate and fit-diagnostic PNGs visually inspected; no clipped labels or
  missing color scales. No existing manuscript source or figure was replaced.

Generated runs are ignored by Git and must be retained in the separate delivered
archive. Their manifests and exact JSON records are necessary for provenance.
The numerical checks support the implementation; the trusted MPFR rounding
guarantees and documented operation-level inclusion argument are the basis for
the enclosure claim. No independent backend audit or formal verification is claimed.

## Shallow-network comparison

Added one-hidden-layer tanh architecture `[2,200,1]` (801 parameters), compared
with the existing `[2,32,32,32,1]` deep model (2241 parameters). The shallow
training loads the exact saved x/y arrays from the deep pilot; both arrays were
checked for exact equality. The domain, target, seed, dtype, Adam configuration,
MPFR precision, norm convention, initial box, refinement rules and 4096-leaf
budget are unchanged. The initial shallow run has the same 3000 Adam steps and
250-iteration LBFGS budget. It underfits substantially, so a second, explicitly
separate shallow run uses a 5000-iteration LBFGS budget. Neither is claimed to
match the deep model's fit quality.

Commands executed:

```sh
.venv/bin/python -u exp_verified_two_bumps.py --output results/two-bumps-shallow --hidden-widths 200 --training-data results/two-bumps-pilot/training_data.npz --max-leaves 4096
.venv/bin/python -u exp_verified_two_bumps.py --output results/two-bumps-shallow-longer-fit --hidden-widths 200 --training-data results/two-bumps-pilot/training_data.npz --lbfgs-iterations 5000 --max-leaves 4096
.venv/bin/python compare_verified_two_bumps.py --deep results/two-bumps-figure1 --shallow results/two-bumps-shallow --output results/deep-vs-shallow-same-training
.venv/bin/python compare_verified_two_bumps.py --deep results/two-bumps-figure1 --shallow results/two-bumps-shallow-longer-fit --output results/deep-vs-shallow-longer-fit --allow-training-schedule-difference
```

Numerical source revisions: `649129c` for the same-schedule shallow run;
`f6282b9` for the longer-fit run. Both manifests' numerical-source hashes were
checked against their recorded Git revisions. Comparison records additionally
include source and input-network hashes and identify training-schedule differences.

All following gap and fit values are approximate display summaries. Full
endpoint records remain exact rational strings.

| Network / training | Grid MSE | Maximum grid fit error | Uniform leaves | Squared-H2 gap | Actual H2 norm gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deep, original | 4.4698e-6 | 0.01081 | 4096 | 9825.69 | 98.39 |
| Shallow, same schedule | 1.9866e-3 | 0.26049 | 4096 | 1050.22 | 25.68 |
| Shallow, longer LBFGS | 1.7731e-4 | 0.07877 | 4096 | 28357.34 | 165.50 |

| Shallow run / policy | Leaves | Evaluations | Integration seconds | Squared-H2 gap | Actual H2 norm gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same schedule / adaptive | 3772 | 5029 | 112.05 | 1024.77 | 24.25 |
| Same schedule / uniform | 4096 | 5461 | 125.60 | 1050.22 | 25.68 |
| Longer LBFGS / adaptive | 3322 | 4429 | 97.46 | 22762.49 | 148.47 |
| Longer LBFGS / uniform | 4096 | 5461 | 117.41 | 28357.34 | 165.50 |

All four new policy runs stop with `leaf_budget_exhausted`; no tolerance success
is claimed. The actual-norm gap is computed from stored outward roots, not by
taking the square root of the integral gap. Runtime observations are single
runs on this host; some shallow integrations/training overlap in time, so these
timings are not controlled architecture-performance measurements.

The same-schedule shallow checkpoint has narrower enclosures but a much worse
fit. Longer training improves that fit and produces much wider bounds, including
at identical uniform leaf counts. The largest absolute hidden-layer weight grows
from about 4.85 to 9.64 and the output-weight absolute sum from about 53.62 to
110.88 (ordinary numerical diagnostics). Increased coefficient magnitudes and
possible cancellation are consistent with larger interval overestimation; this
observation does not isolate its causes. No general conclusion that shallow
networks give tighter bounds follows from these runs. The networks also have
different parameter counts and represent different functions.

Validation: the existing 15 tests pass; the exact-record partition, local volume,
global sum, root, width and evaluation-count checks pass for both shallow runs;
checkpoint/training-data hashes and every portable checkpoint coefficient match;
comparison scripts check identical training arrays and enclosure settings and
require an explicit option for different training schedules. Candidate figures,
fit diagnostics and deep/shallow comparison plots were visually inspected.
The verified arithmetic/network/integration modules were not changed in this pass.

## Current Figure 1: L2 only

The current candidate reuses the original deep checkpoint and integrates only
`|Phi|^2`. There is no retraining. Exact coefficient-record equality with the
original deep run was checked. The network's value-only path now omits unused
derivative-range calculations; it still applies the same outward-rounded tanh
endpoint formula. The existing 15 tests, including the nonlinear analytic L2
integral case, pass after this change.

Executed numerical run at source revision `c1bc41f`:

```sh
.venv/bin/python -u exp_verified_two_bumps.py --reuse results/two-bumps-figure1 --output results/two-bumps-deep-l2 --norm L2 --reference-slope --max-leaves 4096
```

Both policies use 128-bit MPFR arithmetic, theta=7/20 for adaptive marking,
all-coordinate halving, the same root box and zero stopping tolerance.

| Policy | Leaves | Evaluations | Integration seconds | Integral gap (approx.) | Actual L2 norm gap (approx.) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adaptive | 3850 | 5133 | 20.29 | 0.8771074 | 0.8053895 |
| Uniform | 4096 | 5461 | 22.72 | 1.1719976 | 1.0581944 |

Both stopped with `leaf_budget_exhausted`. Conservative outward decimal
summaries of the final bounds are:

- Adaptive integral: `[0.0201154, 0.8972229]`; actual L2 norm: `[0.14182, 0.94722]`.
- Uniform integral: `[0.00060886, 1.17260644]`; actual L2 norm: `[0.02467, 1.08288]`.

Panel (a) shows the integral gap `U-L` for the square of the L2 norm, rather
than the actual norm gap. The slope triangle has a fourfold horizontal increase
in N and a twofold vertical decrease in gap, giving slope -1/2 on logarithmic
axes. It is a **theoretical reference**, not a fitted rate or a prediction that
every finite segment follows that slope. Its position does not specify a
theoretical constant. The reference comes from gamma=1 width decay in d=2 in
the exact-arithmetic theorem; fixed-precision computations can reach a rounding
floor. For the rooted norm gap, the same rate requires a positive integral
lower bound; the general near-zero root estimate alone gives exponent -1/4.

The triangle's factor labels were clarified after the numerical run. The
saved `figure_render_source.py` and `figure_metadata.json` record that graphical
revision independently of the immutable numerical-source hashes. The candidate
PNG was visually inspected and exported-record checks pass for partition
coverage, local/global endpoints, roots, widths and evaluation counts. All
numerical source hashes match the recorded numerical Git revision.

These bounds are much smaller in absolute value than for H2 but remain broad
relative to the integral's scale. Restricting the quantity to L2 has not by
itself established a practically tight first-figure result. The figure remains
a candidate illustrating localization and gap reduction at the stated budget.
