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
