# Verified arithmetic: implementation boundary and audit

Terminology: use **integral enclosure gap** for `U-L`, **norm enclosure gap**
for the difference of the rooted endpoints, and **half-width** for half an
enclosure gap. These are deterministic interval quantities and carry no
statistical interpretation.

## Audit of the archived path

Inspected baseline: `887a4c6852c61d05b8174ba46bb173909821da96`.

| Component | Finding | Verified replacement |
| --- | --- | --- |
| `lib_nn_interval_ops.py`, `lib_jacobian_enclosures.py` | NumPy point evaluations, affine operations and matrix products round normally; no accumulated rounding enclosure | Every scalar add and multiply uses an explicitly directed MPFR context |
| `lib_activation_enclosures.py` | Ordinary elementary functions; clipping at +/-500; approximate Hessian critical points and ad hoc tolerances | Directed MPFR tanh/sech; critical locations and values enclosed by interval operations; no clipping |
| `lib_hessian_enclosures.py` | NumPy products/einsum and `fastmath=True` JIT paths do not establish inclusion | Forward interval differentiation with no BLAS, JIT or reassociation in the certificate path |
| `lib_lp_enclosures.py`, `lib_w1p_enclosures.py`, `lib_w2p_enclosures.py` | Ordinary powers/roots/volume arithmetic; second-order aggregation counts both mixed entries | Direct nonnegative integrand enclosure; each mixed multi-index once; integer powers and roots rounded outward |
| Historical experiment scripts | Floating geometry, ordinary sums, incompatible plotted indicators, no archived checkpoint mapping | Exact rational geometry and subdivision; directed integral endpoints and global sums; exact rational marking; recorded work and checkpoint |
| ReLU affine-region shortcut | A zero indicator requires verified activation signs and local polynomial integration, including rounding | Not supported by the new path; no zero-indicator shortcut is inherited |

The historical scripts and libraries remain a distinct, **unvalidated** path.
The new Figure 1 experiment imports only `verified_*` for enclosure work. It
does not call the legacy bound routines. No archived figure becomes certified
by adding this implementation.

## Backend choice and trusted components

Backend: `gmpy2` wrapping GNU MPFR/GMP. Two explicit contexts use `RoundDown`
and `RoundUp` at a chosen bit precision (128 by default). All numerical
certificate operations call context methods; they do not rely on the ambient
Python/gmpy2 rounding mode. The contexts trap underflow, overflow, invalid
operations and division by zero. Nonfinite endpoints are rejected.

MPFR specifies correctly rounded arithmetic and elementary functions in directed
rounding modes. This is the backend guarantee used by the inclusion argument,
not a conclusion inferred from sampling tests:

- https://www.mpfr.org/mpfr-current/mpfr.html (Rounding; MPFR Interface)
- https://gmpy2.readthedocs.io/en/latest/contexts.html

FLINT/Arb was also considered: it propagates rigorous midpoint-radius bounds and
would be suitable in principle. Explicit MPFR endpoints were selected for a
direct match to the paper's lower/upper interval formulation and explicit
control of endpoint export and extrema.

Trusted components include Python's control flow and exact integers, GMP rational
arithmetic, gmpy2/MPFR, the installed binaries and hardware. This is an
implementation with a documented inclusion argument and regression tests, not a
machine-checked proof or an independent audit of the dependencies themselves.

## Inclusion argument

1. Float32/float64 checkpoint values are converted through Python floats to
   exact rationals using `as_integer_ratio`. Every coefficient denotes that
   exact real number. This defines a real network; it does not assert equality
   with the finite-precision result of a PyTorch forward pass. The new adapter
   accepts only the known `FlexibleNet` or an alternating Linear/Tanh Sequential.
2. Domain endpoints and subdivision midpoints are exact rationals. Children
   partition their parent, with shared boundaries only. Volume is the exact
   product of rational side lengths, then converted to an interval outward.
   There is no rounded midpoint that can silently collapse a child.
3. Scalar addition, subtraction, multiplication and division use directed
   endpoints. Powers use the exact absolute-range shape (including zero
   crossing) followed by directed positive integer powers. Roots use MPFR's
   directed integer-root operation. Unsupported noninteger powers are rejected.
4. Tanh is monotone. Its first derivative has its maximum at zero and endpoint
   minima. `sech^2` avoids the catastrophic cancellation of `1-tanh^2` in
   saturation. The second derivative is `-2*tanh*sech^2`; endpoints and the
   enclosed critical points `+/-atanh(1/sqrt(3))`, with critical values
   `-/+4/(3*sqrt(3))`, give the range. If a critical-point interval overlaps
   the input interval, its value is included conservatively.
5. Forward interval automatic differentiation applies the real chain rule:
   `D sigma(z) = sigma'(z) Dz` and
   `Dij sigma(z) = sigma''(z) Di z Dj z + sigma'(z) Dij z`.
   Every operation encloses its real counterpart; input dimension is small,
   so carrying input derivatives avoids the legacy cubic Hessian recurrence.
6. Sum `|D^alpha Phi_i|^p` over outputs and multi-indices through the selected
   order (0, 1 or 2), counting each mixed derivative once. Intersect a child's
   integrand interval with its ancestor interval: both enclose the integrand
   on that child. This safely retains earlier information despite roundoff.
7. Multiply each range by an outward enclosure of its exact volume. Accumulate
   global lower and upper endpoints downward and upward. Their upward-rounded
   difference is a bound on enclosure width. Take outward roots of these
   nonnegative endpoints for norm intervals. No Monte Carlo denominator enters
   a certificate or tolerance decision.

Marking sorts exact rational widths of the *computed* local integral intervals,
then reaches the exact rational bulk threshold and includes cutoff ties. The
reported global gap also includes global accumulation rounding. The paper's
exact-arithmetic rate theorem is not a claim that a fixed-precision gap tends
to zero; this implementation has explicit budget and gap-stagnation statuses.
There is no automatic precision increase. Re-evaluate at a larger `--bits`
value when needed. No unconditional finite-precision convergence claim is made.

## Supported scope

- Fully connected tanh networks with affine output, or explicit affine networks.
- Any positive input/output dimension; derivative order 0, 1, or 2.
- Positive integer p, including the Figure 1 case p=2.
- Multi-index Sobolev integrals and their outward norm roots.
- Generic nonnegative scalar integrand callbacks in `Integrator`, provided the
  caller establishes their inclusion property and uses the same arithmetic.
- All-coordinate dyadic subdivision, uniform or Dörfler marking.

RecurJac, sigmoid, ReLU affine-region integration, arbitrary PyTorch graphs,
noninteger p, PDE stability estimates and full PINN certificates are outside
this verified implementation. Unsupported options fail explicitly.

## Setup and checks

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-verified.txt
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover exact-rational arithmetic oracles, cancellation and sums,
near-zero powers/roots, tanh extrema and saturation, ambient context changes,
affine derivatives, high-precision differentiation through multiple layers,
checkpoint conversion, exact tiny-box subdivision, an analytic integral,
zero gaps, budgets and stagnation. Numerical differential/sampling checks are
regression evidence; the preceding operation-level argument establishes the
implementation's intended inclusion property conditional on its trusted backend.

## Two-bump experiment

```sh
.venv/bin/python exp_verified_two_bumps.py --output results/two-bumps-pilot --max-leaves 64
.venv/bin/python exp_verified_two_bumps.py --reuse results/two-bumps-pilot --output results/two-bumps-figure1 --max-leaves 1024
```

The first command trains and saves a [2,32,32,32,1] float64 tanh network; the
second uses the *same* exact coefficient record. Both policies use the same
W^{2,2} integrand, precision, domain and initial box. Uniform refinement is
full-level; its attainable counts are 1,4,16,64,... . Adaptive refinement stops
before a full bulk set would exceed the leaf budget. Panel titles state actual
counts, which can differ. Curve endpoints must not be interpreted as equal-work
results unless the costs match. Raw records retain leaf counts, cumulative
enclosure evaluations and elapsed integration time at every recorded partition.

The stopping status reports tolerance, leaf/step/depth budget or stagnation.
The default tolerance is zero: reaching the budget is not tolerance success.
The experiment does not claim that training error, sampling, or decay of the
gap proves a PDE error bound. Plotting and training use ordinary arithmetic;
float conversion for rendering never feeds back into certificates or marking.

Exact endpoints and coefficients are exported as rational JSON strings, not
rounded decimal approximations. Each run saves the training points, checkpoint,
fit diagnostic grid, exact coefficient record, source hashes, dependency versions,
Git state, configuration and both policy histories/partitions. Keep these with
any paper figure. `results/` is ignored by Git; archive runs separately with
checksums instead of committing changing generated outputs to the source tree.
