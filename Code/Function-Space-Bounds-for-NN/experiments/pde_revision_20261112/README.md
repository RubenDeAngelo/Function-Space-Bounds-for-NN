# PDE revision, 12 November 2026

This package supplies the approved main-paper ReLU energy/boundary example and
exactly two appendix tanh interior PDE-residual experiments. No PDE solution-error
or energy-to-error estimate is claimed. The network coefficients are frozen during
certification; a certificate encloses an integral of the exact real network with
those stored coefficients, not floating-point inference output.

## Quantities and provenance

- `corner_source/`: byte-preserved RICAM source/checkpoint/record copies. The
  model is Cartesian ReLU (2,64,1), seed 0, 257 binary32 parameters, SHA-256
  `024928721d94fa6400f5f02c00df6eda9c203de1dc2c100f127c44b211556df9`.
  The **only presented quantities** are J(v) and B2(v), with sqrt(B2) enclosed
  separately. The raw source records contain additional quantities from their
  original application, which are outside this revision's scope. The saved
  joint partition sequence is not J-maxgap or B2-maxgap. No box-rate theorem is
  applied to polar cells. See `corner_source/provenance.json` for original
  locations and hashes. The original verifier binary hash matches the archived
  run metadata; its arithmetic/geometry self-tests passed again in this revision.
  The manuscript retains the original records; a subsequent full validation replay with the copied archived executable reproduces every J/B2 hexadecimal endpoint and work count exactly (validation/corner-replay-result.json).
- `runs/corner-energy-boundary.csv`: only approved quantities extracted from
  the exact hexadecimal binary64 endpoints. The extraction checks decimal/hex
  agreement and checkpoint identity. Final counts are 1,048,576 interior cells,
  1,246 boundary cells, 2,099,634 evaluations; stop: interior leaf cap.
- `poisson_model/`: unchanged (2,32,32,32,1) archived tanh network from
  `results/paper-maxgap-v2/tanh/models/pinn`. Original initialization seed 42,
  data seed 1042; 10,000 Adam steps, learning rate .001, weight decay .001,
  1000 fresh interior points and 200 fresh points/edge/step. Forcing is
  Delta tanh(x+y-1); prescribed Dirichlet data are tanh(x+y-1).
  Source model metadata is preserved. These new runs use a new local
  implementation and are not asserted to equal the older tighter derivative
  range implementation's endpoints.
- `cdr_model/`: newly jointly trained ordinary tanh (2,32,32,32,1) network.
  Every parameter is trained. See `train_cdr.py` and `manifest.json`: binary64,
  CPU, 1 thread, deterministic PyTorch, default Linear initialization seed 42,
  10,000 full-batch Adam steps, lr .001, zero weight decay. Fixed base-(2,3)
  Halton points with a NumPy random shift seeded 1042: 3500 interior and 875
  per edge. Saved arrays define the finite training problem. The PDE is
  -Delta v+(3-x)v_x+(4-y)v_y+v=f, unit diffusion. Forcing comes from
  sin(x)(1-exp(-2(1-x))) y^2(1-exp(-3(1-y))). Homogeneous Dirichlet on x=0,
  x=1,y=1 and outward Neumann on y=0. The loss sums the interior mean squared
  residual and each of the four separate boundary mean squared residuals.
  Manufactured solution values are never training labels. Training history
  metrics are noncertified and are evaluated before the corresponding update.

## Residual inclusion construction

`residual_kernel.cpp` uses MPFR precision 128, RNDD/RNDU in every local operation,
exact rational coefficient loading with representability checks, directed fused
multiply-add for affine maps, and the full chain rule for value/x/y/xx/yy jets.
Tanh is monotone; its derivative bounds use interval expressions 1-tanh(Z)^2
and -2 tanh(Z)(1-tanh(Z)^2). Squares use the exact interval-square range;
products use all endpoint combinations. These are inclusion-preserving natural
extensions, not tight scalar derivative ranges. The analytic forcing uses the
same directed operations, exp, and monotone sin/cos ranges on [0,1].

Dyadic geometry, endpoint-times-volume products, global sums, maxgap priorities
and tolerance comparisons are exact GMP rationals. Descendant ranges intersect
ancestor ranges. Every replacement asserts parent/child integral nesting.
Maxgap splits both coordinates of one largest-gap leaf, with creation-order
ties. Uniform checks completed levels. For N leaves, E=1+4(N-1)/3. Tolerance
is the **full squared residual integral gap**, 1/10000; cap 65,536 leaves, E=87,381.
All four runs exhaust this cap; no tolerance success or solution-error estimate
is claimed. In particular the CDR upper bounds remain broad despite a small
sampled training loss. The Poisson policies coincide at the recorded complete
levels. Endpoint fractions, stop metadata and descriptive timings are in runs/.
Fixed precision can limit further width reduction; the exact-arithmetic rate
analysis is not a claim of decay below a rounding floor.

## Build and replay

Use the recorded Python environment (or equivalent versions in environment.json).
The original local virtual environment is the code repository's `.venv`.
From this package directory:

```sh
clang++ -O3 -std=c++17 residual_kernel.cpp \
  -I/opt/homebrew/include \
  -I'/Users/moritz/Projekte/RICAM Presentation/Code/.deps/include' \
  -L'/Users/moritz/Projekte/RICAM Presentation/Code/.deps/lib' \
  -L/opt/homebrew/lib -lmpfr -lgmpxx -lgmp -o residual_kernel
../../.venv/bin/python -B validate_residual.py
../../.venv/bin/python -B replay.py
../../.venv/bin/python -B render_assets.py
```

The include/library locations are the recorded local MPFR/GMP installation;
other machines must supply their installation paths. `replay.py` regenerates
all four 1024-leaf checkpoints in a temporary directory and compares every
saved exact endpoint and evaluation count. `replay.py --full` replays the full
four runs; it does not overwrite the saved records. `render_assets.py` renders
the manuscript figures and tables and verifies printed residual/root endpoints
against exact rational values. It also uses the existing frozen hinge/disk
histories from the surrounding repository.

To repeat training without overwriting this checkpoint, copy train_cdr.py to a
new directory and run it there. It refuses to replace cdr_model. Dependencies,
points, source hash and network hash are saved. Runtime and version differences
can change a newly trained model; the saved coefficient export defines this
experiment's certificate target.

## Validation performed

`validation/result.json`: 20 independent 100-decimal-digit residual values from
mpmath numerical differentiation of value-only full networks and manufactured
forcing lie in tested interval enclosures; 20 dyadic-box tests of an exactly
represented tanh Poisson solution enclose zero. Full histories pass exact
rational nesting, aggregation and work-count checks. `validation/replay-1024.json`
records exact replay agreement for both PDEs and both policies. The complete four histories at 65,536 leaves also replay exactly (validation/replay-full.json); their final source and executable hashes are in manifest.json. The imported
corner kernel passes its arithmetic, affine-network, area, moment and
trigonometric self-tests, and its full saved J/B2 histories replay exactly. Printed endpoints and square roots are checked by
exact inequalities. These are implementation checks, not formal software
verification and not PDE solution-error estimates.

For the imported corner computation, a full replay uses the preserved executable
(the same MPFR/GMP runtime assumptions still apply):

```sh
corner_source/verify_archived --model corner_source/weights.bin --precision 64 \
  --seconds 600 --leaves 1048576 --boundary 262144 --depth 40 \
  --history-out /private/tmp/corner-replay-history.txt \
  > /private/tmp/corner-replay.jsonl
```

The manuscript uses only J and B2 from that raw output. The saved validation
comparison checks all original records' J/B2 hexadecimal endpoints, partition
counts and evaluation counts; other original-program outputs are outside the
paper's scope. The copied source snapshot is hashed separately from the archived
executable; binary identity with the original run is established by its hash.
