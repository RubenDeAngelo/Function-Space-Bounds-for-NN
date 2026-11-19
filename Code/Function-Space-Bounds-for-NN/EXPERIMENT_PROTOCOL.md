# Experimental protocol

All new paper runs use exact stored binary weights, supported tanh networks,
p=2, one mixed derivative per multi-index, MPFR at 128 bits, and full integral
gap U-L. Geometry, volumes and the marking threshold are rational; all-coordinate
dyadic subdivision starts from one box. Natural local integrand intervals are
intersected with ancestor ranges. Marking uses the largest contributions until
theta=7/20 of their sum is reached, including ties. A marked batch is never
truncated to fit a leaf budget. Uniform refinement takes full levels.

A run exports full integral and rooted norm endpoints, both gaps, raw local
ranges and volumes (through exact bounds), history, actual leaves, cumulative
range-oracle evaluations and wall time. No Monte Carlo normalization is used.
Rooted gaps are differences of outward roots, never roots of integral gaps.
The configured tolerance concerns the full integral gap. Report unmet tolerances
and budget/stagnation statuses. Plotting floats never re-enter the certificate.

The new 1D study has three predefined seeds (0,1,2), not 100 purportedly recovered
runs. Training is paired with each seed's saved untrained network. All traces are
shown; no inferential statistics or claims about typical architectures follow.
The target exp(-100(x-0.5)^2) is on [0,1], sampled on a fixed 150-point grid.
Both H1/H2 use the same checkpoints; budgets are 256 leaves per policy.

Disk H1/H2 use one saved scalar deep tanh network. The target is the C-infinity
smoothed disk with epsilon=0.5 on [-1.5,1.5]^2. Exactly 5000 uniform float64
samples are stored. Architecture [2,32,32,32,1], initialization seed 20261024,
independent NumPy data seed 20261025, 10000 Adam steps, lr=0.001, decay=0.001.
Each norm/policy has a 1024-leaf budget. Heatmaps and convergence come from the
same raw run records. Color scales match adaptive/uniform within each norm row.

The PINN uses u=tanh(x+y-1), Delta u=-4(1-tanh(z)^2)tanh(z), boundary data u.
Same deep tanh architecture; initialization seed 42, independent CPU sample
generator seed 1042, 10000 Adam steps, lr=0.001, decay=0.001. Each step uses
1000 interior samples and 200 points per boundary edge; loss is interior MSE
plus boundary MSE. Training is ordinary float64 on CPU, one thread, deterministic
algorithms. Only the saved network is used for interval certification, with a
1024-leaf budget and the validated forcing adapter. Boundary training is not
boundary certification; interior norm bounds are not solution-error bounds.

Source snapshots and hashes record the implementation actually executed, including
model definitions and experiment configuration. Training replay is distinct from
certificate replay. Saved checkpoints define the reproducibility requirement;
training at another software/platform combination may differ. Timings are
single-run observations unless explicitly repeated; training and rendering are
excluded from recorded integration times. Stated localizations concern computed
gap contributions, not necessarily the true function or residual.

No new main-text numerical claim relies on the historical ReLU/PINN/Sobolev
scripts. Their old figures and source are preserved in the archival material.
