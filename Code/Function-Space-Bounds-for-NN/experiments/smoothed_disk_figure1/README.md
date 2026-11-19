# Smoothed-disk ReLU experiment for the proposed Figure 1

This package preserves the data behind the RICAM slide “Smoothed disk:
certified integration of a ReLU network.” The two vector panels in panels/
are the slide's integral-gap plot and its paired refinement maps. Figure
assembly uses just these two panels. The slide's rooted-norm plot is separate.

The experiment uses the same frozen (2,32,32,32,1) checkpoint as
results/paper-maxgap-v2/relu/deep/disk. The exact stored binary64 coefficients
are recorded as fractions in model/network.json. The checkpoint and network
hashes are in data/provenance.json. Training provenance is in model/training.json.
Replay evaluates those coefficients and does not need PyTorch or retraining.
The integration domain is [-1.5,1.5]^2 and the quantity is the integral of the
squared network output. The smoothed disk is the fitting target.

## Numerical method and saved data

Both policies use the exact-rational hybrid rule: test preactivation signs by
affine propagation, integrate the square of the affine output exactly on
resolved boxes, and otherwise use natural interval ranges with ancestor
intersections. Maxgap bisects both coordinates of one largest-gap leaf, with
creation-order tie breaking. Uniform completes entire levels. The original
GMP rational backend and cached exact-integer implementation are included.
The runner starts from the root using the same cached implementation.
The original million-leaf records used a continuation from saved smaller runs.
NumPy's object arrays in the cached implementation hold GMP integers.

data/history.csv contains all 33 saved gap records from the original study.
The original run archives were checked against this compact history before
import. The two compressed partitions contain the original 16,384-leaf
snapshots and exact local endpoints. Grey in the slide maps denotes resolved
cells with zero integral width. The color scale is shared. These maps use a
different budget from the terminal comparison.

| Leaves per policy | Hybrid maxgap gap | Hybrid uniform gap |
| ---: | ---: | ---: |
| 16,384 | 0.02991280565 | 0.17277599245 |
| 1,048,576 | 0.00021260794 | 0.00507566328 |

Both policies have 1,398,101 local calls at the last budget. These are
budget-limited runs. The gap reduction is approximately 95.81% at the
largest matched count. The N^-2 triangle is the two-dimensional theoretical
reference for the squared L2 maxgap bound. The plotted lines connect recorded
states. The comparison concerns the width of the frozen-network integral
enclosure. Fitting error, runtime and a fitted convergence exponent are outside
that comparison.

## Replay and checks

Use Python with gmpy2 and NumPy. The local verification used gmpy2 2.3.2 and
NumPy 2.5.4. From this folder:

    python -B -m unittest -v test_fast
    python -B validate.py
    python -B run_experiment.py --policy maxgap --cap 16384 --output /new/maxgap-directory
    python -B run_experiment.py --policy uniform --cap 16384 --output /new/uniform-directory
    python -B validate.py --maxgap-replay /new/maxgap-directory --uniform-replay /new/uniform-directory

Both 16,384-leaf runs were replayed and matched every saved history row through
that budget and every cell in the corresponding partition. The five oracle
tests also passed. The imported data/original-million-validation.json describes
the original full-budget checks. The 1,048,576-leaf runs were not rerun for this
import. That cap is the runner default and needs substantially more time and
memory. The runner writes sparse exact histories and saves the 16,384-leaf
partition when it reaches that budget.

To assemble the original vector panels in a standalone preview, run
latexmk -pdf figure.tex from this folder, or use its -outdir option to keep
the build elsewhere.
