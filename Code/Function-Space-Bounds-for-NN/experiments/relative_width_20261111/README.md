# Relative integral enclosure width study (11 November 2026)

The frozen networks and hybrid integral rule are unchanged. The stopping test is
R = 2*(U-L)/(U+L) <= 1/100 (R=0 for [0,0]), evaluated in exact GMP rational
arithmetic. This is full interval width divided by its midpoint, and is twice
the Corliss--Rall relative-error bound for their harmonic estimator when L>0.

kernel.cpp implements the same 2D layerwise affine sign test, polynomial moments,
natural interval value and conservative weak-gradient propagation, and ancestor
intersections as the frozen paper-geometry-v1/backend.py. GMP integer layer
scales and cached affine forms speed up the computations. No floating-point
number affects an endpoint, sign, priority comparison, or stopping decision.
Floating-point seconds are elapsed-time measurements only.

Uniform refinement uses complete levels. A zero-width cell's already certified
integral is reused for its entire subtree: all its descendant widths are zero
under the same rule, so subdivision leaves the aggregate endpoints unchanged.
Consequently, the number of full-grid leaves differs from actually evaluated
cells. Both are reported. Maxgap counts actual evaluations and breaks equal-gap
ties by creation order, in the same child order as the frozen backend.

The source was compiled with clang++ -O3 -std=c++17 and linked against local
GMP/GMPXX. To reproduce, compile kernel.cpp, run prepare_validate.py, test_kernel.py,
and run_suite.py after setting ROOT/REF to the supplied frozen record directory.
The suite runs up to four independent single-threaded jobs concurrently. Elapsed
integration times exclude model parsing and process startup. They are descriptive, include contention, and are not isolated speed benchmarks.

Validation: all 16 trained terminal enclosures at 1024 leaves match the frozen
backend exactly. Randomized tests additionally compare both policies and both
norms with the reference and with independently integrated activation polygons.
All parent/child integral nesting assertions are checked during execution.

models.json stores exact source coefficients, domains and source model hashes.
models/*.txt are exact integer encodings of those coefficients. runs/*.csv store
exact integral endpoints, relative widths, evaluated-cell counts, positive-gap
leaf counts and elapsed times; adjacent .meta files store stopping status and
operation counts. validation/ stores the reference comparisons.

To rebuild the manuscript assets, set PAPER_CODE_ROOT to the code checkout and
PAPER_LATEX_ROOT to the sibling LaTeX directory, then run summarize.py with the
project Python environment. Use the manuscript renderer for frozen figures,
compile LaTeX/disk_figure_revised.tex, copy its PDF into LaTeX/images/, and build
LaTeX/main.tex and maxgap_supplement.tex with latexmk. The renderer keeps all
original fixed-budget tables; summarize.py writes the additional tolerance tables.
