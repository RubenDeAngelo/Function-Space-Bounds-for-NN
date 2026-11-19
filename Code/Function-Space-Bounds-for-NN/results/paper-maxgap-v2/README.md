## Current manuscript status (31 October 2026)

The candidate material is now integrated into the manuscript. Statements below about an unchanged paper describe the earlier standalone research stage. Numerical records are unchanged. Public release is deferred at the author's request.

# Maxgap/ReLU manuscript artifact, 30 October 2026

This local artifact is installed in Code/Function-Space-Bounds-for-NN/results/paper-maxgap-v2.
It includes 16 tanh and 8 ReLU cases, with 48 maxgap/uniform records and
24 original Dorfler records on the identical frozen networks. No retraining
was performed for this revision. Original reports describe the earlier isolated
execution stage; their statements that the paper was unchanged apply to that stage.

- tanh/: saved coefficients, checkpoints, data, records, source snapshots and environments.
- relu/: corresponding deep [2,32,32,32,1] and shallow [2,200,1] ReLU artifacts;
  dorfler.json is copied from the earlier adaptive run for each case.
- all_endpoints.csv: all 72 terminal records, exact integral/root endpoints and costs.
- replay.py: portable replay using the appropriate frozen source backend for each case.
- tools/: figure/table renderer and exact printed-endpoint validator.
- manuscript_checks.json: exact rounding, table/CSV and reference checks.
- SHA256.json: complete package hashes (excluding itself and runtime caches).

The frozen ReLU backend supports value and almost-everywhere weak-gradient
enclosures, rejects H2, and does not use affine-region polynomial integration.
The tanh backend supports the full multi-index H2 convention and the saved
interior residual adapter. All certificates use 128-bit directed MPFR arithmetic;
network coefficients and box geometry are exact rational values. The portable
replay compares all recorded fields and every maxgap tree event exactly, except
wall times. It never trains or overwrites an output directory.

From the code checkout, with the recorded Python environment:

```sh
.venv/bin/python results/paper-maxgap-v2/replay.py --output results/paper-maxgap-v2-replay
# Optional single case:
.venv/bin/python results/paper-maxgap-v2/replay.py --case relu-deep-disk-H1 --output results/relu-disk-check
.venv/bin/python results/paper-maxgap-v2/tools/render_revision.py --artifact results/paper-maxgap-v2 --latex ../../LaTeX
.venv/bin/python results/paper-maxgap-v2/tools/validate_revision.py --artifact results/paper-maxgap-v2 --latex ../../LaTeX
```

Rebuild both documents from the manuscript directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error maxgap_supplement.tex
```

The source-snapshot audit scripts independently scan all active indicators at
every split, check geometry/ancestor inclusion and validate the final records.
To run the tanh backend tests, use PYTHONPATH=tanh/sources with unittest discovery
in tanh/sources/tests; separate maxgap tests are in tanh/sources/test_maxgap.py.
The ReLU/maxgap tests are in relu/sources/test_relu.py and test_maxgap.py.
Fresh logs are preserved. These checks support the implementation, not a formal
verification of the backend. All runs exhaust budgets; broad enclosures remain.
Equal enclosure-evaluation counts do not imply equal runtime, and the artifact
contains no claim of superiority over established validated higher-order quadrature.

This is a local artifact reference, not a public persistent deposit. Historical
source runners retain original machine paths for provenance. Use replay.py for
portable certificate replay; original training recipes and data are preserved
separately and training reproduction is not asserted across environments.
