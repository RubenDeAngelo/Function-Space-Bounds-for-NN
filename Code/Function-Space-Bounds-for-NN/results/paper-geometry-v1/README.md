# Multidimensional certified ReLU integration — local research package

31 October 2026. This package implements the fixed-architecture affine-cell/range oracle in the manuscript's new multidimensional theorem. The theorem and proof are in `manuscript/geometry_results.tex`; its proof/literature audit is `LITERATURE_AND_PROOF_REVIEW.md`. Public release is deferred by author instruction.

## What was executed

- 152 runs: 64 trained-network policy/rule/norm/budget combinations; 64 small 2D controls; 20 shifted hinges; four 3D hinge runs.
- 18 positive-tolerance successes, four exact zero-tolerance terminations and 130 cap exhaustions. Every trained-model run fails its requested full integral gap 0.01.
- Thirteen automated tests pass. A separate `fractions.Fraction` polygon/triangle implementation checks 56,421 terminal control-cell integrals across 84 runs. Exact partition-tree, volume, endpoint aggregation, stopping, work-count and rooted-endpoint checks cover every run.
- All 152 runs replay exactly excluding elapsed time. This checks reproducibility in the recorded environment, not formal software verification or independent mathematical peer review.

## Reproduce saved evidence

Use Python with gmpy2, matching `environment.json` for the recorded environment. No PyTorch, training or network downloads are needed for replay. Matplotlib is additionally needed only for rendering. From this directory:

```sh
python -B -m unittest -v test_study
python -B validate.py
python -B replay.py --all
python -B assets.py --latex /path/to/manuscript/LaTeX
```

`validate.py` refreshes the summary/validation ledgers. `replay.py` does not change completed records. To execute the full experiment suite into a separate fresh directory, with the older frozen artifact available:

```sh
python -B experiment.py --frozen ../paper-maxgap-v2 --output /new/study-directory/results
```

The runner skips completed record filenames and does not retrain or alter coefficients. The records embed their network definitions, so `replay.py` is portable independently of that older package. All records have exact rational endpoint strings. Decimal gaps in manuscript tables are approximations, not certificate endpoints.

## Arithmetic and work

All new integral-certificate operations are exact rational arithmetic through gmpy2/GMP: geometry, network value/weak-gradient intervals, affine signs/coefficients, moments, sums and maxgap comparisons. Conservative ReLU slope intervals include [0,1] whenever the preactivation interval includes zero. This differs slightly from the earlier backend's boundary shortcuts. This package is a separate oracle/accumulation implementation, not a claim of bit-identical MPFR replay of the old range suite.

`summary.json` adds rooted-norm bounds computed using exact integer square roots at 128 fractional binary bits. Their squared endpoints enclose the integral endpoints. Python float appears only in timing and descriptive/plotting conversions, never in certificate endpoints.

Each run records local calls, range-network calls, range weighted terms, affine-cell tests, affine weighted-coefficient terms, neuron sign tests, exact moments and maximum **local** endpoint bit size. These are separate measures; they are not a single runtime currency. History includes full integral endpoints, gaps and resolved-leaf counts. A complete b-ary tree at N leaves uses 1+b(N−1)/(b−1) local calls. Single-run elapsed times include history and exact aggregation overhead and support no general runtime ranking.

## Validation limits

The trained-model enclosures rely on the interval inclusion construction. They are not checked against a global region enumeration; that independent validator is deliberately restricted to small 2D controls. The 3D axis hinge has known analytic integrals checked in tests. General deep-network support of the theorem is mathematical, not inferred from random tests. Constants can grow rapidly with architecture. The theorem is specific to the stated oracle, fixed class, isotropic dyadic geometry and exact arithmetic; it is not an optimality assertion against all integration algorithms.

## Package contents

`backend.py`: exact-rational implementation; `independent.py`: separate 2D polygon expansion and triangle integrals; `test_study.py`: deterministic and randomized checks; `experiment.py`: configurations and runner; `results/`: all complete records; `validate.py`/`validation.json`: independent record checks; `replay.py`: exact rerun; `summary.json`: endpoints, rooted endpoints and work/timing; `assets.py`: manuscript table/figure renderer; `environment.json`: software environment; `SOURCES.json`: original frozen network-file hashes; `SHA256.json`: package integrity manifest; `manuscript/`: integrated new text and assets; `logs/`: executed checks.

The original trained coefficient files and existing manuscript numerical records were not edited. No public repository was updated and no paper was submitted.
