# Hybrid experiments for the manuscript

This runner selects the hybrid records from the preserved study archives and
replays their original configurations with the frozen numerical backends.
The submission uses the following runs.

| Suite | Hybrid runs | Positive tolerance | Exact zero stop | Budget stop |
| --- | ---: | ---: | ---: | ---: |
| One-dimensional controls | 40 | 12 | 12 | 16 |
| Multidimensional controls and trained networks | 88 | 18 | 4 | 66 |
| Larger-budget smoothed disk | 2 | 0 | 0 | 2 |

The one-dimensional suite uses exact affine integration and a directed MPFR
fallback. The multidimensional and larger-budget disk suites use exact
rational arithmetic throughout the integral enclosure. Both maxgap and uniform
refinement use the same local hybrid rule within each comparison.

Run the following commands from the code checkout with Python and gmpy2.
Choose new output directories. Use repeated --case options for focused replays.

    python -B experiments/hybrid_submission/replay.py --suite controls --output /new/controls-directory
    python -B experiments/hybrid_submission/replay.py --suite geometry --output /new/geometry-directory

The runner compares each result against the saved record, including endpoints,
partitions, work counts, and statuses. Elapsed times are excluded. The frozen
inputs remain in results/paper-originality-v1 and results/paper-geometry-v1.
The paper-maxgap-v2 artifact remains the source of the frozen trained models.

The larger-budget disk replay and figure assembly are in
experiments/smoothed_disk_figure1/. The manuscript asset generator
LaTeX/render_relu_submission_assets.py selects the same hybrid records and
builds the controlled figures, gap tables, and supplement endpoints.

The earlier range-only and mixed campaigns remain archived with their
original renderers and exact records.
