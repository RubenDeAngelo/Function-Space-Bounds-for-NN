# Release validation, 19 November 2026

The release is a standalone snapshot of the code and data supporting the paper
version dated 12 November 2026. The original base-code commit is
`ba433dceab5958fe842ef98569667b9d7cf5f53a`; frozen source hashes identify the
subsequent experiment implementations. The submitted software does not depend
on the newer local consolidated library. No Mothercode files were copied.

## Environment and independence from the development checkout

A fresh Python 3.13 virtual environment was created and the exact 13 dependencies
in `requirements-replay.txt` were installed from the package index. It does not
install PyTorch or the development project. Tests use only the copied source,
frozen networks and records in this snapshot. C++17 kernels were rebuilt using
Apple Clang, GMP 6.3.0 and the installed MPFR headers/library. Other platforms
are not claimed to have been tested. The installation prefixes are configurable;
no original absolute path is required by the release wrapper.

## Checks performed

- All included files checked against the release SHA-256 manifest; all 186
  preserved Python code files parsed and scanned for external consolidation imports.
- Three C++ source builds: exact relative-width kernel, directed-MPFR smooth-PDE
  kernel, and the corner verifier with strict floating-point flags. Corner
  arithmetic, moment, geometry, affine-network and trigonometric self-tests passed.
- Five controlled-study tests, thirteen multidimensional-study tests, and five
  disk-kernel tests passed (23 tests in these three suites).
- Independent controlled-study audit checked 86 saved runs and 6,012 terminal
  ReLU leaf integrals. Multidimensional audit checked 152 saved partitions and
  56,421 independently integrated terminal cells in 84 small-control runs.
- All 40 current one-dimensional hybrid records replayed exactly, excluding time.
- All 16 original 1,024-leaf trained hybrid certificates replayed exactly with
  the rebuilt integer/GMP kernel.
- Eighty seeded randomized policy/quantity comparisons matched the frozen
  rational backend and enclosed 40 independently computed polygon integrals.
- Both 16,384-leaf disk partition snapshots audited against their saved global
  endpoints, dyadic coverage, resolved-cell counts and recorded histories.
- Twenty independent 100-decimal-digit residual-value checks and twenty
  exact-solution zero-residual box checks passed. All four 1,024-leaf smooth-PDE
  checkpoints replayed exactly, including evaluation counts.
- Every saved corner J/B2 record re-extracted from the archived hexadecimal
  endpoints; model/source identity and evaluation counts retained.
- All sixteen relative-width histories audited for exact interval nesting,
  stopping, frozen model identity, evaluated-cell bounds and printed outward
  rounding; all terminal relative full widths are at most 1/100. Seventeen
  shared disk checkpoints match the earlier histories exactly.
- Thirteen convergence panels, twenty prescribed rate triangles, all relevant
  numerical tables and the composite Figure 1 regenerated. Sampled corner
  artwork and disk partition artwork are supplied as original hashed PDFs.
- Main paper and supplement built from separately supplied local manuscript
  sources with generated assets: 34 and two pages. Manuscript bodies/PDFs are
  deliberately not included in the public code release.

The final execution report and logs are under `release/validation/`. They record
the commands and exit codes; paths are normalized for portability. The private
manuscript source files are not copied into those logs.

## Preserved evidence and omissions

The release keeps frozen numerical inputs and histories unchanged. Validation
and rendering occur in a disposable copy. Exact timing fields are preserved as
historical observations, not rerun-performance claims. The original package
READMEs/manifests describe historical stages; the root README provides the
current map and instructions.

Three historical native executables are omitted rather than redistributing
platform-specific linked libraries. Their original hashes are retained in
`EXCLUDED_BINARIES.json` and the original experiment manifests. Source and
portable build instructions are provided. The slope validator's packaging-only
adaptation checks every included frozen PDE file, while explicitly accounting
for the two omitted executables named by that particular manifest; the original
validator is also preserved. Draft manuscript bodies and drafting notes are
omitted with hashes recorded in `EXCLUDED_MANUSCRIPT_MATERIAL.json`.

The complete largest-budget trained trajectories, million-leaf disk runs,
million-leaf corner run and four full 65,536-leaf smooth-PDE trajectories were
not all rerun for this release. Their exact saved histories, original full-replay
evidence and supported reproduction commands are supplied. The checks above
establish the stated release tests and record consistency; they are not formal
software verification, a fresh proof audit or a guarantee of certification on
an untested platform. No new numerical result or PDE solution-error claim is made.
