# Revision validation, 30 October 2026

- Current main source/PDF were verified byte-identical to the existing
  Past Versions/24_10_2026.tex and .pdf before any edit. Those archives remain unchanged.
- All 292 original artifact checksums match. Frozen sources, coefficients, data,
  original records and original environments are preserved without mutation.
- All 13 model artifact sets pass checkpoint/coefficient/data consistency checks.
- Independent linear scans verify all 12,272 maximum-gap marking decisions,
  subdivisions, ancestor intersections and final partitions across 24 cases.
- All 48 maxgap/uniform production records were freshly recomputed from exact
  frozen coefficients. Every saved field except elapsed seconds matches exactly;
  all maximum-gap tree nodes and split events also match exactly.
- All 24 original bulk records pass exact geometry, aggregation, rooted endpoint
  and terminal policy/status checks. No bulk replay or retraining is claimed.
- 28 tests pass: 20 tanh backend/record tests, 3 tanh maxgap tests and 5 ReLU/maxgap tests.
- All 72 terminal CSV records and 48 summary entries match their full saved records.
  All 160 printed interval pairs (16 main, 144 supplement) round outward from
  exact endpoints; all eight ReLU printed gap pairs and percentage reductions agree
  with the exact rational records at the displayed precision.
- The maxgap theorem proof was checked using maximum >= mean, gap monotonicity,
  the full-tree count, and the internal-node/leaf identity. Bulk cutoff ties,
  zero gap, initial partition and the separation from geometric contraction were checked.
  No instance optimality or ReLU H1 smooth-width rate is claimed.
- Forward derivative recurrences and the ReLU slope rule were checked against
  the executed frozen backend. ReLU H1 is interpreted through weak gradients
  almost everywhere, including zero level sets; no ReLU H2 computation is claimed.
- Both final documents build with latexmk: 25-page paper, 7-page supplement.
  No undefined references/citations, duplicate labels, missing characters or
  overfull boxes appear in final logs. The two existing warnings remain:
  todonotes margin width and an underfull author-affiliation box.
- All 32 rendered pages were visually inspected in montages, with detailed
  checks of changed theory, figures and tables. Final renders match the inspected
  build pixel for pixel; the final abstract-only clarification changed only page 1,
  which was rendered and inspected again.
- The citation-key set and bibliography are unchanged. No new literature claim
  or source verification is asserted. Reproducibility and revision registries are updated.

Limitations: these checks are not formal verification of the proof, implementation
or MPFR/GMP. All 48 main runs exhaust leaf budgets; broad enclosures and the adverse
shallow two-bump L2 result are retained. Equal oracle counts do not establish equal
runtime. No higher-order validated comparator, new tolerance-success campaign,
public artifact deposit or PDE solution-error certificate is supplied by this revision.
