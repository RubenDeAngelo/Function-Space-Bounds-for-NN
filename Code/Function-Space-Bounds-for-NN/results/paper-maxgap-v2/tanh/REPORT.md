# Current paper: maximum-gap marking reruns

## Scope and method

All 16 cases used in the current compiled manuscript are included: the two-bump L2 case; H1 and H2 for paired untrained/trained 1D networks, seeds 0, 1 and 2; disk H1 and H2; and the PINN squared interior residual. These are the saved tanh networks in the active paper. Archived experiments and the separate ReLU runs from this chat are outside this scope. No network is retrained. The saved checkpoint coefficients, training artifacts and configurations are retained. The paper directory is unchanged.

The sole numerical-method change is marking: choose exactly one current leaf with maximum local integral endpoint gap U_K−L_K. Comparisons use exact rational indicators, and the smallest creation id breaks ties. Split the selected interval into two children in 1D or box into four children in 2D, halving every coordinate. Use ancestor intersections and the paper source snapshot for the enclosure oracle, 128-bit directed MPFR arithmetic, exact rational geometry, and the saved coefficients interpreted as exact real numbers. All maxgap runs continue to the specified leaf budget; no early stagnation heuristic is applied. Positive stopping tolerances are disabled.

The final leaf counts are exactly matched between maximum-gap and uniform policies: 256 in 1D, 4096 for two bumps, and 1024 for disk and PINN. Corresponding enclosure evaluations are 511, 5461, and 1365. Uniform integration was rerun in all 16 cases and every record was checked to reproduce the saved paper uniform run exactly, excluding timings. The original Dörfler records are also included, but their final leaf counts can be lower.

## Equal-size final integral gaps

Gaps are full U−L for the integral defining the squared norm, or the squared interior residual. Positive reduction means maxgap is smaller. Negative reduction means maxgap is larger. These are enclosure gaps, not the norms, network fitting errors or PDE solution errors.

| Experiment | Leaves (both) | Maximum-gap U−L | Uniform U−L | Reduction vs uniform | Paper Dörfler U−L (leaves) |
|---|---:|---:|---:|---:|---:|
| two-bumps-L2 | 4096 | 0.832886049 | 1.171997572 | 28.93% | 0.8771073954 (3850) |
| one-d-seed0-untrained-H1 | 256 | 0.008448477467 | 0.008448477467 | 0.00% | 0.009671974815 (235) |
| one-d-seed0-untrained-H2 | 256 | 0.009234086668 | 0.009234086668 | 0.00% | 0.01101948128 (229) |
| one-d-seed0-trained-H1 | 256 | 0.8787984468 | 1.938181192 | 54.66% | 0.9611745251 (234) |
| one-d-seed0-trained-H2 | 256 | 245.4646669 | 621.7616918 | 60.52% | 246.5651885 (255) |
| one-d-seed1-untrained-H1 | 256 | 0.01330996879 | 0.01330996879 | 0.00% | 0.0145988469 (241) |
| one-d-seed1-untrained-H2 | 256 | 0.01403961785 | 0.01403961785 | 0.00% | 0.01622406031 (232) |
| one-d-seed1-trained-H1 | 256 | 1.1189262 | 2.444924541 | 54.23% | 1.153632676 (248) |
| one-d-seed1-trained-H2 | 256 | 300.7996747 | 743.999579 | 59.57% | 351.9755468 (223) |
| one-d-seed2-untrained-H1 | 256 | 0.008347018708 | 0.008347018708 | 0.00% | 0.009534646388 (234) |
| one-d-seed2-untrained-H2 | 256 | 0.009469385421 | 0.009469385421 | 0.00% | 0.01038497971 (241) |
| one-d-seed2-trained-H1 | 256 | 1.055605627 | 2.286523046 | 53.83% | 1.175326463 (229) |
| one-d-seed2-trained-H2 | 256 | 306.8527212 | 811.243227 | 62.18% | 341.281656 (231) |
| disk-H1 | 1024 | 36.25760068 | 36.40763746 | 0.41% | 38.61535028 (865) |
| disk-H2 | 1024 | 307.3897919 | 315.5660334 | 2.59% | 378.9889102 (709) |
| pinn-residual | 1024 | 0.03407396522 | 0.03454611213 | 1.37% | 0.04213271509 (757) |

## Integral and rooted-norm enclosures

The following endpoints are rounded outward to six decimal places. PINN rooted endpoints are the interior residual L2 norm, not a solution-error bound. H2 follows the paper multi-index convention with each mixed derivative counted once. Norm/root gaps differ from integral gaps.

| Experiment | Policy | Integral [L,U] | Rooted norm interval | Evaluations | Seconds |
|---|---|---|---|---:|---:|
| two-bumps-L2 | maxgap | [0.020229, 0.853116] | [0.142230, 0.923643] | 5461 | 21.33 |
| two-bumps-L2 | uniform | [0.000608, 1.172607] | [0.024675, 1.082870] | 5461 | 21.28 |
| one-d-seed0-untrained-H1 | maxgap | [0.030303, 0.038753] | [0.174079, 0.196856] | 511 | 5.99 |
| one-d-seed0-untrained-H1 | uniform | [0.030303, 0.038753] | [0.174079, 0.196856] | 511 | 5.93 |
| one-d-seed0-untrained-H2 | maxgap | [0.030478, 0.039713] | [0.174580, 0.199281] | 511 | 7.99 |
| one-d-seed0-untrained-H2 | uniform | [0.030478, 0.039713] | [0.174580, 0.199281] | 511 | 8.00 |
| one-d-seed0-trained-H1 | maxgap | [11.994400, 12.873199] | [3.463293, 3.587925] | 511 | 6.55 |
| one-d-seed0-trained-H1 | uniform | [11.452503, 13.390685] | [3.384154, 3.659329] | 511 | 6.15 |
| one-d-seed0-trained-H2 | maxgap | [3647.889290, 3893.353958] | [60.397758, 62.396747] | 511 | 8.12 |
| one-d-seed0-trained-H2 | uniform | [3459.197742, 4080.959434] | [58.814944, 63.882388] | 511 | 8.15 |
| one-d-seed1-untrained-H1 | maxgap | [0.069695, 0.083006] | [0.263998, 0.288107] | 511 | 6.16 |
| one-d-seed1-untrained-H1 | uniform | [0.069695, 0.083006] | [0.263998, 0.288107] | 511 | 6.02 |
| one-d-seed1-untrained-H2 | maxgap | [0.070130, 0.084170] | [0.264820, 0.290121] | 511 | 8.10 |
| one-d-seed1-untrained-H2 | uniform | [0.070130, 0.084170] | [0.264820, 0.290121] | 511 | 8.16 |
| one-d-seed1-trained-H1 | maxgap | [11.906169, 13.025096] | [3.450531, 3.609030] | 511 | 6.00 |
| one-d-seed1-trained-H1 | uniform | [11.232015, 13.676941] | [3.351419, 3.698235] | 511 | 6.00 |
| one-d-seed1-trained-H2 | maxgap | [3613.492553, 3914.292229] | [60.112332, 62.564305] | 511 | 8.07 |
| one-d-seed1-trained-H2 | uniform | [3392.120874, 4136.120454] | [58.241916, 64.312678] | 511 | 8.11 |
| one-d-seed2-untrained-H1 | maxgap | [0.022644, 0.030992] | [0.150480, 0.176044] | 511 | 6.01 |
| one-d-seed2-untrained-H1 | uniform | [0.022644, 0.030992] | [0.150480, 0.176044] | 511 | 6.05 |
| one-d-seed2-untrained-H2 | maxgap | [0.023007, 0.032478] | [0.151683, 0.180215] | 511 | 8.13 |
| one-d-seed2-untrained-H2 | uniform | [0.023007, 0.032478] | [0.151683, 0.180215] | 511 | 8.31 |
| one-d-seed2-trained-H1 | maxgap | [11.902292, 12.957898] | [3.449969, 3.599708] | 511 | 6.05 |
| one-d-seed2-trained-H1 | uniform | [11.273529, 13.560053] | [3.357607, 3.682398] | 511 | 6.00 |
| one-d-seed2-trained-H2 | maxgap | [3643.100089, 3949.952811] | [60.358098, 62.848651] | 511 | 8.23 |
| one-d-seed2-trained-H2 | uniform | [3394.596619, 4205.839847] | [58.263166, 64.852447] | 511 | 8.21 |
| disk-H1 | maxgap | [10.606368, 46.863970] | [3.256742, 6.845727] | 1365 | 21.36 |
| disk-H1 | uniform | [10.248461, 46.656099] | [3.201321, 6.830527] | 1365 | 22.53 |
| disk-H2 | maxgap | [45.412680, 352.802473] | [6.738893, 18.783037] | 1365 | 38.36 |
| disk-H2 | uniform | [42.382359, 357.948393] | [6.510173, 18.919525] | 1365 | 38.09 |
| pinn-residual | maxgap | [0.000000, 0.034074] | [0.000000, 0.184592] | 1365 | 37.85 |
| pinn-residual | uniform | [0.000000, 0.034547] | [0.000000, 0.185866] | 1365 | 37.89 |

All 32 new policy runs stopped with leaf_budget_exhausted. Timings are single-run observations excluding training, rendering and the independent audit. Maxgap and uniform have different history-recording and marking bookkeeping; timings are not controlled speed benchmarks. Maxgap saves selected intermediate histories and every individual split in maxgap_audit.json.

## Validation and reproducibility

The suite source-snapshot tests and three maxgap tests passed. New tests cover exact maximum selection, selecting exactly one box under ties, stable tie-breaking, and 1D/2D leaf/evaluation counts. An independent linear scan (without the heap used for selection) verifies every individual marking decision. Audits check every subdivision, local integral enclosure/volume, ancestor inclusion, final partition and saved intermediate endpoint aggregation. The standard exact-record checks verify global geometry, volume, endpoints and roots. All 16 uniform records replay exactly apart from timings, and checkpoint/coefficient consistency is checked before integration. These checks support the implementation, not formal verification. See validation.json, tests.log, audit.log and execution.log.

All pre-existing paper-directory files (excluding runtime caches, virtual environment and Git internals) were hashed before execution and checked unchanged afterwards. No TeX, PDF, figure, source code or reproducibility registry in that directory is edited. This standalone report serves as the new result registry.

Reproduce with the recorded environment by copying sources into a fresh result directory, retaining experiments.json and the original source result directories recorded in each manifest, then running sources/run_paper_maxgap.py, sources/audit_paper_maxgap.py and sources/report_paper_maxgap.py. Existing output directories are deliberately not overwritten. Source hashes, exact rational endpoints, full partitions and selection audit trails are included. The original paper source snapshot supplies verified_arithmetic.py, verified_network.py, verified_integration.py and verified_residual.py unchanged; the record validator is the newer repository version.
