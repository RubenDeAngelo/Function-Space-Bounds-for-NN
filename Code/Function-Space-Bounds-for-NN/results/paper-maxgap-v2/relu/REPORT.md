# ReLU experiments: one maximum-gap leaf per refinement

All four trained ReLU checkpoints are reused without retraining. Architectures: deep [2,32,32,32,1], shallow [2,200,1]. Both two-bump and disk experiments are rerun for L2 and full H1. These are bounds on the network norms, not fitting errors. The saved checkpoints, coefficients and training records are included. Norm estimates and fit diagnostics remain those in the previous experiments because the networks are unchanged.

## Refinement rule

At each step select exactly one current leaf maximizing its local integral upper endpoint minus lower endpoint, with exact rational comparison. Break ties by smallest creation id. Subdivide that leaf into four equal dyadic children, bisecting both coordinates. Child integrand bounds are intersected with their parent bounds. Use the same 128-bit directed MPFR enclosure oracle, domains and coefficients as before. Each step adds three leaves; both policies finish at exactly 4096 leaves for two bumps and 1024 for disk. The new maximum-gap policy runs to this budget without an early stagnation heuristic. No positive tolerance is requested.

Uniform integration was rerun for all eight cases and its complete records, excluding wall times, were asserted identical to the previous saved uniform records. Thus all changes in gaps relative to the old adaptive runs result from the refinement strategy, not the networks or oracle.

## Equal-size final gap comparison

Gaps below are U−L for the squared norm integral. Positive reduction means maximum-gap marking is better.

| Architecture | Experiment | Norm | Leaves (both) | Max-gap U−L | Uniform U−L | Reduction vs uniform |
|---|---|---|---:|---:|---:|---:|
| deep | two-bumps | L2 | 4096 | 0.788678408 | 0.84362042 | 6.51% |
| deep | two-bumps | H1 | 4096 | 15.5669113 | 19.9023808 | 21.78% |
| deep | disk | L2 | 1024 | 0.813657438 | 1.27452127 | 36.16% |
| deep | disk | H1 | 1024 | 55.6998163 | 74.6837877 | 25.42% |
| shallow200 | two-bumps | L2 | 4096 | 0.804181278 | 0.803150593 | -0.13% |
| shallow200 | two-bumps | H1 | 4096 | 6.07866349 | 7.03127124 | 13.55% |
| shallow200 | disk | L2 | 1024 | 5.12420412 | 5.12724251 | 0.06% |
| shallow200 | disk | H1 | 1024 | 13.1927514 | 13.3390391 | 1.10% |

## Final norm enclosures and cost

Endpoints are displayed with outward rounding to six decimal places. Both policies have equal leaf counts and enclosure-evaluation counts in each case. Timings are single-run observations; different history-recording schedules and marking bookkeeping affect wall times, so these are not controlled speed benchmarks. Maximum-gap histories are saved every 32 splits, at uniform levels and at the final partition; every individual split is retained in maxgap_audit.json.

| Architecture | Experiment | Norm | Policy | Norm enclosure | Evaluations | Seconds |
|---|---|---|---|---|---:|---:|
| deep | two-bumps | L2 | maxgap | [0.213440, 0.913365] | 5461 | 19.66 |
| deep | two-bumps | L2 | uniform | [0.141839, 0.929376] | 5461 | 20.51 |
| deep | two-bumps | H1 | maxgap | [1.274186, 4.146139] | 5461 | 59.76 |
| deep | two-bumps | H1 | uniform | [0.769412, 4.527072] | 5461 | 59.15 |
| deep | disk | L2 | maxgap | [2.060976, 2.249730] | 1365 | 5.00 |
| deep | disk | L2 | uniform | [2.002658, 2.298948] | 1365 | 4.98 |
| deep | disk | H1 | maxgap | [3.965590, 8.451374] | 1365 | 14.70 |
| deep | disk | H1 | uniform | [2.764417, 9.073357] | 1365 | 14.67 |
| shallow200 | two-bumps | L2 | maxgap | [0.207130, 0.920372] | 5461 | 9.91 |
| shallow200 | two-bumps | L2 | uniform | [0.193294, 0.916796] | 5461 | 10.12 |
| shallow200 | two-bumps | H1 | maxgap | [1.896180, 3.110332] | 5461 | 27.54 |
| shallow200 | two-bumps | H1 | uniform | [1.717781, 3.159438] | 5461 | 26.87 |
| shallow200 | disk | L2 | maxgap | [1.529859, 2.732156] | 1365 | 2.49 |
| shallow200 | disk | L2 | uniform | [1.523700, 2.729269] | 1365 | 2.48 |
| shallow200 | disk | H1 | maxgap | [3.620444, 5.128389] | 1365 | 6.79 |
| shallow200 | disk | H1 | uniform | [3.591528, 5.122316] | 1365 | 6.78 |

All sixteen new policy runs ended with leaf_budget_exhausted. Comparisons with previous Dörfler runs use the same maximum budget but the previous Dörfler partitions can have fewer leaves. These are eight fixed-network cases, not a population study.

## Checks and provenance

The new maximum-gap tests check that exactly one box is selected, stable tie-breaking and correct work counts, and selection of the actual largest gap. Three existing ReLU tests also passed. Every selection in all eight maximum-gap runs was independently checked by a linear scan of exact leaf indicators, without using the heap employed by the runner. The audit also checks every subdivision, ancestor inclusion, local volume aggregation and final active partition. The standard exact-record checks cover global geometry, endpoint aggregation and rooted endpoints for all sixteen records. Uniform records replay exactly apart from time. Checkpoint checks ensure the portable coefficients match the saved PyTorch state. See validation.json and execution.log.

All paper TeX/PDF files and all previous ReLU result files were checked unchanged. This separate report is the reproducibility registry for the new results. Source files and environment hashes are included; run sources/run_maxgap.py in a fresh result location with the recorded environment and the previous result folders available. Then run sources/audit_maxgap.py and sources/report_maxgap.py.
