from pathlib import Path
import json,hashlib,platform,sys,csv
from fractions import Fraction as Q
import gmpy2
from experiment_records import dump,sha
root=Path(__file__).resolve().parent.parent
base=Path('/Users/moritz/Projekte/FunctionSpaceNormsOfNeuralNetworks_Research_Paper/Code/Function-Space-Bounds-for-NN')
rows=json.loads((root/'summary.json').read_text());sampled=json.loads((root/'sampled_norms.json').read_text())
def f(x):return float(Q(x))
lines=['# ReLU two-bump and disk experiments','',
'Both networks have architecture [2,32,32,32,1]: three hidden ReLU layers, affine scalar output, 2241 parameters. These are newly trained ReLU networks, using the exact saved training points and labels of the active tanh baselines. The two-bump target is unscaled. The norms are those of the trained network, not the approximation error. H1 is the full norm sqrt(integral(Phi^2 + Phi_x^2 + Phi_y^2)), using weak derivatives almost everywhere.','',
'## Approximate norm values','',
'Midpoint grids of 256², 512² and 1024² points, ordinary float64 arithmetic; these values have no certified error bound.','',
'| Experiment | Grid | L2 | H1 |','|---|---:|---:|---:|']
for name,ss in sampled.items():
    for s in ss:lines.append(f"| {name} | {s['grid']}² | {s['L2']:.9g} | {s['H1']:.9g} |")
lines+=['','## Validated enclosure runs','','128-bit MPFR directed arithmetic, exact binary checkpoint coefficients, exact rational dyadic geometry and marking, theta=7/20, zero stopping tolerance. Interval forward propagation bounds ReLU slopes by 0, 1, or [0,1] on ambiguous preactivations. Weak-gradient chain rules justify H1; no H2 claim is made. The constant-envelope integrator is used for both activations; no exact activation-region polynomial integration is used. Endpoints below are rounded outward for display.','',
'| Experiment | Norm | Policy | Norm enclosure | Squared-norm gap U−L | Leaves | Evaluations | Seconds |',
'|---|---|---|---|---:|---:|---:|---:|']
import math
for r in rows:
    lo=f(r['norm']['lower']);hi=f(r['norm']['upper'])
    low=math.floor(lo*1e6)/1e6;high=math.ceil(hi*1e6)/1e6
    lines.append(f"| {r['experiment']} | {r['norm_name']} | {r['policy']} | [{low:.6f}, {high:.6f}] | {f(r['gap_upper']):.7g} | {r['leaves']} | {r['evaluations']} | {r['seconds']:.2f} |")
lines+=['','Every run ended with `leaf_budget_exhausted`. No positive accuracy tolerance was requested or achieved as a stopping criterion. Adaptive partitions stop before the budget if the next complete marked batch would exceed it; equal budgets need not give equal final leaf counts. Times are single-run integration times, not benchmark averages. Bounds remain broad, especially for H1; sampled norm stability does not make the bounds tighter.','',
'## Training and baseline comparison','','| Experiment | Activation | Grid MSE | Grid maximum absolute fit error |','|---|---|---:|---:|']
for name,baseline in [('two-bumps',base/'results/two-bumps-deep-l2/training.json'),('disk',base/'results/validated-paper-suite-v1/models/disk/model.json')]:
    info=json.loads((root/name/'training.json').read_text());m=info['metrics']
    lines.append(f"| {name} | ReLU | {m['grid_MSE']:.8g} | {m['grid_max_abs_error']:.8g} |")
    b=json.loads(baseline.read_text());m=b.get('metrics',b)
    # Disk baseline stores maximum error only; derive same-grid MSE from its saved arrays.
    import numpy as np
    if name=='disk':
        with np.load(baseline.parent/'fit_grid.npz') as d:mse=float(np.mean((d['prediction']-d['target'])**2))
    else:mse=m['grid_MSE']
    lines.append(f"| {name} | saved tanh | {mse:.8g} | {m['grid_max_abs_error']:.8g} |")
lines+=['',
'Two bumps: seed 20261020, 4096 saved points, 3000 Adam steps at 0.001, then LBFGS max_iter=250, float64 CPU. Disk: seed 20261024, 5000 saved uniform points, epsilon=0.5, 10000 Adam steps at 0.001 with weight_decay=0.001, float64 CPU. Same seeds, optimizers and training budgets as the respective saved tanh runs; changing activation changes optimization behavior. Training uses one thread and deterministic algorithms. The tanh comparison uses existing saved results, not a new tanh training run.','',
'## Validation and provenance','',
'26 existing automated tests and 3 added ReLU tests passed. Added checks cover known weak L2/H1 integrals of ReLU(x), a multilayer absolute-value network, inactive/active/crossing and identically-zero cases, and H2 rejection. Both checkpoints were checked against their exact coefficient records. For each model, 50 random point values/gradients were compared with PyTorch autograd. Every saved policy record passed exact geometry, volume, endpoint/root aggregation and terminal-policy checks. This supports the implementation but is not formal verification.','',
'Raw rational endpoints, histories, final partitions, training data, checkpoints, source files, environment and hashes are included beside this report. `sampled_norms.json` records the approximate grid results and `summary.json` records all final exact endpoints and costs. The paper PDF/TeX hashes were checked unchanged. No paper asset is updated. This report serves as the reproducibility registry for these separate experiments.','',
'Reproduce in a fresh output location by copying `sources/`, setting ROOT in run_relu.py as needed, and running it with the recorded Python environment; it reads the original saved tanh training files at BASE. Run diagnostics.py afterwards. Existing result folders are deliberately not overwritten.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
dump(root/'environment.json',{'python':sys.version,'platform':platform.platform(),'gmpy2':gmpy2.version(),'mpfr':gmpy2.mpfr_version(),'gmp':gmpy2.mp_version(),'command':'python -u sources/run_relu.py; python sources/diagnostics.py','source_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'sources').rglob('*') if p.is_file() and '__pycache__' not in str(p)}})
