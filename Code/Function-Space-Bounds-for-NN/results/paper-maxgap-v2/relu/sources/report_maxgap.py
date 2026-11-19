from pathlib import Path
import json,sys,platform,os,hashlib
from fractions import Fraction as Q
import gmpy2
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_records import dump,sha
root=Path(__file__).resolve().parent.parent
rows=json.loads((root/'summary.json').read_text())
lines=['# ReLU experiments: one maximum-gap leaf per refinement','','All four trained ReLU checkpoints are reused without retraining. Architectures: deep [2,32,32,32,1], shallow [2,200,1]. Both two-bump and disk experiments are rerun for L2 and full H1. These are bounds on the network norms, not fitting errors. The saved checkpoints, coefficients and training records are included. Norm estimates and fit diagnostics remain those in the previous experiments because the networks are unchanged.','','## Refinement rule','','At each step select exactly one current leaf maximizing its local integral upper endpoint minus lower endpoint, with exact rational comparison. Break ties by smallest creation id. Subdivide that leaf into four equal dyadic children, bisecting both coordinates. Child integrand bounds are intersected with their parent bounds. Use the same 128-bit directed MPFR enclosure oracle, domains and coefficients as before. Each step adds three leaves; both policies finish at exactly 4096 leaves for two bumps and 1024 for disk. The new maximum-gap policy runs to this budget without an early stagnation heuristic. No positive tolerance is requested.','','Uniform integration was rerun for all eight cases and its complete records, excluding wall times, were asserted identical to the previous saved uniform records. Thus all changes in gaps relative to the old adaptive runs result from the refinement strategy, not the networks or oracle.','','## Equal-size final gap comparison','','Gaps below are U−L for the squared norm integral. Positive reduction means maximum-gap marking is better.','','| Architecture | Experiment | Norm | Leaves (both) | Max-gap U−L | Uniform U−L | Reduction vs uniform |','|---|---|---|---:|---:|---:|---:|']
for arch,source in [('deep',root.parent/'relu-experiments'),('shallow200',root.parent/'relu-shallow200-experiments')]:
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for i,name in enumerate(['two-bumps','disk']):
  for j,norm in enumerate(['L2','H1']):
   selected=[r for r in rows if r['architecture']==arch and r['experiment']==name and r['norm_name']==norm];a=next(r for r in selected if r['policy']=='maxgap');u=next(r for r in selected if r['policy']=='uniform')
   ga,gu=Q(a['gap_upper']),Q(u['gap_upper']);reduction=100*(1-float(ga/gu))
   assert a['leaves']==u['leaves'] and a['evaluations']==u['evaluations']
   lines.append(f"| {arch} | {name} | {norm} | {a['leaves']} | {float(ga):.9g} | {float(gu):.9g} | {reduction:.2f}% |")
   ax=axes[i,j]
   for policy,label in [('maxgap','One maximum-gap leaf'),('uniform','Uniform')]:
    r=json.loads((root/arch/name/norm/(policy+'.json')).read_text());ax.loglog([h['leaves'] for h in r['history']],[float(Q(h['gap_upper'])) for h in r['history']],label=label)
   d=json.loads((source/name/norm/'adaptive.json').read_text());ax.loglog([h['leaves'] for h in d['history']],[float(Q(h['gap_upper'])) for h in d['history']],'--',label='Previous Dörfler')
   ax.set(title=f'{arch}: {name}, {norm}',xlabel='Leaf boxes',ylabel='Squared-norm integral gap U − L');ax.grid(alpha=.25);ax.legend(fontsize=7,loc='upper right');ax.title.set_fontsize(11);ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10);ax.tick_params(labelsize=9)
 fig.savefig(root/(arch+'_convergence.png'),dpi=180,bbox_inches='tight');plt.close(fig)
lines+=['','## Final norm enclosures and cost','','Endpoints are displayed with outward rounding to six decimal places. Both policies have equal leaf counts and enclosure-evaluation counts in each case. Timings are single-run observations; different history-recording schedules and marking bookkeeping affect wall times, so these are not controlled speed benchmarks. Maximum-gap histories are saved every 32 splits, at uniform levels and at the final partition; every individual split is retained in maxgap_audit.json.','','| Architecture | Experiment | Norm | Policy | Norm enclosure | Evaluations | Seconds |','|---|---|---|---|---|---:|---:|']
for r in rows:
 lo=Q(r['norm']['lower']);hi=Q(r['norm']['upper']);scale=10**6
 lower=(lo*scale).__floor__()/scale;upper=(hi*scale).__ceil__()/scale
 lines.append(f"| {r['architecture']} | {r['experiment']} | {r['norm_name']} | {r['policy']} | [{lower:.6f}, {upper:.6f}] | {r['evaluations']} | {r['seconds']:.2f} |")
lines+=['','All sixteen new policy runs ended with leaf_budget_exhausted. Comparisons with previous Dörfler runs use the same maximum budget but the previous Dörfler partitions can have fewer leaves. These are eight fixed-network cases, not a population study.','','## Checks and provenance','','The new maximum-gap tests check that exactly one box is selected, stable tie-breaking and correct work counts, and selection of the actual largest gap. Three existing ReLU tests also passed. Every selection in all eight maximum-gap runs was independently checked by a linear scan of exact leaf indicators, without using the heap employed by the runner. The audit also checks every subdivision, ancestor inclusion, local volume aggregation and final active partition. The standard exact-record checks cover global geometry, endpoint aggregation and rooted endpoints for all sixteen records. Uniform records replay exactly apart from time. Checkpoint checks ensure the portable coefficients match the saved PyTorch state. See validation.json and execution.log.','','All paper TeX/PDF files and all previous ReLU result files were checked unchanged. This separate report is the reproducibility registry for the new results. Source files and environment hashes are included; run sources/run_maxgap.py in a fresh result location with the recorded environment and the previous result folders available. Then run sources/audit_maxgap.py and sources/report_maxgap.py.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
dump(root/'environment.json',{'python':sys.version,'platform':platform.platform(),'gmpy2':gmpy2.version(),'mpfr':gmpy2.mpfr_version(),'gmp':gmpy2.mp_version(),'command':'python -u sources/run_maxgap.py; python sources/audit_maxgap.py; python sources/report_maxgap.py','source_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'sources').rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts}})
