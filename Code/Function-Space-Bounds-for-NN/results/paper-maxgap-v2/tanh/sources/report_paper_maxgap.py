from pathlib import Path
import os,json,sys,platform,csv
from fractions import Fraction as Q
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'.cache'))
import gmpy2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from experiment_records import dump,sha
root=Path(__file__).resolve().parent.parent
specs=json.loads((root/'experiments.json').read_text());rows=json.loads((root/'summary.json').read_text())
styles={'maxgap':('#0072B2','-','One maximum-gap box'),'uniform':('#D55E00','-','Uniform'),'dorfler':('#009E73','--','Paper Dörfler')}
def history_plot(ax,spec,title):
 for policy,(color,style,label) in styles.items():
  r=json.loads((root/'runs'/spec['id']/(policy+'.json')).read_text())
  ax.loglog([h['leaves'] for h in r['history']],[float(Q(h['gap_upper'])) for h in r['history']],color=color,linestyle=style,label=label)
 ax.set(title=title,xlabel='Leaf boxes',ylabel='Integral gap U − L');ax.title.set_fontsize(11);ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10);ax.tick_params(labelsize=9);ax.grid(alpha=.25);ax.legend(fontsize=7)
for order in [1,2]:
 fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
 for i,state in enumerate(['untrained','trained']):
  for seed in range(3):
   spec=next(s for s in specs if s['id']==f'one-d-seed{seed}-{state}-H{order}')
   history_plot(axes[i,seed],spec,f'H{order}, {state}, seed {seed}')
 fig.savefig(root/f'one_d_H{order}.png',dpi=180,bbox_inches='tight');plt.close(fig)
two_d=[s for s in specs if len(s['domain'])==2]
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,spec in zip(axes.flat,two_d):history_plot(ax,spec,spec['id'])
fig.savefig(root/'two_d_convergence.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(4,3,figsize=(12,15),layout='constrained')
for i,spec in enumerate(two_d):
 records={p:json.loads((root/'runs'/spec['id']/(p+'.json')).read_text()) for p in styles}
 positive=[float(Q(c['indicator'])) for r in records.values() for c in r['cells'] if Q(c['indicator'])>0];scale=LogNorm(min(positive),max(positive))
 for j,(policy,record) in enumerate(records.items()):
  ax=axes[i,j];patches=[];values=[]
  for cell in record['cells']:
   (x0,x1),(y0,y1)=[[float(Q(v)) for v in pair] for pair in cell['bounds']]
   patches.append(Rectangle((x0,y0),x1-x0,y1-y0));values.append(float(Q(cell['indicator'])))
  pc=PatchCollection(patches,cmap='viridis',norm=scale,edgecolor='white',linewidth=.1);pc.set_array(np.ma.masked_less_equal(values,0));ax.add_collection(pc)
  ax.set(xlim=spec['domain'][0],ylim=spec['domain'][1],title=f"{spec['id']}: {policy}, N={len(record['cells'])}",xlabel='x',ylabel='y');ax.set_aspect('equal');ax.title.set_fontsize(9);ax.tick_params(labelsize=8)
 fig.colorbar(pc,ax=list(axes[i]),shrink=.8,label='Local integral-gap contribution')
fig.savefig(root/'partitions.png',dpi=160,bbox_inches='tight');plt.close(fig)
lines=['# Current paper: maximum-gap marking reruns','','## Scope and method','','All 16 cases used in the current compiled manuscript are included: the two-bump L2 case; H1 and H2 for paired untrained/trained 1D networks, seeds 0, 1 and 2; disk H1 and H2; and the PINN squared interior residual. These are the saved tanh networks in the active paper. Archived experiments and the separate ReLU runs from this chat are outside this scope. No network is retrained. The saved checkpoint coefficients, training artifacts and configurations are retained. The paper directory is unchanged.','','The sole numerical-method change is marking: choose exactly one current leaf with maximum local integral endpoint gap U_K−L_K. Comparisons use exact rational indicators, and the smallest creation id breaks ties. Split the selected interval into two children in 1D or box into four children in 2D, halving every coordinate. Use ancestor intersections and the paper source snapshot for the enclosure oracle, 128-bit directed MPFR arithmetic, exact rational geometry, and the saved coefficients interpreted as exact real numbers. All maxgap runs continue to the specified leaf budget; no early stagnation heuristic is applied. Positive stopping tolerances are disabled.','','The final leaf counts are exactly matched between maximum-gap and uniform policies: 256 in 1D, 4096 for two bumps, and 1024 for disk and PINN. Corresponding enclosure evaluations are 511, 5461, and 1365. Uniform integration was rerun in all 16 cases and every record was checked to reproduce the saved paper uniform run exactly, excluding timings. The original Dörfler records are also included, but their final leaf counts can be lower.','','## Equal-size final integral gaps','','Gaps are full U−L for the integral defining the squared norm, or the squared interior residual. Positive reduction means maxgap is smaller. Negative reduction means maxgap is larger. These are enclosure gaps, not the norms, network fitting errors or PDE solution errors.','','| Experiment | Leaves (both) | Maximum-gap U−L | Uniform U−L | Reduction vs uniform | Paper Dörfler U−L (leaves) |','|---|---:|---:|---:|---:|---:|']
comparisons=[]
for spec in specs:
 rr=[r for r in rows if r['id']==spec['id']];a=next(r for r in rr if r['policy']=='maxgap');u=next(r for r in rr if r['policy']=='uniform');d=json.loads((root/'runs'/spec['id']/'dorfler.json').read_text())['history'][-1]
 ga,gu=Q(a['gap_upper']),Q(u['gap_upper']);reduction=100*(1-float(ga/gu))
 assert a['leaves']==u['leaves'] and a['evaluations']==u['evaluations']
 lines.append(f"| {spec['id']} | {a['leaves']} | {float(ga):.10g} | {float(gu):.10g} | {reduction:.2f}% | {float(Q(d['gap_upper'])):.10g} ({d['leaves']}) |")
 comparisons.append({'id':spec['id'],'leaves':a['leaves'],'maxgap_gap':str(ga),'uniform_gap':str(gu),'reduction_percent':reduction,'dorfler_gap':d['gap_upper'],'dorfler_leaves':d['leaves']})
dump(root/'comparisons.json',comparisons)
lines+=['','## Integral and rooted-norm enclosures','','The following endpoints are rounded outward to six decimal places. PINN rooted endpoints are the interior residual L2 norm, not a solution-error bound. H2 follows the paper multi-index convention with each mixed derivative counted once. Norm/root gaps differ from integral gaps.','', '| Experiment | Policy | Integral [L,U] | Rooted norm interval | Evaluations | Seconds |','|---|---|---|---|---:|---:|']
def endpoints(record):
 lo,hi=Q(record['lower']),Q(record['upper']);scale=10**6
 return f'[{(lo*scale).__floor__()/scale:.6f}, {(hi*scale).__ceil__()/scale:.6f}]'
for r in rows:lines.append(f"| {r['id']} | {r['policy']} | {endpoints(r['integral'])} | {endpoints(r['norm'])} | {r['evaluations']} | {r['seconds']:.2f} |")
lines+=['','All 32 new policy runs stopped with leaf_budget_exhausted. Timings are single-run observations excluding training, rendering and the independent audit. Maxgap and uniform have different history-recording and marking bookkeeping; timings are not controlled speed benchmarks. Maxgap saves selected intermediate histories and every individual split in maxgap_audit.json.','','## Validation and reproducibility','','The suite source-snapshot tests and three maxgap tests passed. New tests cover exact maximum selection, selecting exactly one box under ties, stable tie-breaking, and 1D/2D leaf/evaluation counts. An independent linear scan (without the heap used for selection) verifies every individual marking decision. Audits check every subdivision, local integral enclosure/volume, ancestor inclusion, final partition and saved intermediate endpoint aggregation. The standard exact-record checks verify global geometry, volume, endpoints and roots. All 16 uniform records replay exactly apart from timings, and checkpoint/coefficient consistency is checked before integration. These checks support the implementation, not formal verification. See validation.json, tests.log, audit.log and execution.log.','','All pre-existing paper-directory files (excluding runtime caches, virtual environment and Git internals) were hashed before execution and checked unchanged afterwards. No TeX, PDF, figure, source code or reproducibility registry in that directory is edited. This standalone report serves as the new result registry.','','Reproduce with the recorded environment by copying sources into a fresh result directory, retaining experiments.json and the original source result directories recorded in each manifest, then running sources/run_paper_maxgap.py, sources/audit_paper_maxgap.py and sources/report_paper_maxgap.py. Existing output directories are deliberately not overwritten. Source hashes, exact rational endpoints, full partitions and selection audit trails are included. The original paper source snapshot supplies verified_arithmetic.py, verified_network.py, verified_integration.py and verified_residual.py unchanged; the record validator is the newer repository version.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
with (root/'endpoints.csv').open('w') as f:
 fields=['id','policy','status','leaves','evaluations','seconds','integral_lower','integral_upper','norm_lower','norm_upper','gap_upper'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for r in rows:writer.writerow({**{k:r[k] for k in fields if k in r},'integral_lower':r['integral']['lower'],'integral_upper':r['integral']['upper'],'norm_lower':r['norm']['lower'],'norm_upper':r['norm']['upper']})
dump(root/'environment.json',{'python':sys.version,'platform':platform.platform(),'gmpy2':gmpy2.version(),'mpfr':gmpy2.mpfr_version(),'gmp':gmpy2.mp_version(),'source_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'sources').rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.cache' not in p.parts},'commands':['python -u sources/run_paper_maxgap.py','python sources/audit_paper_maxgap.py','python sources/report_paper_maxgap.py']})
