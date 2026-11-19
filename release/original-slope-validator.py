"""Check guide-to-theorem mapping, unchanged source data, and exact table values."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json,math
HERE=Path(__file__).resolve().parent;C=HERE.parents[1];P=C.parents[1]
m=json.loads((HERE/'validation/plot_metadata.json').read_text())
expected={
 'large-disk-L2':{'maxgap':2,'uniform':1},
 'hinge-2D-L2':{'maxgap':2,'uniform':1},
 'hinge-2D-H1':{'maxgap':1,'uniform':.5},
 'disk-deep-L2':{'maxgap':2,'uniform':1},
 'disk-shallow200-L2':{'maxgap':2,'uniform':1},
 'disk-deep-H1':{'maxgap':1,'uniform':.5},
 'disk-shallow200-H1':{'maxgap':1,'uniform':.5},
 'control-hinge-third-maxgap':{'maxgap':('exponential',1)},
 'control-three-hinges-maxgap':{'maxgap':('exponential',3)},
 'control-hinge-third-uniform':{'uniform':1},
 'control-three-hinges-uniform':{'uniform':1},
 'tanh-poisson':{'both':.5},'tanh-cdr':{'both':.5},
}
assert {p['panel'] for p in m['panels']}==set(expected)
for panel,policies in expected.items():
 guides=[g for g in m['guides'] if g['panel']==panel];assert len(guides)==len(policies)
 for g in guides:
  wanted=policies[g['policy']];v=g['vertices'];x,y=v[0];xx,yy=v[2]
  if isinstance(wanted,tuple):
   assert g['kind']=='exponential' and g['m']==wanted[1] and g['theorem']=='thm:spline_separation'
   assert math.isclose(math.log2(yy/y)/(xx-x),-1/wanted[1],abs_tol=1e-12)
  else:
   assert g['kind']=='power' and g['alpha']==wanted
   assert math.isclose(math.log(yy/y)/math.log(xx/x),-wanted,abs_tol=1e-12)
   target=('thm:spline_separation' if panel.startswith('control-') else 'thm: dyadic rate; eq: uniform rate' if panel.startswith('tanh-') else 'thm:deep_relu_separation')
   assert g['theorem']==target
for path,digest in m['input_sha256'].items():assert hashlib.sha256((P/path).read_bytes()).hexdigest()==digest,path
# Preserve the entire frozen PDE package, including model coefficients and all exact histories.
E=C/'experiments/pde_revision_20261112';old=json.loads((E/'manifest.json').read_text())
for path,digest in old['files_sha256'].items():assert hashlib.sha256((E/path).read_bytes()).hexdigest()==digest,path
records=json.loads((C/'results/paper-originality-v1/results/records.json').read_text());hats=[r for r in records if r['method']=='hybrid' and r['precision_bits']==128 and r['case'].startswith('normalized-hat-128')]
assert len(hats)==4
for r in hats:
 f=r['final'];assert Q(f['lower'])<=Q(1,3)<=Q(f['upper']) and Q(f['gap'])==Q(f['upper'])-Q(f['lower'])
 if r['policy']=='maxgap':assert f['leaves']==16 and f['counts']['cell_calls']==31 and Q(f['lower'])==Q(f['upper'])==Q(1,3)
 else:assert f['leaves']==128 and f['counts']['cell_calls']==255
result={'status':'passed','convergence_panels':13,'verified_triangles':20,'hashed_input_files':len(m['input_sha256']),'unchanged_frozen_pde_files':len(old['files_sha256']),'hat_table_records':4,'checks':['prescribed theorem rate per policy and quantity','independently recomputed triangle slopes','source data and complete frozen PDE package hashes','exact hat table work counts and integral inclusion']}
(HERE/'validation/result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
