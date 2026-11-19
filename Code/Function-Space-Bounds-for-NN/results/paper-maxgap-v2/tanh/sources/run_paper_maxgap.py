from pathlib import Path
import os,json,shutil
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'.cache'))
from fractions import Fraction as Q
from verified_arithmetic import Arithmetic
from verified_integration import Box,Integrator
from maxgap import MaxGapIntegrator
from verified_residual import poisson_residual_integrand
from exp_verified_studies import load_model
from exp_verified_two_bumps import load_saved_network
from experiment_records import dump,sha,validate_record,validate_policy
ROOT=Path(__file__).resolve().parent.parent
BASE=Path('/Users/moritz/Projekte/FunctionSpaceNormsOfNeuralNetworks_Research_Paper/Code/Function-Space-Bounds-for-NN/results')
SUITE=BASE/'validated-paper-suite-v1'
specs=json.loads((ROOT/'experiments.json').read_text());summary=[]
(ROOT/'models').mkdir();(ROOT/'runs').mkdir()
for spec in specs:
 a=Arithmetic(128);two=spec['group']=='two_bumps';src=BASE/'two-bumps-deep-l2' if two else SUITE/'models'/spec['model']
 dest=ROOT/'models'/spec['model']
 if not dest.exists():
  dest.mkdir()
  for p in src.iterdir():
   if p.is_file() and p.suffix in ('.pt','.npz','.json'):shutil.copy2(p,dest/p.name)
 net=load_saved_network(dest,a) if two else load_model(dest,a)
 original=BASE/'two-bumps-deep-l2' if two else SUITE/'runs'/spec['id']
 folder=ROOT/'runs'/spec['id'];folder.mkdir()
 manifest={**spec,'bits':128,'p':2,'theta':'7/20','tolerance':'0','network_sha256':sha(dest/'network.json'),'source_model':str(src),'source_run':str(original),'maxgap_rule':'exactly one maximum local integral-gap leaf; smallest creation id breaks ties; budget-only stopping','subdivision':'bisect every coordinate'}
 dump(folder/'manifest.json',manifest)
 oracle=(lambda b:poisson_residual_integrand(net,b)) if spec['kind']=='poisson_residual' else (lambda b:net.sobolev_integrand(b,order=spec['order']))
 shutil.copy2(original/'adaptive.json',folder/'dorfler.json')
 for policy in ['maxgap','uniform']:
  cls=MaxGapIntegrator if policy=='maxgap' else Integrator
  run=cls(a,oracle,Box(tuple(map(tuple,spec['domain']))))
  if policy=='maxgap':run.run_maxgap(spec['max_leaves'])
  else:run.run(policy='uniform',theta='7/20',tolerance='0',max_leaves=spec['max_leaves'])
  record=run.serialise();validate_record(record,spec['domain']);assert len(run.cells)==spec['max_leaves']
  if policy=='uniform':
   validate_policy(record,manifest,'uniform');old=json.loads((original/'uniform.json').read_text())
   strip=lambda r:{**r,'history':[{k:v for k,v in h.items() if k!='seconds'} for h in r['history']]}
   assert strip(record)==strip(old),'Uniform replay mismatch: '+spec['id']
  else:dump(folder/'maxgap_audit.json',{'nodes':run.node_records,'refinements':run.refinements})
  dump(folder/(policy+'.json'),record);h=record['history'][-1]
  summary.append({'id':spec['id'],'policy':policy,'status':record['status'],**h});dump(ROOT/'summary.json',summary)
  print(spec['id'],policy,'N',h['leaves'],'gap',float(Q(h['gap_upper'])),'seconds',h['seconds'],flush=True)
