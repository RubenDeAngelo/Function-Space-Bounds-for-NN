from pathlib import Path
import json,shutil,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parent/'.cache'))
from fractions import Fraction as Q
from verified_arithmetic import Arithmetic
from verified_network import VerifiedNetwork
from verified_integration import Box,Integrator
from maxgap import MaxGapIntegrator
from exp_verified_two_bumps import load_saved_network
from experiment_records import dump,sha,validate_record,validate_policy
ROOT=Path(__file__).resolve().parent.parent
summary=[]
for arch,source in [('deep',ROOT.parent/'relu-experiments'),('shallow200',ROOT.parent/'relu-shallow200-experiments')]:
 for name,domain,budget in [('two-bumps',[[-1,1],[-1,1]],4096),('disk',[[-1.5,1.5],[-1.5,1.5]],1024)]:
  a=Arithmetic(128);net=load_saved_network(source/name,a)
  modeldir=ROOT/arch/name;modeldir.mkdir(parents=True)
  for file in ['checkpoint.pt','network.json','training.json','training_data.npz','fit_grid.npz']:shutil.copy2(source/name/file,modeldir/file)
  for norm,order in [('L2',0),('H1',1)]:
   folder=modeldir/norm;folder.mkdir()
   manifest={'domain':domain,'bits':128,'p':2,'order':order,'norm':norm,'max_leaves':budget,'tolerance':'0','theta':'7/20','maxgap_rule':'exactly one leaf maximizing integral.upper-integral.lower; smallest creation id breaks ties','subdivision':'four dyadic children; all coordinates bisected','network_sha256':sha(modeldir/'network.json'),'source_model':str(source/name),'stagnation':'maxgap runs to leaf budget; no heuristic stagnation stop'}
   dump(folder/'manifest.json',manifest)
   for policy in ['maxgap','uniform']:
    cls=MaxGapIntegrator if policy=='maxgap' else Integrator
    run=cls(a,lambda b:net.sobolev_integrand(b,order=order),Box(tuple(map(tuple,domain))))
    if policy=='maxgap':run.run_maxgap(budget)
    else:run.run(policy='uniform',theta='7/20',max_leaves=budget,tolerance='0')
    record=run.serialise();validate_record(record,domain);assert len(run.cells)==budget
    if policy=='uniform':
     validate_policy(record,manifest,'uniform')
     old=json.loads((source/name/norm/'uniform.json').read_text())
     strip=lambda r:{**r,'history':[{k:v for k,v in h.items() if k!='seconds'} for h in r['history']]}
     assert strip(record)==strip(old),'Uniform replay differs'
    else:dump(folder/'maxgap_audit.json',{'nodes':run.node_records,'refinements':run.refinements})
    dump(folder/(policy+'.json'),record);last=record['history'][-1]
    summary.append({'architecture':arch,'experiment':name,'norm_name':norm,'policy':policy,'status':record['status'],**last})
    dump(ROOT/'summary.json',summary)
    print(arch,name,norm,policy,'N',last['leaves'],'gap',float(Q(last['gap_upper'])),'seconds',last['seconds'],flush=True)
