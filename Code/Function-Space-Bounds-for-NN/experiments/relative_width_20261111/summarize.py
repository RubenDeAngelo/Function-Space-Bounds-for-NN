"""Validate exact recorded endpoints and emit manuscript tables; no experiments."""
from pathlib import Path
from fractions import Fraction as Q
import csv,json,hashlib,platform,subprocess,sys,importlib.util,os
HERE=Path(__file__).resolve().parent
LATEX=Path(os.environ.get('PAPER_LATEX_ROOT',HERE.parents[3]/'LaTeX'))
CODE=Path(os.environ.get('PAPER_CODE_ROOT',str(HERE.parents[1])))
spec=importlib.util.spec_from_file_location('assets',LATEX/'render_relu_submission_assets.py');assets=importlib.util.module_from_spec(spec);spec.loader.exec_module(assets)
models=json.loads((HERE/'models.json').read_text()); records={}; endpoints=[]; work=[]; widthrows=[]
for case in models:
 for quantity in ['L2','H1']:
  for policy in ['maxgap','uniform']:
   path=HERE/'runs'/f'{case}-{quantity}-{policy}.csv';meta=path.with_suffix('.csv.meta')
   assert meta.exists(),f'Incomplete: {path.name}'
   assert 'status relative_tolerance_reached' in meta.read_text(),path.name
   rows=list(csv.DictReader(path.open()));last=rows[-1]
   prevlo,prevhi=None,None
   for row in rows:
    lo,hi=Q(row['lower']),Q(row['upper']);n=int(row['leaves']);r=Q(0) if hi==0 else 2*(hi-lo)/(hi+lo)
    assert 0<=lo<=hi and Q(row['relative_width'])==r
    if prevlo is not None:assert prevlo<=lo<=hi<=prevhi
    prevlo,prevhi=lo,hi
    assert n%3==1 and int(row['evaluated_cells'])<=(4*n-1)//3
    if policy=='uniform':assert n&(n-1)==0 and (n.bit_length()-1)%2==0
   assert Q(last['relative_width'])<=Q(1,100),path.name
   assert len(rows)==1 or Q(rows[-2]['relative_width'])>Q(1,100),path.name
   # Earlier recorded levels exactly reproduce frozen 1024-leaf endpoints.
   matches=list((CODE/'results/paper-geometry-v1/results').glob(f'{case}-{quantity}-hybrid-{policy}-1024-*.json'))
   assert len(matches)==1
   old=json.loads(matches[0].read_text())
   oldrow=next(r for r in rows if int(r['leaves'])==1024)
   assert Q(oldrow['lower'])==Q(old['final']['lower']) and Q(oldrow['upper'])==Q(old['final']['upper'])
   assert models[case]['network']==old['network'] and models[case]['domain']==old['domain']
   records[case,quantity,policy]=last
   lo,hi=Q(last['lower']),Q(last['upper'])
   arch='D' if '-deep-' in case else 'S';target='Bumps' if 'two-bumps' in case else 'Disk';norm=r'$L^2$' if quantity=='L2' else r'$H^1$'
   # Relative widths are rounded upward, so displayed values also certify <=0.01.
   rel=assets.outward(Q(last['relative_width']),True,5)
   assert Q(last['relative_width'])<=Q(rel)<=Q(1,100)
   digits=9
   while True:
    pl,pu=(Q(assets.outward(v,upper,digits)) for v,upper in [(lo,False),(hi,True)])
    if 2*(pu-pl)/(pu+pl)<=min(Q(1,100),Q(rel)):break
    digits+=1;assert digits<=20
   endpoints.append([arch,target,norm,policy.title(),f"{int(last['leaves']):,}",*assets.enclosures(lo,hi,digits=digits),f'${rel}$'])
   work.append([arch,target,norm,policy.title(),f"{int(last['leaves']):,}",f"{int(last['evaluated_cells']):,}",f"{float(last['seconds']):.2f}"])
for arch,label in [('deep','D'),('shallow200','S')]:
 for target,name in [('two-bumps','Bumps'),('disk','Disk')]:
  for quantity,norm in [('L2',r'$L^2$'),('H1',r'$H^1$')]:
   case=f'trained-{arch}-{target}';a,u=(records[case,quantity,p] for p in ['maxgap','uniform'])
   widthrows.append([label,name,norm,f"{int(a['leaves']):,}",f"{int(u['leaves']):,}",
    f"${assets.outward(Q(a['relative_width']),True,5)}$",f"${assets.outward(Q(u['relative_width']),True,5)}$",
    f"{int(a['evaluated_cells']):,}",f"{int(u['evaluated_cells']):,}"])
assets.write_table(LATEX/'tables/relative_width_success.tex','lllrrrrrr',
 ['Arch.','Target','Norm',r'$N_{\rm a}$',r'$N_{\rm u}$',r'$R_{\rm a}$',r'$R_{\rm u}$',r'$E_{\rm a}$',r'$E_{\rm u}$'],widthrows)
assets.write_table(LATEX/'tables/relative_width_endpoints.tex','llllrrrr',
 ['Arch.','Target','Norm','Policy','Leaves','Squared norm','Norm',r'$R$'],endpoints,long=True)
assets.write_table(LATEX/'tables/relative_width_work.tex','llllrrr',
 ['Arch.','Target','Norm','Policy','Leaves','Evaluated cells','Seconds'],work,long=True)
# Compare original disk trajectory checkpoints without changing the source records.
old=list(csv.DictReader((CODE/'experiments/smoothed_disk_figure1/data/history.csv').open()));diskchecks=0
for p in ['maxgap','uniform']:
 new={int(r['leaves']):r for r in csv.DictReader((HERE/'runs'/f'trained-deep-disk-L2-{p}.csv').open())}
 for r in old:
  if r['policy']==p and int(r['leaves']) in new:
   t=new[int(r['leaves'])];assert Q(t['lower'])==Q(r['lower_exact']) and Q(t['upper'])==Q(r['upper_exact']);diskchecks+=1
summary={'run_count':len(records),'relative_full_width_definition':'2*(U-L)/(U+L), zero for [0,0]',
 'requested_relative_width':'1/100','all_reached':True,'largest_exact_relative_width':str(max(Q(r['relative_width']) for r in records.values())),
 'original_disk_checkpoint_matches':diskchecks,'kernel_sha256':hashlib.sha256((HERE/'kernel.cpp').read_bytes()).hexdigest(),
 'runs':[dict(case=k[0],quantity=k[1],policy=k[2],**r) for k,r in records.items()]}
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
env={'platform':platform.platform(),'python':sys.version,'compiler':subprocess.check_output(['clang++','--version'],text=True).splitlines()[0],
 'gmp_version':'6.3.0','gmp_version_header':'/opt/homebrew/include/gmp.h','concurrent_jobs':4,
 'timing_scope':'Integration only, excluding model parsing and process startup; single-threaded jobs, up to four concurrent; descriptive elapsed times, not isolated benchmarks.',
 'base_code_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()}
(HERE/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*')) if p.is_file() and p.name not in ['SHA256.json','kernel'] and '__pycache__' not in p.parts}
(HERE/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
