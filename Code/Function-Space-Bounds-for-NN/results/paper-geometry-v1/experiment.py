"""Frozen-coefficient studies; never trains or overwrites completed records."""
from pathlib import Path
import json,hashlib,platform,sys,argparse
import gmpy2
from backend import run,Q
from test_study import cases,hinge

ROOT=Path(__file__).resolve().parent

def execute(out,name,record,domain,quantity,rule,policy,cap,tol):
 path=out/f'{name}-{quantity}-{rule}-{policy}-{cap}-{str(tol).replace("/","_")}.json'
 if path.exists():return
 result=run(record,domain,quantity,rule,policy,cap,tol,scan=name.startswith('control'))
 result.update(case=name,domain=domain,network=record,network_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest())
 path.write_text(json.dumps(result,indent=2)+'\n')
 f=result['final'];print(name,quantity,rule,policy,cap,result['status'],f['leaves'],float(Q(f['gap'])),round(result['seconds'],3),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--controls-only',action='store_true');args=p.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=True)
 if not args.controls_only:
  for arch in ['deep','shallow200']:
   for target,rad in [('two-bumps','1'),('disk','1.5')]:
    record=json.loads((args.frozen/'relu'/arch/target/'network.json').read_text())
    for quantity in ['L2','H1']:
     for cap in [256,1024]:
      for rule in ['range','hybrid']:
       for policy in ['uniform','maxgap']:
        execute(out,f'trained-{arch}-{target}',record,[['-'+rad,rad]]*2,quantity,rule,policy,cap,'1/100')
 for name,record in cases().items():
  for quantity,tol in [('L2','1/1000'),('H1','1/100')]:
   for rule in ['range','hybrid']:
    for policy in ['uniform','maxgap']:
     execute(out,'control-'+name,record,[['0','1']]*2,quantity,rule,policy,1024,tol)
 for a in ['1/7','1/3','1/2','511/1024','513/1024']:
  for quantity in ['L2','H1']:
   for policy in ['uniform','maxgap']:
    execute(out,'control-offset-'+a.replace('/','_'),hinge(a=a),[['0','1']]*2,quantity,'hybrid',policy,256,'0')
 for quantity in ['L2','H1']:
  for policy in ['uniform','maxgap']:
   execute(out,'control-axis-3d',hinge(3),[['0','1']]*3,quantity,'hybrid',policy,512,'0')
 env={'python':sys.version,'platform':platform.platform(),'gmpy2':gmpy2.version(),'gmp':gmpy2.mp_version(),'arithmetic':'exact rational throughout endpoints, geometry, signs, moments, sums and marking','timing':'single run, includes history recording; no universal ranking'}
 (out.parent/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
if __name__=='__main__':main()
