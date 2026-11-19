from pathlib import Path
from fractions import Fraction as Q
import json,subprocess,csv,hashlib,concurrent.futures,sys,os
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('PAPER_CODE_ROOT',str(HERE.parents[1])))

def export(record,domain,path):
 layers=[]
 for ww,bb in zip(record['weights'],record['biases']):
  w=[[Q(v) for v in row] for row in ww];b=[Q(v) for v in bb];den=max(x.denominator for row in w for x in row)
  den=max(den,max(x.denominator for x in b))
  assert den&(den-1)==0 and all(den%x.denominator==0 for row in w for x in row)
  layers.append((w,b,den))
 rad=Q(domain[0][1])
 assert Q(domain[0][0])==-rad and all(list(d)==list(domain[0]) for d in domain)
 lines=[f'{rad.numerator} {rad.denominator} {len(layers)}']
 for w,b,den in layers:
  lines.append(f'{len(w)} {len(w[0])} {den}')
  lines.extend(' '.join(str(int(x*den)) for x in row+[bias]) for row,bias in zip(w,b))
 path.write_text('\n'.join(lines)+'\n')

records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/paper-geometry-v1/results').glob('trained-*-hybrid-*-1024-*.json'))]
assert len(records)==16,len(records)
(HERE/'models').mkdir(exist_ok=True);(HERE/'validation').mkdir(exist_ok=True)
models={}
for r in records:
 model=HERE/'models'/f"{r['case']}.txt"
 export(r['network'],r['domain'],model)
 models[r['case']]={'network':r['network'],'domain':r['domain'],'network_sha256':r['network_sha256']}
(HERE/'models.json').write_text(json.dumps(models,indent=2)+'\n')

def check(r):
 out=HERE/'validation'/f"{r['case']}-{r['quantity']}-{r['policy']}.csv"
 res=subprocess.run([str(HERE/'kernel'),str(HERE/'models'/f"{r['case']}.txt"),r['quantity'],r['policy'],'1024',str(out),'fixed'],capture_output=True,text=True)
 assert res.returncode==0,res.stderr
 rows=list(csv.DictReader(out.open()));f=rows[-1];old=r['final']
 assert int(f['leaves'])==old['leaves']
 assert Q(f['lower'])==Q(old['lower']),(r['case'],r['quantity'],r['policy'],'lower')
 assert Q(f['upper'])==Q(old['upper']),(r['case'],r['quantity'],r['policy'],'upper')
 return f"{r['case']} {r['quantity']} {r['policy']} matched; {f['seconds']} s"
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for answer in pool.map(check,records):print(answer,flush=True)
print('All 16 trained terminal certificates exactly match frozen rational backend.')
