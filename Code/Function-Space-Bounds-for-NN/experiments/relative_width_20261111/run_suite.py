from pathlib import Path
import subprocess,json,concurrent.futures,hashlib,platform,sys
HERE=Path(__file__).resolve().parent
(HERE/'runs').mkdir(exist_ok=True)
models=json.loads((HERE/'models.json').read_text())
tasks=[(case,q,p) for case in models for q in ['H1','L2'] for p in ['maxgap','uniform']]
def run(t):
 case,q,p=t;out=HERE/'runs'/f'{case}-{q}-{p}.csv'
 if out.exists() and out.with_suffix('.csv.meta').exists():return 'retained '+out.name
 cap=4194304 if p=='maxgap' else 268435456
 with (out.with_suffix('.log')).open('w') as log:
  proc=subprocess.run([str(HERE/'kernel'),str(HERE/'models'/f'{case}.txt'),q,p,str(cap),str(out),'tolerance'],stderr=log,stdout=log)
 if proc.returncode:raise RuntimeError(out.name)
 return out.name+' '+out.with_suffix('.csv.meta').read_text().replace('\n','; ')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for future in concurrent.futures.as_completed([pool.submit(run,t) for t in tasks]):print(future.result(),flush=True)
