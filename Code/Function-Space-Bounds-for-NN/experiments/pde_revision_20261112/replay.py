"""Replay residual certificates into a temporary directory; preserve saved evidence."""
from pathlib import Path
import argparse,csv,json,subprocess,tempfile
E=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
cap=65536 if args.full else 1024
checks=[]
with tempfile.TemporaryDirectory(dir=E/'validation',prefix='replay-') as tmp:
 for case,model in [('poisson','poisson_model'),('cdr','cdr_model')]:
  for policy in ['maxgap','uniform']:
   out=Path(tmp)/f'{case}-{policy}.csv'
   subprocess.run([str(E/'residual_kernel'),str(E/model/'network.txt'),case,policy,str(cap),str(out),'1/10000'],check=True)
   fresh=list(csv.DictReader(out.open()));saved={r['leaves']:r for r in csv.DictReader((E/f'runs/{case}-{policy}.csv').open())}
   for row in fresh:
    old=saved[row['leaves']]
    for key in ['lower','upper','gap','evaluations']:assert row[key]==old[key],(case,policy,key,row['leaves'])
   checks.append({'case':case,'policy':policy,'leaf_cap':cap,'matching_records':len(fresh)})
(E/'validation'/('replay-full.json' if args.full else 'replay-1024.json')).write_text(json.dumps(checks,indent=2)+'\n')
print('All exact endpoint and work-count comparisons passed.')
