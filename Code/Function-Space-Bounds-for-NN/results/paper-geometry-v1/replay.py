"""Repeat each saved experiment and compare everything except elapsed time."""
from pathlib import Path
import json,argparse
from backend import run
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');args=p.parse_args()
n=0
for path in sorted((ROOT/'results').glob('*.json')):
 r=json.loads(path.read_text())
 if not args.all and r['case'].startswith('trained') and not (r['case']=='trained-deep-disk' and r['cap']==256):continue
 fresh=run(r['network'],r['domain'],r['quantity'],r['rule'],r['policy'],r['cap'],r['tolerance'],scan=r['case'].startswith('control'))
 assert {k:v for k,v in fresh.items() if k!='seconds'}=={k:r[k] for k in fresh if k!='seconds'},path.name
 n+=1
print('Exact replay passed',n,'runs excluding elapsed time.')
