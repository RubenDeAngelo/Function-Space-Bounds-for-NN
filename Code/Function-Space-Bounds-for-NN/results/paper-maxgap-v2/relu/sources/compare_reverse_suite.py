"""Compare all frozen-network forward/reverse certificates with exact endpoints."""
from pathlib import Path
import sys,json,csv,argparse
from fractions import Fraction as Q

from experiment_records import sha, dump, validate_record, validate_policy
from compare_verified_runs import settings,compare_settings
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--forward-results',type=Path,default=Path('results'))
parser.add_argument('--reverse-results',type=Path,default=Path('results'))
parser.add_argument('--output',type=Path,default=Path('results/reverse-comparison'))
args=parser.parse_args()
old=args.forward_results
new=args.reverse_results
rows=[]
def add(name,policy,left,right):
    l=json.loads(left.read_text());r=json.loads(right.read_text())
    a,b=l['history'][-1],r['history'][-1]
    if policy=='uniform':
        geometry=lambda rec: sorted(tuple(tuple(Q(x) for x in pair) for pair in c['bounds']) for c in rec['cells'])
        assert geometry(l)==geometry(r)
        assert [h['leaves'] for h in l['history']]==[h['leaves'] for h in r['history']]
    endpoints=lambda h,k:[float(Q(h[k][x])) for x in ('lower','upper')]
    same=lambda x,y: json.dumps(x,sort_keys=True)==json.dumps(y,sort_keys=True)
    def no_time(obj):
        if isinstance(obj,dict):return {k:no_time(v) for k,v in obj.items() if k!='seconds'}
        if isinstance(obj,list):return [no_time(v) for v in obj]
        return obj
    row=dict(study=name,policy=policy,
      old_leaves=a['leaves'],new_leaves=b['leaves'],old_gap=float(Q(a['gap_upper'])),new_gap=float(Q(b['gap_upper'])),
      gap_ratio=float(Q(b['gap_upper'])/Q(a['gap_upper'])),
      old_integral=str(endpoints(a,'integral')),new_integral=str(endpoints(b,'integral')),
      old_norm=str(endpoints(a,'norm')),new_norm=str(endpoints(b,'norm')),
      exact_integral_unchanged=same(a['integral'],b['integral']),
      entire_record_unchanged_except_time=same(no_time(l),no_time(r)),
      new_evaluations=b['evaluations'],new_seconds=b['seconds'],new_status=r['status'])
    rows.append(row)
for spec in json.loads((new/'validated-paper-reverse-v1/experiments.json').read_text()):
    left=old/'validated-paper-suite-v1/runs'/spec['id'];right=new/'validated-paper-reverse-v1/runs'/spec['id']
    lm,rm=[json.loads((p/'manifest.json').read_text()) for p in (left,right)]
    assert set(compare_settings(settings(lm),settings(rm),vary=('enclosure',)))=={'enclosure'}
    assert sha(old/'validated-paper-suite-v1/models'/spec['model']/'network.json')==sha(new/'validated-paper-reverse-v1/models'/spec['model']/'network.json')
    for policy in ('adaptive','uniform'):
        rr=json.loads((right/f'{policy}.json').read_text());validate_record(rr,spec['domain']);validate_policy(rr,rm,policy)
        add(spec['id'],policy,left/f'{policy}.json',right/f'{policy}.json')
left=old/'two-bumps-deep-l2';right=new/'two-bumps-reverse-v1'
lm,rm=[json.loads((p/'manifest.json').read_text()) for p in (left,right)]
assert not compare_settings(settings(lm),settings(rm))
assert sha(left/'network.json')==sha(right/'network.json')
for policy in ('adaptive','uniform'):
    record=json.loads((right/f'{policy}.json').read_text())
    validate_record(record,lm['domain']);validate_policy(record,rm,policy)
    add('two-bumps-L2',policy,old/f'two-bumps-deep-l2/{policy}.json',new/f'two-bumps-reverse-v1/{policy}.json')
out=args.output;out.mkdir(parents=True,exist_ok=False)
dump(out/'comparison.json',rows)
with (out/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump(out/'comparison_provenance.json',{'comparison_source_sha256':sha(__file__),'forward_results':str(old.resolve()),'reverse_results':str(new.resolve()),'permitted_change':'enclosure method only; two-bump value-only settings unchanged','rows':len(rows),'input_sha256':{label:{str(p.relative_to(base)):sha(p) for family in families for p in sorted((base/family).rglob('*.json')) if 'sources' not in p.parts} for label,base,families in [('forward',old,['validated-paper-suite-v1','two-bumps-deep-l2']),('reverse',new,['validated-paper-reverse-v1','two-bumps-reverse-v1'])]}})
print('Compared',len(rows),'policy records')
