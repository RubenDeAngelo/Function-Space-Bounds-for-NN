"""Export exact endpoint tables and tolerance crossings from saved histories."""
import argparse
import csv
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as Q
import json
from pathlib import Path
from experiment_records import dump


def bound(q,upper=False):
    q=Q(q)
    with localcontext() as c:
        c.prec=4;c.rounding=ROUND_CEILING if upper else ROUND_FLOOR
        v=Decimal(q.numerator)/Decimal(q.denominator)
        assert (Q(v)>=q) if upper else (Q(v)<=q)
    if not v:return '0'
    if -3<=v.adjusted()<=3:return format(v,'f')
    mant,exponent=format(v,'.3E').split('E')
    return mant+r'\!\times\!10^{'+str(int(exponent))+'}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--figure1',type=Path,default=Path('results/two-bumps-deep-l2'));args=p.parse_args()
    folder=args.directory;table=folder/'tables';table.mkdir(exist_ok=True)
    records=[];summary=[];crossings=[]
    for spec in json.loads((folder/'experiments.json').read_text()):
        for policy in ('adaptive','uniform'):
            r=json.loads((folder/'runs'/spec['id']/f'{policy}.json').read_text());h=r['history'][-1]
            record={'id':spec['id'],'policy':policy,'status':r['status'],'leaves':h['leaves'],'evaluations':h['evaluations'],'seconds':h['seconds'],
                    **{prefix+'_'+k:h[key][k] for prefix,key in [('integral','integral'),('norm','norm')] for k in ('lower','upper')},'gap_upper':h['gap_upper']}
            records.append(record)
            summary.append({'id':spec['id'],'policy':policy,'gap':float(Q(h['gap_upper'])),'leaves':h['leaves'],'evaluations':h['evaluations']})
            for tau in ('1','0.1','0.01'):
                hit=next((x for x in r['history'] if Q(x['gap_upper'])<=Q(tau)),None)
                crossings.append({'id':spec['id'],'policy':policy,'tolerance':tau,'reached':hit is not None,
                                  'leaves':hit['leaves'] if hit else None,'evaluations':hit['evaluations'] if hit else None})
    with (table/'exact_endpoints.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    dump(table/'tolerance_crossings.json',crossings);dump(table/'summary.json',summary)
    rows=[]
    for name,path in [('Two bumps $L^2$',args.figure1),('Disk $H^1$',folder/'runs/disk-H1'),('Disk $H^2$',folder/'runs/disk-H2'),('PINN residual',folder/'runs/pinn-residual')]:
        for policy in ('adaptive','uniform'):
            r=json.loads((path/f'{policy}.json').read_text());h=r['history'][-1]
            interval=lambda key:'$['+bound(h[key]['lower'])+','+bound(h[key]['upper'],True)+']$'
            rows.append(f"{name} & {policy.capitalize()} & {h['leaves']:,} & {interval('integral')} & {interval('norm')} \\\\")
    (table/'terminal_bounds.tex').write_text('\n'.join(rows)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
