"""Compare saved certificates while rejecting unintended experimental changes."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from experiment_records import dump, sha


def settings(m):
    order=m.get('order',m.get('derivative_order'))
    if order is None and m.get('norm') in ('L2','H1','H2'):
        order={'L2':0,'H1':1,'H2':2}[m['norm']]
    if order is None and m.get('kind')!='poisson_residual':
        raise ValueError('Manifest lacks explicit integrand order; do not infer it from a filename')
    return {'network':m['network_sha256'],'domain':m['domain'],
            'integrand':{'kind':m.get('kind','sobolev'),'order':order,'p':m.get('p',2)},
            'bits':m.get('bits',m.get('precision_bits')),'theta':str(Fraction(m['theta'])),
            'max_leaves':m['max_leaves'],'tolerance':str(Fraction(m.get('tolerance','0'))),
            'enclosure':m.get('enclosure_method','natural interval AD with ancestor intersections')}


def compare_settings(left,right,vary=(),different_network=False):
    differences={k:[left[k],right[k]] for k in left if left[k]!=right[k]}
    allowed=set(vary)|({'network'} if different_network else set())
    if set(differences)-allowed:
        raise ValueError(f'Unintended experiment differences: {set(differences)-allowed}')
    return differences


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--vary',nargs='*',choices=['theta','bits','max_leaves','tolerance','enclosure'],default=[])
    p.add_argument('--different-network',action='store_true',help='Explicitly permit different represented functions')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    manifests=[json.loads((d/'manifest.json').read_text()) for d in (args.left,args.right)]
    differences=compare_settings(*(settings(m) for m in manifests),args.vary,args.different_network)
    summary={'differences':differences,'different_functions':bool('network' in differences),
             'input_manifest_sha256':[sha(d/'manifest.json') for d in (args.left,args.right)],'runs':{}}
    for name,folder in [('left',args.left),('right',args.right)]:
        summary['runs'][name]={}
        for policy in ('adaptive','uniform'):
            r=json.loads((folder/f'{policy}.json').read_text())
            summary['runs'][name][policy]={'status':r['status'],'final':r['history'][-1]}
    args.output.mkdir(parents=True,exist_ok=False);dump(args.output/'comparison.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
