"""Portable frozen-coefficient replay; compare all saved fields except time.

python replay.py --output /new/path [--case two-bumps-L2]
No training, network mutation, or overwriting of a result directory.
"""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--case', action='append')
parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
args = parser.parse_args()
load = lambda p: json.loads(p.read_text())
specs = []
for s in load(root/'tanh'/'experiments.json'):
    specs.append(dict(s, suite='tanh', run=str(root/'tanh'/'runs'/s['id']),
                      network=str(root/'tanh'/'models'/s['model']/'network.json')))
for arch in ['deep','shallow200']:
    for target in ['two-bumps','disk']:
        for norm in ['L2','H1']:
            run = root/'relu'/arch/target/norm
            m = load(run/'manifest.json')
            specs.append(dict(m, id=f'relu-{arch}-{target}-{norm}', suite='relu',
                              kind='sobolev', run=str(run), network=str(run.parent/'network.json')))
selected = [s for s in specs if not args.case or s['id'] in args.case]
if args.case and set(args.case)-{s['id'] for s in selected}:
    parser.error('Unknown case requested')
if not args.worker:
    args.output.mkdir(parents=True, exist_ok=False)
    for s in selected:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                        '--case', s['id'], '--output', str(args.output/s['id'])], check=True)
    print('Exact replay passed for',len(selected),'cases, excluding timing.')
    sys.exit()
assert len(selected)==1
s = selected[0]
sys.path.insert(0,str(root/s['suite']/'sources'))
from verified_arithmetic import Arithmetic
from verified_network import VerifiedNetwork
from verified_integration import Box, Integrator
from maxgap import MaxGapIntegrator
from verified_residual import poisson_residual_integrand
from experiment_records import validate_record
args.output.mkdir(parents=True,exist_ok=False)
network_path = Path(s['network'])
manifest = load(Path(s['run'])/'manifest.json')
assert hashlib.sha256(network_path.read_bytes()).hexdigest()==manifest['network_sha256']
strip = lambda r: dict(r, history=[{k:v for k,v in h.items() if k!='seconds'} for h in r['history']])
for policy in ['maxgap','uniform']:
    a=Arithmetic(128)
    net=VerifiedNetwork.from_record(load(network_path),a)
    oracle=(lambda b:poisson_residual_integrand(net,b)) if s['kind']=='poisson_residual' else (lambda b:net.sobolev_integrand(b,order=s['order']))
    cls=MaxGapIntegrator if policy=='maxgap' else Integrator
    run=cls(a,oracle,Box(tuple(map(tuple,s['domain']))))
    if policy=='maxgap':run.run_maxgap(s['max_leaves'])
    else:run.run(policy='uniform',theta='7/20',tolerance='0',max_leaves=s['max_leaves'])
    record=run.serialise()
    validate_record(record,s['domain'])
    assert strip(record)==strip(load(Path(s['run'])/(policy+'.json'))),s['id']+' '+policy
    (args.output/(policy+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    if policy=='maxgap':
        audit=dict(nodes=run.node_records,refinements=run.refinements)
        assert audit==load(Path(s['run'])/'maxgap_audit.json')
        (args.output/'maxgap_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(s['id'],policy,'exact replay passed',flush=True)
