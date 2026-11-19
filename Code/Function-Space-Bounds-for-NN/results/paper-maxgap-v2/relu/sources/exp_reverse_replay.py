"""Replay frozen paper models using the reverse derivative backend, no training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
from experiment_records import dump, source_snapshot
from verified_network_reverse import VerifiedNetwork as ReverseNetwork
from exp_verified_studies import certify, validate_suite, render_suite, check_config

def worker(args):
    folder,config,spec=args
    task=folder/'tasks'/spec['id'];task.mkdir(parents=True)
    (task/'models').symlink_to((folder/'models').resolve(),target_is_directory=True)
    certify(task,config,[spec], network_class=ReverseNetwork,
            enclosure_method="reverse suffix Jacobian/Hessian interval AD (paper recurrences), ancestor intersections")
    return spec['id']

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.input/'config.json').read_text());check_config(config)
    specs=json.loads((args.input/'experiments.json').read_text())
    dump(out/'config.json',config);dump(out/'experiments.json',specs)
    shutil.copytree(args.input/'models',out/'models');source_snapshot(out)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in pool.map(worker,[(out,config,s) for s in specs]):
            print('Completed',name,flush=True)
    (out/'runs').mkdir();summary=[]
    for spec in specs:
        task=out/'tasks'/spec['id']
        shutil.move(task/'runs'/spec['id'],out/'runs'/spec['id'])
        summary.extend(json.loads((task/'summary.json').read_text()))
    dump(out/'summary.json',summary)
    shutil.rmtree(out/'tasks')
    validate_suite(out);render_suite(out)
if __name__=='__main__':main()
