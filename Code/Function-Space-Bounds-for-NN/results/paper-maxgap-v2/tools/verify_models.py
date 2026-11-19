"""Check saved coefficients against checkpoints, data and training manifests."""
from pathlib import Path
import argparse,sys,subprocess,json
root=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--suite',choices=['tanh','relu']);args=p.parse_args()
if args.suite is None:
    for suite in ['tanh','relu']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--suite',suite],check=True)
    print('All 13 checkpoint/coefficient/data artifact sets passed.')
    sys.exit()
sys.path.insert(0,str(root/args.suite/'sources'))
from verified_arithmetic import Arithmetic
from exp_verified_two_bumps import load_saved_network
from exp_verified_studies import load_model
if args.suite=='tanh':
    for folder in sorted((root/'tanh/models').iterdir()):
        if not folder.is_dir():continue
        (load_saved_network if folder.name=='two-bumps' else load_model)(folder,Arithmetic(128))
        print('tanh',folder.name,'passed',flush=True)
else:
    for arch in ['deep','shallow200']:
        for target in ['two-bumps','disk']:
            load_saved_network(root/'relu'/arch/target,Arithmetic(128))
            print('relu',arch,target,'passed',flush=True)
