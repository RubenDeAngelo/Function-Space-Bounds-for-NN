"""Validate the bit-exact export already written by train.py."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--config',default='config/sector_relu.yaml');args=p.parse_args()
for folder in sorted(Path('models').glob('seed*')):
 d=torch.load(folder/'checkpoint.pt',weights_only=True)
 raw=b''.join(d[k].numpy().astype('<f4').tobytes() for k in ['w','b','a','b0'])
 assert raw==(folder/'weights.bin').read_bytes()
 manifest=json.loads((folder/'manifest.json').read_text());assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
 a=np.frombuffer(raw,dtype='<f4');assert np.array_equal(a.astype(np.float64).astype(np.float32),a)
 print(folder.name,manifest['sha256'],'verified')
