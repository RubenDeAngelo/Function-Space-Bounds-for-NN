"""Verify a release checkout without changing stored scientific evidence."""
from pathlib import Path
import ast,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'release/SHA256.json').read_text())
for name,digest in manifest.items():
    path=ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise SystemExit('Release integrity failure: '+name)
checked=0
for path in (ROOT/'Code').rglob('*.py'):
    tree=ast.parse(path.read_text(),filename=str(path))
    for node in ast.walk(tree):
        names=[]
        if isinstance(node,ast.Import):names=[a.name for a in node.names]
        elif isinstance(node,ast.ImportFrom):names=[node.module or '']
        if any(n.split('.')[0]=='mothercode' for n in names):
            raise SystemExit('Unexpected external consolidation import: '+str(path))
    if 'mothercode' in path.read_text().lower():
        raise SystemExit('Unexpected consolidation reference in executable source: '+str(path))
    checked+=1
print(f'Verified {len(manifest)} file hashes and {checked} Python sources; no Mothercode dependency.')
