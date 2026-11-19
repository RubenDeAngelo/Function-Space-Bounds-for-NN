"""Shared exact-record validation and immutable-source snapshots."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def source_snapshot(output):
    import gmpy2
    root = Path(__file__).resolve().parent
    target = Path(output) / 'sources'
    target.mkdir(exist_ok=False)
    files = list(root.glob('*.py')) + list((root/'tests').glob('*.py'))
    files += list((root/'experiments').glob('*.json'))
    files += [root/'requirements-verified.txt', root/'requirements-verified-lock.txt']
    hashes = {}
    for p in sorted(files):
        rel = p.relative_to(root)
        dst = target/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
        hashes[str(rel)] = sha(dst)
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True)
    except subprocess.CalledProcessError:
        head, status = None, 'Source snapshot without Git checkout'
    record = {'source_sha256': hashes, 'git_head': head, 'git_status': status,
              'python': sys.version, 'platform': platform.platform(),
              'gmpy2': gmpy2.version(), 'mpfr': gmpy2.mpfr_version(),
              'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)}
    dump(Path(output)/'environment.json', record)
    return record


def validate_record(record, domain, p=2):
    """Check exact dyadic geometry, volumes, endpoint aggregation and roots.

    This checks record consistency, not the range oracle's inclusion proof.
    """
    domain = [[Q(str(x)) for x in pair] for pair in domain]
    dim = len(domain); branching = 2**dim
    root_volume = Q(1)
    for lo, hi in domain:
        if lo >= hi: raise ValueError('Invalid root domain')
        root_volume *= hi-lo
    cells = record['cells']; history = record['history']
    assert history and len(cells) == history[-1]['leaves']
    keys = set(); volume_sum = Q(0); lower = Q(0); upper = Q(0)
    for cell in cells:
        bounds = [[Q(x) for x in pair] for pair in cell['bounds']]
        depth = cell['depth']; assert isinstance(depth, int) and depth >= 0
        assert len(bounds) == dim
        indices = []; volume = Q(1)
        for (a,b),(lo,hi) in zip(domain,bounds):
            assert a <= lo < hi <= b and hi-lo == (b-a)/2**depth
            index = (lo-a)*2**depth/(b-a)
            assert index.denominator == 1
            indices.append(int(index)); volume *= hi-lo
        key = (depth, *indices); assert key not in keys; keys.add(key)
        volume_sum += volume
        fl,fu = (Q(cell['integrand'][k]) for k in ('lower','upper'))
        il,iu = (Q(cell['integral'][k]) for k in ('lower','upper'))
        assert 0 <= fl <= fu and il <= volume*fl <= volume*fu <= iu
        assert Q(cell['indicator']) == iu-il
        lower += il; upper += iu
    for key in keys:
        depth, *indices = key
        for ancestor in range(depth):
            assert (ancestor, *(i//2**(depth-ancestor) for i in indices)) not in keys
    assert volume_sum == root_volume
    for h in history:
        n = h['leaves']; assert n >= 1 and (n-1) % (branching-1) == 0
        assert h['evaluations'] == 1+branching*(n-1)//(branching-1)
        lo,hi = (Q(h['integral'][k]) for k in ('lower','upper'))
        nl,nu = (Q(h['norm'][k]) for k in ('lower','upper'))
        assert 0 <= lo <= hi and Q(h['gap_upper']) >= hi-lo
        assert 0 <= nl <= nu and nl**p <= lo and nu**p >= hi
        assert Q(h['norm_gap_upper']) >= nu-nl
    lo,hi = (Q(history[-1]['integral'][k]) for k in ('lower','upper'))
    assert lo <= lower <= upper <= hi
    return {'cells': len(cells), 'volume': str(volume_sum), 'status': record['status']}
