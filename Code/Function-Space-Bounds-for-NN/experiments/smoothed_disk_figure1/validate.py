"""Check saved Figure 1 data and, optionally, exact 16,384-leaf replays."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from fractions import Fraction as Q
import sys

ROOT = Path(__file__).resolve().parent
POLICIES = ("maxgap", "uniform")


def read_mesh(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def validate_mesh(policy, mesh, recorded):
    assert mesh["policy"] == policy and mesh["leaves"] == len(mesh["cells"]) == 16_384
    lower, upper, area, resolved = Q(0), Q(0), Q(0), 0
    keys = set()
    for row in mesh["cells"]:
        box = [(Q(a), Q(b)) for a, b in row["box"]]
        h = box[0][1] - box[0][0]
        side_count = Q(3) / h
        assert side_count.denominator == 1
        sides = int(side_count)
        assert sides > 0 and not (sides & (sides - 1))
        depth = sides.bit_length() - 1
        indices = []
        for a, b in box:
            assert Q(-3, 2) <= a < b <= Q(3, 2) and b - a == h
            i = (a + Q(3, 2)) / h
            assert i.denominator == 1 and 0 <= i < sides
            indices.append(int(i))
        key = (depth, *indices)
        assert key not in keys
        if policy == "uniform":
            assert depth == 7
        keys.add(key)
        a, b = Q(row["lower"]), Q(row["upper"])
        assert 0 <= a <= b and row["resolved"] == (a == b)
        lower += a
        upper += b
        area += h * h
        resolved += row["resolved"]
    for depth, x, y in keys:
        assert all((j, x >> (depth - j), y >> (depth - j)) not in keys for j in range(depth))
    assert area == 9 and lower == Q(recorded["lower_exact"]) and upper == Q(recorded["upper_exact"])
    assert resolved == int(recorded["resolved"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxgap-replay", type=Path)
    parser.add_argument("--uniform-replay", type=Path)
    args = parser.parse_args()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)

    provenance = json.loads((ROOT / "data/provenance.json").read_text())
    assert hashlib.sha256((ROOT / "model/network.json").read_bytes()).hexdigest() == provenance["frozen_network_sha256"]
    for name, sha in provenance["figure_source_sha256"].items():
        assert hashlib.sha256((ROOT / "panels" / name).read_bytes()).hexdigest() == sha

    all_rows = read_rows(ROOT / "data/history.csv")
    for policy in POLICIES:
        rows = [r for r in all_rows if r["policy"] == policy]
        leaves = [int(r["leaves"]) for r in rows]
        assert leaves == sorted(set(leaves)) and leaves[0] == 1 and leaves[-1] == 1_048_576
        if policy == "uniform":
            assert leaves == [4 ** j for j in range(11)]
        for row, n in zip(rows, leaves):
            a, b = Q(row["lower_exact"]), Q(row["upper_exact"])
            assert 0 <= a <= b and Q(row["gap_exact"]) == b - a
            assert (n - 1) % 3 == 0 and int(row["local_calls"]) == 1 + 4 * (n - 1) // 3
            assert 0 <= int(row["resolved"]) <= n
        for a, b in zip(rows, rows[1:]):
            assert Q(a["lower_exact"]) <= Q(b["lower_exact"]) <= Q(b["upper_exact"]) <= Q(a["upper_exact"])
        mesh_file = f"partition-{policy}-16384.json.gz"
        mesh = read_mesh(ROOT / "data" / mesh_file)
        recorded = next(row for row in rows if row["leaves"] == "16384")
        validate_mesh(policy, mesh, recorded)
        replay = getattr(args, f"{policy}_replay")
        if replay:
            assert read_rows(replay / f"history-{policy}.csv") == [r for r in rows if int(r["leaves"]) <= 16_384]
            assert read_mesh(replay / mesh_file) == mesh
        print(f"{policy}: checked {len(rows)} history rows and 16,384 mesh leaves", flush=True)


if __name__ == "__main__":
    main()
