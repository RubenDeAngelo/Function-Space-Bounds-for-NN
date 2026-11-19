"""Run the disk experiment from the frozen checkpoint using exact arithmetic."""
import argparse
from collections import deque
import csv
import gzip
import heapq
import json
from pathlib import Path
import sys

from backend import Oracle, Q, children, volume
from fast_oracle import FastOracle

ROOT = Path(__file__).resolve().parent
DOMAIN = ((Q(-3, 2), Q(3, 2)),) * 2
MESH_LEAVES = 16_384
HISTORY_FIELDS = (
    "policy", "leaves", "lower_exact", "upper_exact", "gap_exact",
    "resolved", "local_calls",
)
EXTRA_MAXGAP = {
    24_577, 32_767, 49_153, 81_919, 100_000, 131_071, 196_609,
    393_217, 524_287, 786_433, 1_000_000,
}


def save_mesh(path, policy, cells):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as output:
            header = {"policy": policy, "leaves": len(cells)}
            output.write((json.dumps(header)[:-1] + ',"cells":[').encode())
            for i, cell in enumerate(cells.values()):
                if i:
                    output.write(b",")
                output.write(json.dumps(cell.record(), separators=(",", ":")).encode())
            output.write(b"]}\n")


def run(network, policy, cap, output, reference_oracle=False):
    oracle = Oracle(network) if reference_oracle else FastOracle(network)
    checkpoints = {4 ** j for j in range(cap.bit_length()) if 4 ** j <= cap} | {cap}
    if policy == "maxgap":
        checkpoints |= EXTRA_MAXGAP

    root = oracle.cell(DOMAIN)
    active = {0: root}
    queue = deque([0]) if policy == "uniform" else None
    heap = [(-root.gap, 0)] if policy == "maxgap" else None
    next_id = 1
    lower, upper = root.lower, root.upper
    with (output / f"history-{policy}.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        while True:
            leaves = len(active)
            if leaves in checkpoints or lower == upper:
                calls = oracle.counts["cell_calls"]
                assert calls == 1 + 4 * (leaves - 1) // 3
                writer.writerow(dict(
                    policy=policy, leaves=leaves, lower_exact=str(lower),
                    upper_exact=str(upper), gap_exact=str(upper - lower),
                    resolved=sum(cell.resolved for cell in active.values()),
                    local_calls=calls,
                ))
                stream.flush()
                print(policy, leaves, float(upper - lower), flush=True)
                if leaves == MESH_LEAVES:
                    save_mesh(output / f"partition-{policy}-16384.json.gz", policy, active)
            if leaves == cap or lower == upper:
                break
            index = heapq.heappop(heap)[1] if heap is not None else queue.popleft()
            if heap is not None and next_id <= 401:
                assert index == min(active, key=lambda key: (-active[key].gap, key))
            parent = active.pop(index)
            child_lower, child_upper = Q(0), Q(0)
            for box in children(parent.box):
                cell = oracle.cell(box, parent)
                active[next_id] = cell
                child_lower += cell.lower
                child_upper += cell.upper
                if heap is not None and cell.gap > 0:
                    heapq.heappush(heap, (-cell.gap, next_id))
                if queue is not None:
                    queue.append(next_id)
                next_id += 1
            assert parent.lower <= child_lower <= child_upper <= parent.upper
            lower += child_lower - parent.lower
            upper += child_upper - parent.upper

    assert lower == sum((cell.lower for cell in active.values()), Q(0))
    assert upper == sum((cell.upper for cell in active.values()), Q(0))
    assert volume(DOMAIN) == sum((volume(cell.box) for cell in active.values()), Q(0))
    return {"policy": policy, "leaves": len(active), "cap": cap,
            "status": "exact" if lower == upper else "leaf_budget_exhausted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("maxgap", "uniform"))
    parser.add_argument("--cap", type=int, default=1_048_576)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument("--reference-oracle", action="store_true")
    args = parser.parse_args()
    if args.cap < 1 or (args.cap - 1) % 3:
        parser.error("a quadtree cap must be positive and congruent to 1 modulo 3")
    if args.policy == "uniform" and not (
        args.cap & (args.cap - 1) == 0 and (args.cap.bit_length() - 1) % 2 == 0
    ):
        parser.error("a uniform cap must be a power of four")
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    network = json.loads((ROOT / "model/network.json").read_text())
    args.output.mkdir(parents=True)
    result = run(network, args.policy, args.cap, args.output, args.reference_oracle)
    (args.output / "run.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
