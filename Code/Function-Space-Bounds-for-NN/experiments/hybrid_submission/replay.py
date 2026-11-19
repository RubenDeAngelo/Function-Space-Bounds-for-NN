"""Replay the retained hybrid runs using their frozen numerical backends."""

import argparse
from collections import Counter
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

RECORDS_ROOT = Path(__file__).resolve().parents[2] / "results"
OUTCOMES = {
    "controls": {"tolerance_reached": 12, "exact": 12, "leaf_budget_exhausted": 16},
    "geometry": {"tolerance_reached": 18, "exact": 4, "leaf_budget_exhausted": 66},
}


def load_module(path):
    spec = importlib.util.spec_from_file_location("_frozen_hybrid_backend", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def without_times(value):
    if isinstance(value, dict):
        return {k: without_times(v) for k, v in value.items() if k != "seconds"}
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


def controls(records_root):
    archive = records_root / "paper-originality-v1"
    saved = json.loads((archive / "results/records.json").read_text())
    retained = [r for r in saved if r["method"] == "hybrid"]
    assert len(retained) == 40 and all(r["network"]["activation"] == "relu" for r in retained)
    run = load_module(archive / "experiment.py").run

    def replay(record):
        tolerance = record["tolerance"]
        return run(
            record["network"], record["quantity"], "hybrid", record["policy"],
            record["max_leaves"], None if tolerance is None else Fraction(tolerance),
            record["precision_bits"],
        )

    return retained, replay


def geometry(records_root):
    archive = records_root / "paper-geometry-v1"
    retained = [
        json.loads(path.read_text())
        for path in sorted((archive / "results").glob("*-hybrid-*.json"))
    ]
    assert len(retained) == 88 and all(r["rule"] == "hybrid" for r in retained)
    run = load_module(archive / "backend.py").run

    def replay(record):
        return run(
            record["network"], record["domain"], record["quantity"], "hybrid",
            record["policy"], record["cap"], record["tolerance"],
            scan=record["case"].startswith("control"),
        )

    return retained, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=("controls", "geometry"))
    parser.add_argument("--records-root", type=Path, default=RECORDS_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="A new output directory")
    parser.add_argument("--case", action="append", help="Replay only these saved case names")
    args = parser.parse_args()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    load = controls if args.suite == "controls" else geometry
    retained, replay = load(args.records_root.resolve())
    assert Counter(r["status"] for r in retained) == OUTCOMES[args.suite]
    if args.case:
        unknown = set(args.case) - {r["case"] for r in retained}
        if unknown:
            parser.error("unknown case(s): " + ", ".join(sorted(unknown)))
        retained = [r for r in retained if r["case"] in args.case]
    args.output.mkdir(parents=True)
    for i, record in enumerate(retained, 1):
        result = replay(record)
        saved = {k: record[k] for k in result}
        if without_times(result) != without_times(saved):
            raise RuntimeError(f"Replay mismatch for {record['case']} ({record['policy']})")
        for field in ("case", "network", "domain", "network_sha256", "exact_integral"):
            if field in record:
                result[field] = record[field]
        path = args.output / f"{i:03d}-{record['case']}-{record['quantity']}-{record['policy']}.json"
        with path.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(f"{i}/{len(retained)} {record['case']} {record['quantity']} {record['policy']}", flush=True)
    print(f"Exact replay matched {len(retained)} hybrid records excluding elapsed time.")


if __name__ == "__main__":
    main()
