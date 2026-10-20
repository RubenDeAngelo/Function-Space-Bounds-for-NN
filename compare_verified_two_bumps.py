"""Compare saved deep/shallow experiments; never recompute certificates as floats."""
import argparse
import json
from pathlib import Path
from exp_verified_two_bumps import dump, sha


def compare(deep, shallow, output, allow_training_schedule_difference=False):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from verified_arithmetic import rational

    output.mkdir(parents=True, exist_ok=False)
    paths = {"Deep": deep, "Shallow": shallow}
    manifests = {name: json.loads((p / "manifest.json").read_text()) for name, p in paths.items()}
    for key in ("quantity", "domain", "precision_bits", "max_leaves", "theta", "target"):
        if manifests["Deep"][key] != manifests["Shallow"][key]:
            raise ValueError(f"Experiment mismatch: {key}")
    with np.load(deep / "training_data.npz", allow_pickle=False) as d, np.load(
            shallow / "training_data.npz", allow_pickle=False) as s:
        if not all(np.array_equal(d[k], s[k]) for k in ("x", "y")):
            raise ValueError("Training inputs or labels differ")
    trainings = {name: json.loads((p / "training.json").read_text()) for name, p in paths.items()}
    for key in ("seed", "activation", "dtype"):
        if trainings["Deep"][key] != trainings["Shallow"][key]:
            raise ValueError(f"Training protocol mismatch: {key}")
    schedule_differences = {key: {name: t[key] for name, t in trainings.items()}
                           for key in ("adam_epochs", "adam_lr", "lbfgs_max_iter")
                           if trainings["Deep"][key] != trainings["Shallow"][key]}
    if schedule_differences and not allow_training_schedule_difference:
        raise ValueError(f"Training schedule mismatch: {schedule_differences}")

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    summary = {"same_training_arrays": True, "same_training_protocol": not schedule_differences,
               "training_schedule_differences": schedule_differences,
               "comparison_source_sha256": sha(Path(__file__)),
               "input_network_sha256": {name: sha(p / "network.json") for name, p in paths.items()},
               "same_enclosure_settings": True, "experiments": {}}
    for name, path in paths.items():
        network = json.loads((path / "network.json").read_text())
        parameters = sum(len(row) for w in network["weights"] for row in w)
        parameters += sum(len(b) for b in network["biases"])
        experiment = {"architecture": trainings[name]["architecture"],
                      "parameter_count": parameters,
                      "fit_metrics": trainings[name]["metrics"], "policies": {}}
        for policy, style in (("adaptive", "-"), ("uniform", "--")):
            record = json.loads((path / f"{policy}.json").read_text())
            history = record["history"]
            color = "#0072B2" if name == "Deep" else "#D55E00"
            for axis, quantity in zip(axes, ("gap_upper", "norm_gap_upper")):
                axis.loglog([h["leaves"] for h in history],
                            [float(rational(h[quantity])) for h in history],
                            style, marker="o", markersize=3, color=color,
                            label=f"{name}, {policy}")
            last = history[-1]
            experiment["policies"][policy] = {"status": record["status"],
                "exact_record": last,
                "approximate_integral_endpoints": [float(rational(last["integral"][k]))
                                                   for k in ("lower", "upper")],
                "approximate_norm_endpoints": [float(rational(last["norm"][k]))
                                               for k in ("lower", "upper")],
                "approximate_integral_gap": float(rational(last["gap_upper"])),
                "approximate_norm_gap": float(rational(last["norm_gap_upper"]))}
        summary["experiments"][name] = experiment
    for axis in axes:
        axis.set_xlabel("Leaf boxes $N$")
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=9)
    axes[0].set(ylabel="Squared $H^2$-norm enclosure gap $U-L$",
                title="(a) Integral uncertainty")
    axes[1].set(ylabel=r"$H^2$-norm enclosure gap $\sqrt{U}-\sqrt{L}$",
                title="(b) Norm uncertainty")
    fig.savefig(output / "deep-versus-shallow.png", dpi=220)
    fig.savefig(output / "deep-versus-shallow.svg")
    plt.close(fig)
    dump(output / "comparison.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--shallow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-training-schedule-difference", action="store_true")
    args = parser.parse_args()
    compare(args.deep, args.shallow, args.output, args.allow_training_schedule_difference)
