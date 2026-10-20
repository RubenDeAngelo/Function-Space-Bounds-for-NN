"""Reproducible Figure 1 candidate: one fixed network, one verified integrand.

Training and rendering use ordinary arithmetic; all enclosure and stopping
decisions go through verified_* modules. Plot floats are display-only.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(output, epochs, seed):
    import numpy as np
    import torch
    from lib_nn_model import FlexibleNet

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # 50% uniform, 25% around each bump, truncated by rejection, not clipping.
    points = [rng.uniform(-1, 1, (2048, 2))]
    for centre, width in (([-0.4, -0.3], 0.18), ([0.4, 0.35], 0.09)):
        accepted = []
        while sum(len(p) for p in accepted) < 1024:
            proposed = rng.normal(centre, 2*width, (1024, 2))
            accepted.append(proposed[(abs(proposed) <= 1).all(axis=1)])
        points.append(np.concatenate(accepted)[:1024])
    x = np.concatenate(points)
    rng.shuffle(x)

    def target(x):
        return (torch.exp(-((x[:, 0]+0.4)**2+(x[:, 1]+0.3)**2)/(2*0.18**2))
                + 0.7*torch.exp(-((x[:, 0]-0.4)**2+(x[:, 1]-0.35)**2)/(2*0.09**2)))[:, None]

    inputs = torch.tensor(x, dtype=torch.float64)
    targets = target(inputs)
    model = FlexibleNet([2, 32, 32, 32, 1], activation="tanh").double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    started = perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = ((model(inputs)-targets)**2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"training {epoch+1}/{epochs}: MSE={loss.item():.6g}", flush=True)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=250,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-10)

    def closure():
        optimizer.zero_grad()
        loss = ((model(inputs)-targets)**2).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    torch.save(model.state_dict(), output / "checkpoint.pt")
    np.savez_compressed(output / "training_data.npz", x=x, y=targets.detach().numpy())
    line = np.linspace(-1, 1, 161)
    xx, yy = np.meshgrid(line, line)
    grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float64)
    with torch.no_grad():
        prediction, truth = model(grid), target(grid)
        metrics = {"grid_MSE": ((prediction-truth)**2).mean().item(),
                   "grid_max_abs_error": (prediction-truth).abs().max().item(),
                   "training_MSE": ((model(inputs)-targets)**2).mean().item(),
                   "seconds": perf_counter()-started}
    np.savez_compressed(output / "fit_grid.npz", x=xx, y=yy,
                        target=truth.numpy().reshape(xx.shape),
                        prediction=prediction.numpy().reshape(xx.shape))
    dump(output / "training.json", {"seed": seed, "architecture": [2, 32, 32, 32, 1],
         "activation": "tanh", "dtype": "float64", "training_points": 4096,
         "sampling": "2048 uniform; 1024 each N(centre,(2*width)^2 I), reject outside domain",
         "adam_epochs": epochs, "adam_lr": 0.001, "lbfgs_max_iter": 250,
         "metrics": metrics, "checkpoint_sha256": sha(output / "checkpoint.pt"),
         "training_data_sha256": sha(output / "training_data.npz")})
    print("fit diagnostics:", metrics, flush=True)
    return model


def render(output, records):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LogNorm
    from verified_arithmetic import rational

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), layout="constrained",
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    colors = {"adaptive": "#0072B2", "uniform": "#D55E00"}
    for policy, record in records.items():
        axes[0].loglog([r["leaves"] for r in record["history"]],
                       [float(rational(r["gap_upper"])) for r in record["history"]],
                       "o-", markersize=3, label=policy.capitalize(), color=colors[policy])
    axes[0].set(xlabel="Leaf boxes $N$", ylabel="Integral enclosure gap $U-L$",
                title="(a) Adaptive versus uniform")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend()
    positive = [float(rational(c["indicator"])) for r in records.values()
                for c in r["cells"] if rational(c["indicator"]) > 0]
    norm = LogNorm(min(positive), max(positive)) if positive else LogNorm(1e-30, 1)
    for axis, (policy, record), label in zip(axes[1:], records.items(), ("b", "c")):
        patches, indicators = [], []
        for c in record["cells"]:
            (x0, x1), (y0, y1) = [[float(rational(x)) for x in b] for b in c["bounds"]]
            patches.append(Rectangle((x0, y0), x1-x0, y1-y0))
            indicators.append(float(rational(c["indicator"])))
        collection = PatchCollection(patches, cmap="viridis", norm=norm,
                                     edgecolors="#ffffff", linewidths=0.15)
        collection.set_array(np.ma.masked_less_equal(indicators, 0))
        axis.add_collection(collection)
        axis.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="$x$", ylabel="$y$",
                 title=f"({label}) {policy.capitalize()}: {len(patches)} boxes")
        axis.set_aspect("equal")
    fig.colorbar(collection, ax=list(axes[1:]), shrink=0.83,
                 label="Local integral gap contribution")
    fig.savefig(output / "figure1.svg")
    fig.savefig(output / "figure1.png", dpi=220)
    plt.close(fig)
    # Show whether the network represents the intended target; no certificate
    # is inferred from this numerical grid diagnostic.
    fit = np.load(output / "fit_grid.npz")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
    for axis, key, title in zip(axes, ("target", "prediction", "error"),
                                ("Target", "Fitted network", "Absolute fit error")):
        z = abs(fit["target"]-fit["prediction"]) if key == "error" else fit[key]
        im = axis.pcolormesh(fit["x"], fit["y"], z, shading="auto")
        axis.set(title=title, xlabel="$x$", ylabel="$y$")
        axis.set_aspect("equal")
        fig.colorbar(im, ax=axis)
    fig.savefig(output / "fit_diagnostics.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, help="Directory with saved exact network and training artifacts")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--bits", type=int, default=128)
    parser.add_argument("--max-leaves", type=int, default=1024)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_leaves < 1:
        parser.error("Positive epochs and leaf budget required")
    args.output.mkdir(parents=True, exist_ok=False)
    import gmpy2 as g
    from verified_arithmetic import Arithmetic
    from verified_network import VerifiedNetwork
    from verified_integration import Box, Integrator
    a = Arithmetic(args.bits)
    if args.reuse:
        import shutil
        network = VerifiedNetwork.from_record(json.loads((args.reuse / "network.json").read_text()), a)
        for file in ("checkpoint.pt", "training_data.npz", "fit_grid.npz", "training.json"):
            shutil.copy2(args.reuse / file, args.output / file)
    else:
        network = VerifiedNetwork.from_torch(train(args.output, args.epochs, args.seed), a)
    dump(args.output / "network.json", network.serialise())
    source_files = [Path(__file__), Path("verified_arithmetic.py"),
                    Path("verified_network.py"), Path("verified_integration.py")]
    manifest = {"command": sys.argv, "python": sys.version, "platform": platform.platform(),
                "processor": platform.processor(), "gmpy2": g.version(),
                "mpfr": g.mpfr_version(), "gmp": g.mp_version(),
                "precision_bits": args.bits, "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "source_sha256": {p.name: sha(p) for p in source_files},
                "network_sha256": sha(args.output / "network.json"),
                "quantity": "integral sum_{|alpha|<=2} |D^alpha Phi|^2; mixed derivative once",
                "domain": [["-1", "1"], ["-1", "1"]], "theta": "0.35",
                "max_leaves": args.max_leaves, "tolerance": "0",
                "target": "exp(-((x+.4)^2+(y+.3)^2)/(2*.18^2)) + .7*exp(-((x-.4)^2+(y-.35)^2)/(2*.09^2))",
                "coefficient_semantics": "stored binary checkpoint numbers interpreted as exact real coefficients"}
    manifest["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    dump(args.output / "manifest.json", manifest)
    records = {}
    for policy in ("adaptive", "uniform"):
        print(f"{policy}: starting verified integration at {args.bits} bits", flush=True)
        run = Integrator(a, network.sobolev_integrand, Box(((-1, 1), (-1, 1))))
        # Run in individual steps to expose progress without changing the rule.
        while True:
            run.run(policy=policy, max_leaves=args.max_leaves, max_steps=1)
            last = run.history[-1]
            print(f"{policy}: N={last['leaves']}, evaluations={last['evaluations']}, "
                  f"seconds={last['seconds']:.1f}, gap~{float(run.total().width):.6g}", flush=True)
            if run.status != "step_budget_exhausted":
                break
            # Across one-step calls, explicitly retain the stagnation check.
            if len(run.history) >= 9 and all(
                    g.mpq(r["gap_upper"]) >= g.mpq(run.history[-9]["gap_upper"])
                    for r in run.history[-8:]):
                run.status = "gap_stagnation"
                break
        records[policy] = run.serialise()
        dump(args.output / f"{policy}.json", records[policy])
        print(policy, run.status, flush=True)
    render(args.output, records)


if __name__ == "__main__":
    main()
