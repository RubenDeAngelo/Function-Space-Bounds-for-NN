"""Build the submission's hybrid-only figures and tables from frozen records."""

import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from math import isqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

POLICIES = ("maxgap", "uniform")
STYLE = {"maxgap": ("#2264a4", "-"), "uniform": ("#cf632e", "--")}
ROW_END = r" \\"


def select_hybrid(records, key, expected):
    selected = [record for record in records if record[key] == "hybrid"]
    assert len(selected) == expected, (key, len(selected))
    return selected


def index_runs(records, cap_key):
    index = {
        (record["case"], record["quantity"], record["policy"], record[cap_key]): record
        for record in records
    }
    assert len(index) == len(records)
    return index


def rate_triangle(axis, power, color="0.25", position=(0.68, 0.78)):
    """A log-log reference triangle, independent of fitted data."""
    xlo, xhi = axis.get_xlim(); ylo, yhi = axis.get_ylim()
    x = 10**( __import__('math').log10(xlo) + position[0]*__import__('math').log10(xhi/xlo))
    y = 10**( __import__('math').log10(ylo) + position[1]*__import__('math').log10(yhi/ylo))
    factor = 1.3 if yhi/ylo < 100 else (2 if xhi/xlo < 1e4 else 4)
    axis.plot([x, x*factor, x*factor, x], [y, y, y/factor**power, y],
              color=color, linewidth=.9, linestyle="--")
    exponent = "1/2" if power == .5 else f"{power:g}"
    axis.annotate(rf"$N^{{-{exponent}}}$", (x*factor, y/factor**(power/2)),
                  xytext=(3, 0), textcoords="offset points", fontsize=10, color=color)


def plot_panels(runs, panels, output, x_label, controls=False):
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "legend.fontsize": 10})
    count=len(panels)
    figure, axes = plt.subplots(count//2, 2, figsize=(7.5, 3.0*(count//2)),
                                squeeze=False, layout="constrained")
    for panel_index, (axis, (case, quantity, cap, title, tolerance)) in enumerate(zip(axes.flat, panels)):
        exact = []
        exponential = controls and quantity == "H1"
        for policy in POLICIES:
            record = runs[case, quantity, policy, cap]
            history = [row for row in record["history"] if Fraction(row["gap"]) > 0]
            color, line = STYLE[policy]
            marker = "o" if policy == "maxgap" else "s"
            x=[row["leaves"] for row in history]; y=[float(Fraction(row["gap"])) for row in history]
            axis.plot(x,y,color=color,linestyle=line,linewidth=1.35,label=policy.title())
            # Every displayed marker is an actually computed record. Subsampling
            # only reduces overplotting; it does not create new curve values.
            step=max(1,len(history)//14)
            indices=sorted(set(range(0,len(history),step))|{len(history)-1})
            axis.plot([x[i] for i in indices],[y[i] for i in indices],linestyle="none",
                      marker=marker,markersize=3,color=color)
            if Fraction(record["final"]["gap"]) == 0:
                exact.append(f"{policy.title()}: exact at {record['final']['leaves']} leaves")
        axis.set_yscale("log")
        if not exponential:axis.set_xscale("log")
        if tolerance is not None:
            axis.axhline(tolerance,color="0.45",linewidth=.8,linestyle=":")

        label = r"Gap for $\|\Phi\|_{H^1}^2$" if quantity=="H1" else r"Gap for $\|\Phi\|_{L^2}^2$"
        axis.set(title=title+("\n"+"\n".join(exact) if exact else ""),xlabel=x_label,ylabel=label)
        axis.grid(alpha=.2);axis.legend(loc="upper right" if exponential else "lower left")
        if exponential:
            m=1 if case=="hinge-third" else 3
            # Theorem's exponential factor is 2^{-N/m}; a straight power-law
            # triangle on log-log axes would not represent this bound.
            x0=70 if m==1 else 85; dx=2*m; y0=1e-3
            axis.plot([x0,x0+dx,x0+dx,x0],[y0,y0,y0/4,y0],color=".25",ls="--",lw=.9)
            axis.annotate(rf"$2^{{-N/{m}}}$" if m>1 else r"$2^{-N}$",
                          (x0+dx,y0/2),xytext=(4,0),textcoords="offset points",fontsize=10)
        elif not controls:
            power = 2 if quantity=="L2" else 1
            rate_triangle(axis,power,STYLE['maxgap'][0],(.80,.91))
            rate_triangle(axis,power/2,STYLE['uniform'][0],(.80,.72))
    figure.savefig(output,metadata={"CreationDate":None,"ModDate":None})
    plt.close(figure)


def tex_gap(value):
    coefficient, sep, exponent = f"{float(Fraction(value)):.6g}".partition("e")
    if sep:
        return rf"{coefficient}\times10^{{{int(exponent)}}}"
    return coefficient


def write_table(path, columns, header, rows, long=False):
    environment = "longtable" if long else "tabular"
    top = [rf"\begin{{{environment}}}{{@{{}}{columns}@{{}}}}",
           r"\toprule", " & ".join(header) + ROW_END, r"\midrule"]
    if long:
        top.append(r"\endhead")
    lines = [" & ".join(row) + ROW_END for row in rows]
    path.write_text("\n".join(top + lines + [r"\bottomrule", rf"\end{{{environment}}}"]) + "\n")


def write_controls(records, latex):
    assert all(r["method"] == "hybrid" for r in records)
    assert Counter(r["status"] for r in records) == {
        "tolerance_reached": 12, "exact": 12, "leaf_budget_exhausted": 16,
    }
    runs = index_runs([r for r in records if r["precision_bits"] == 128], "max_leaves")
    panels = (
        ("hinge-third", "H1", 128, r"Non-dyadic hinge, squared full $H^1$", 1e-6),
        ("three-hinges", "H1", 128, r"Signed spline, squared full $H^1$", 1e-6),
    )
    plot_panels(runs, panels, latex / "images/controlled_certification.pdf", "Leaf intervals", controls=True)
    hats = (
        ("normalized-hat-128-clipped", "L2", 128, r"Clipped graph, squared $L^2$", None),
        ("normalized-hat-128-cancellation", "L2", 128, r"Cancelling graph, squared $L^2$", None),
    )
    plot_panels(runs,hats,latex / "images/representation_diagnostic.pdf","Leaf intervals",controls=True)
    rows = []
    for case, label in (("hinge-third", r"Hinge at $1/3$"), ("three-hinges", "Signed three-hinge spline")):
        for policy in POLICIES:
            record = runs[case, "H1", policy, 128]
            f = record["final"]
            status = {"tolerance_reached": "Tol.", "leaf_budget_exhausted": "Cap"}[record["status"]]
            rows.append([label, policy.title(), str(f["leaves"]), f"$%s$" % tex_gap(f["gap"]), status])
    write_table(latex / "tables/controlled_success.tex", "llrrl",
                ["Case", "Policy", "Leaves", "Gap", "Status"], rows)


def roots(lower, upper):
    scale = 1 << 128
    a = isqrt((lower.numerator * scale * scale) // lower.denominator)
    b = isqrt((upper.numerator * scale * scale) // upper.denominator)
    if Fraction(b * b, scale * scale) < upper:
        b += 1
    return Fraction(a, scale), Fraction(b, scale)


def outward(value, upper, digits):
    """Round a nonnegative rational outward at the requested precision."""
    assert value >= 0
    if value == 0:
        return "0"
    exponent = len(str(value.numerator)) - len(str(value.denominator))
    if value < Fraction(10) ** exponent:
        exponent -= 1
    places = digits - 1 - exponent
    scaled = value * Fraction(10) ** places
    units, remainder = divmod(scaled.numerator, scaled.denominator)
    if upper and remainder:
        units += 1
    if places <= 0:
        return str(units * 10 ** (-places))
    whole, fraction = divmod(units, 10 ** places)
    return f"{whole}.{fraction:0{places}d}"


def enclosures(lower, upper, digits=4):
    a, b = roots(lower, upper)
    assert a * a <= lower <= upper <= b * b
    intervals = []
    for lo, hi in ((lower, upper), (a, b)):
        l, u = outward(lo, False, digits), outward(hi, True, digits)
        assert Fraction(l) <= lo <= hi <= Fraction(u)
        intervals.append(f"$[{l},{u}]$")
    return intervals


def write_geometry(records, latex):
    assert all(r["rule"] == "hybrid" for r in records)
    assert Counter(r["status"] for r in records) == {
        "tolerance_reached": 18, "exact": 4, "leaf_budget_exhausted": 66,
    }
    runs = index_runs(records, "cap")
    panels = (
        ("control-hinge", "L2", 1024, r"Non-dyadic hinge, squared $L^2$", 1e-3),
        ("control-hinge", "H1", 1024, r"Non-dyadic hinge, squared full $H^1$", 1e-2),
        ("trained-deep-disk", "L2", 1024, r"Deep disk, squared $L^2$", None),
        ("trained-shallow200-disk", "H1", 1024, r"Shallow disk, squared full $H^1$", None),
    )
    plot_panels(runs, panels, latex / "images/geometry_certification.pdf", "Leaf cubes")
    gaps, endpoints = [], []
    for arch, label in (("deep", "D"), ("shallow200", "S")):
        for target, name in (("two-bumps", "Bumps"), ("disk", "Disk")):
            case = f"trained-{arch}-{target}"
            for quantity, norm in (("L2", r"$L^2$"), ("H1", r"$H^1$")):
                u, a = (runs[case, quantity, p, 1024]["final"] for p in ("uniform", "maxgap"))
                gaps.append([label, name, norm, f"$%s$" % tex_gap(u["gap"]),
                             f"$%s$" % tex_gap(a["gap"]), str(a["resolved_leaves"])])
                for cap in (256, 1024):
                    for policy in POLICIES:
                        record = runs[case, quantity, policy, cap]
                        f = record["final"]
                        assert f["leaves"] == cap and record["status"] == "leaf_budget_exhausted"
                        assert f["counts"]["cell_calls"] == (4 * cap - 1) // 3
                        endpoints.append([label, name, norm, policy.title(), f"{cap:,}",
                                          *enclosures(Fraction(f["lower"]), Fraction(f["upper"]))])
    assert len(gaps) == 8 and len(endpoints) == 32
    write_table(latex / "tables/geometry_gaps.tex", "lllrrr",
                ["Arch.", "Target", "Norm", "Uniform gap", "Maxgap gap", "Resolved/maxgap"], gaps)
    write_table(latex / "tables/hybrid_trained_endpoints.tex", "llllrrr",
                ["Arch.", "Target", "Norm", "Policy", "Leaves", "Squared norm", "Norm"],
                endpoints, long=True)


def write_disk(csv_path, latex):
    with csv_path.open(newline="") as stream:
        history = {(int(r["leaves"]), r["policy"]): r for r in csv.DictReader(stream)}
    rows = []
    for leaves in (16_384, 1_048_576):
        for policy in POLICIES:
            r = history[leaves, policy]
            assert int(r["local_calls"]) == (4 * leaves - 1) // 3
            rows.append([policy.title(), f"{leaves:,}",
                         *enclosures(Fraction(r["lower_exact"]), Fraction(r["upper_exact"]), digits=10)])
    write_table(latex / "tables/hybrid_disk_endpoints.tex", "lrrr",
                ["Policy", "Leaves", "Squared norm", "Norm"], rows)


def draw_disk_history(csv_path,latex):
    plt.rcParams.update({"font.size":11,"pdf.fonttype":42})
    rows=list(csv.DictReader(csv_path.open()))
    fig,ax=plt.subplots(figsize=(7.5,2.8),layout="constrained")
    for p in POLICIES:
        data=[r for r in rows if r["policy"]==p]
        color,line=STYLE[p]
        ax.loglog([int(r['leaves']) for r in data],
                  [float(Fraction(r['gap_exact'])) for r in data],
                  color=color,ls=line,marker='o' if p=='maxgap' else 's',
                  markersize=3,lw=1.35,label=p.title())
    ax.set(xlabel=r"Integration cells $N$",ylabel=r"Enclosure width of $\|v\|_{L^2}^2$")
    ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=10)
    rate_triangle(ax,2,position=(.70,.81))
    fig.savefig(latex/'images/disk_gap_revised.pdf',metadata={"CreationDate":None,"ModDate":None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--disk-history", type=Path)
    parser.add_argument("--latex", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    control_records = json.loads((args.records_root / "paper-originality-v1/results/records.json").read_text())
    geometry_paths = (args.records_root / "paper-geometry-v1/results").glob("*-hybrid-*.json")
    geometry_records = [json.loads(path.read_text()) for path in sorted(geometry_paths)]
    disk_history = args.disk_history or (
        args.records_root.parent / "experiments/smoothed_disk_figure1/data/history.csv"
    )
    write_controls(select_hybrid(control_records, "method", 40), args.latex)
    assert len(geometry_records) == 88
    write_geometry(geometry_records, args.latex)
    write_disk(disk_history, args.latex)
    draw_disk_history(disk_history,args.latex)


if __name__ == "__main__":
    main()
