"""
plot_results.py -- turn everything in ./runs into report-ready figures and tables.

Reads  runs/<run_name>/summary.json  and  runs/<run_name>/history.csv  (written by train.py).

Produces (in --out, default ./figures; each figure as .png and .pdf):
    1. params_vs_error      DenseNet-BC vs ResNet, test error vs #parameters
                            (reproduces paper Fig. 4 middle / Table 2 trend)
    2. ablation             growth-rate (k) and compression (theta) ablations
    3. curves               test error + training loss vs epoch (paper Fig. 4 right)
    4. degradation          plain nets vs residual nets (paper Fig. 6 of the ResNet paper)
    5. variants             DenseNet / -B / -C / -BC (paper Fig. 4 left)
    plus results.md and results.csv (full results table for the README / report).

How runs are classified (by run NAME, so use the names from the instructions):
    abl_k*   -> growth-rate ablation          (e.g. --name abl_k12)
    abl_t*   -> compression ablation          (e.g. --name abl_t0.5)
    var_*    -> variant comparison            (e.g. --name var_BC_L40)
    --arch plain -> degradation experiment
    everything else -> "main" runs used in the params-vs-error plot.
Runs that differ only by a trailing "_s<seed>" are treated as seeds of the same
configuration and shown as mean +/- std. So for multiple seeds name runs like
abl_k12_s0, abl_k12_s1 (default names already end in _s<seed>).

Usage:
    python plot_results.py                       # uses ./runs
    python plot_results.py --runs /content/drive/MyDrive/densenet_runs --out figures
    python plot_results.py --curve-runs densenet-BC_L100_k12_t0.5 resnet_L110
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120})

# Reference points from the DenseNet paper, Table 2, C10+ (standard augmentation, 300 epochs)
PAPER_REF = [("paper: DenseNet-BC-100 (k=12)", 0.8, 4.51, "*"),
             ("paper: pre-act ResNet-164", 1.7, 5.46, "s"),
             ("paper: pre-act ResNet-1001", 10.2, 4.62, "s")]


# --------------------------------------------------------------------------- loading
def load_runs(root):
    runs = []
    for sp in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        d = os.path.dirname(sp)
        with open(sp) as f:
            s = json.load(f)
        hist = []
        hp = os.path.join(d, "history.csv")
        if os.path.exists(hp):
            with open(hp) as f:
                hist = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
        s["history"] = hist
        s.setdefault("arch", "densenet")
        runs.append(s)
    return runs


def group_seeds(runs):
    """Merge runs that differ only by '_s<seed>' into one entry with mean/std."""
    buckets = defaultdict(list)
    for r in runs:
        buckets[re.sub(r"_s\d+$", "", r["name"])].append(r)
    groups = {}
    for name, rs in buckets.items():
        r0 = rs[0]
        err = np.array([r["final_test_err"] for r in rs])
        best = np.array([r["best_test_err"] for r in rs])
        groups[name] = dict(
            name=name, arch=r0["arch"], variant=r0["variant"], depth=r0["depth"],
            growth=r0["growth_rate"], theta=r0["theta"], bottleneck=r0["bottleneck"],
            params=r0["params"], epochs=r0["epochs"], aug=r0["augmentation"], n=len(rs),
            err=err.mean(), std=err.std(ddof=1) if len(rs) > 1 else 0.0, best=best.mean(),
            minutes=float(np.mean([r["train_minutes"] for r in rs])),
            hists=[r["history"] for r in rs if r["history"]])
    return groups


def category(g):
    n = g["name"]
    if n.startswith("abl_k"):
        return "abl_k"
    if n.startswith("abl_t"):
        return "abl_t"
    if n.startswith("var_"):
        return "var"
    if g["arch"] == "plain":
        return "plain"
    return "main"


def series_label(g):
    if g["arch"] == "densenet":
        names = {"BC": "DenseNet-BC", "B": "DenseNet-B", "C": "DenseNet-C", "plain": "DenseNet"}
        return f"{names[g['variant']]} (k={g['growth']})"
    if g["arch"] == "resnet":
        return "ResNet (pre-act)" if g["variant"].startswith("preact") else "ResNet"
    return "PlainNet"


def mean_curve(g, key):
    """Mean over seeds of history[key]; truncated to the shortest history."""
    if not g["hists"]:
        return None
    L = min(len(h) for h in g["hists"])
    return np.mean([[row[key] for row in h[:L]] for h in g["hists"]], axis=0)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {out}/{name}.png|pdf")


# --------------------------------------------------------------------------- plots
def plot_series(ax, groups, annotate=True, logx=True):
    """Error vs #params; one line per series label."""
    series = defaultdict(list)
    for g in groups:
        series[series_label(g)].append(g)
    for i, (label, gs) in enumerate(sorted(series.items())):
        gs.sort(key=lambda g: g["params"])
        x = [g["params"] / 1e6 for g in gs]
        y = [g["err"] for g in gs]
        e = [g["std"] for g in gs]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=3, lw=1.6, label=label, color=f"C{i}")
        if annotate:
            for g, xi, yi in zip(gs, x, y):
                ax.annotate(f"{g['depth']}", (xi, yi), textcoords="offset points",
                            xytext=(4, 5), fontsize=7, color=f"C{i}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("# parameters (millions)" + (", log scale" if logx else ""))
    ax.set_ylabel("final test error (%)")


def fig_params_vs_error(groups, out, logx, paper_ref):
    main = [g for g in groups if category(g) == "main"]
    if not main:
        print("  [skip] params_vs_error: no main runs found")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    plot_series(ax, main, logx=logx)
    if paper_ref and any(g["aug"] for g in main):
        for label, x, y, m in PAPER_REF:
            ax.scatter([x], [y], marker=m, s=70, facecolors="none", edgecolors="k", lw=1.2, label=label)
    ax.set_title("Accuracy vs. parameters on CIFAR-10" + ("+" if any(g["aug"] for g in main) else ""))
    ax.legend(fontsize=7.5)
    ax.text(0.99, 0.02, "numbers next to points = network depth", transform=ax.transAxes,
            ha="right", fontsize=7, color="gray")
    save(fig, out, "params_vs_error")


def fig_ablation(groups, out):
    ak = sorted([g for g in groups if category(g) == "abl_k"], key=lambda g: g["growth"])
    at = sorted([g for g in groups if category(g) == "abl_t"], key=lambda g: g["theta"])
    panels = [(p, t, lab, key) for p, t, lab, key in
              [(ak, "Growth-rate ablation", "growth rate k", "growth"),
               (at, "Compression ablation", r"compression factor $\theta$", "theta")] if p]
    if not panels:
        print("  [skip] ablation: no abl_k* / abl_t* runs found")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (gs, title, xlabel, key) in zip(axes, panels):
        xs = [g[key] for g in gs]
        ax.errorbar(xs, [g["err"] for g in gs], yerr=[g["std"] for g in gs],
                    marker="o", capsize=3, lw=1.6, color="C0")
        for g, x in zip(gs, xs):
            ax.annotate(f"{g['params']/1e6:.2f}M", (x, g["err"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="C3")
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("final test error (%)")
        ref = gs[0]
        ax.set_title(f"{title}\n(L={ref['depth']}, red = #params)", fontsize=10)
        ax.margins(y=0.2)
    save(fig, out, "ablation")


def fig_variants(groups, out, logx):
    var = [g for g in groups if category(g) == "var"]
    if not var:
        print("  [skip] variants: no var_* runs found")
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    series = defaultdict(list)
    for g in var:
        series[{"BC": "DenseNet-BC", "B": "DenseNet-B", "C": "DenseNet-C",
                "plain": "DenseNet"}.get(g["variant"], g["variant"])].append(g)
    for i, (label, gs) in enumerate(sorted(series.items())):
        gs.sort(key=lambda g: g["params"])
        ax.errorbar([g["params"] / 1e6 for g in gs], [g["err"] for g in gs],
                    yerr=[g["std"] for g in gs], marker="o", capsize=3, lw=1.6, label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("# parameters (millions)")
    ax.set_ylabel("final test error (%)")
    ax.set_title("DenseNet variants: parameter efficiency")
    ax.legend()
    save(fig, out, "variants")


def fig_curves(groups, out, curve_runs):
    by_name = {g["name"]: g for g in groups}
    if curve_runs:
        chosen = [by_name[n] for n in curve_runs if n in by_name]
        for n in curve_runs:
            if n not in by_name:
                print(f"  [warn] curve run '{n}' not found. Available: {sorted(by_name)}")
    else:  # default: largest main DenseNet-BC and largest main ResNet
        main = [g for g in groups if category(g) == "main" and g["hists"]]
        chosen = []
        for arch, pick in [("densenet", lambda g: g["variant"] == "BC"), ("resnet", lambda g: True)]:
            c = [g for g in main if g["arch"] == arch and pick(g)]
            if c:
                chosen.append(max(c, key=lambda g: g["params"]))
    chosen = [g for g in chosen if g["hists"]]
    if not chosen:
        print("  [skip] curves: no runs with history.csv")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for i, g in enumerate(chosen):
        lbl = f"{series_label(g)} L={g['depth']} ({g['params']/1e6:.2f}M)"
        te = 100 - mean_curve(g, "test_acc")
        tl = mean_curve(g, "train_loss")
        ep = np.arange(1, len(te) + 1)
        a1.plot(ep, te, color=f"C{i}", lw=1.6, label=lbl)
        a2.plot(ep, tl, color=f"C{i}", lw=1.6, label=lbl)
    a1.set_xlabel("epoch"); a1.set_ylabel("test error (%)"); a1.set_title("Test error")
    a1.set_ylim(top=min(40, a1.get_ylim()[1]))
    a2.set_xlabel("epoch"); a2.set_ylabel("training loss"); a2.set_yscale("log")
    a2.set_title("Training loss")
    a1.legend(fontsize=8)
    save(fig, out, "curves")


def fig_degradation(groups, out):
    plain = [g for g in groups if category(g) == "plain" and g["hists"]]
    if not plain:
        print("  [skip] degradation: no --arch plain runs found")
        return
    depths = sorted({g["depth"] for g in plain})
    resnets = [g for g in groups if g["arch"] == "resnet" and not g["variant"].startswith("preact")
               and g["depth"] in depths and g["hists"] and category(g) == "main"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, gs, title in [(axes[0], plain, "Plain nets (no shortcuts)"),
                          (axes[1], resnets, "ResNets (with shortcuts)")]:
        for i, g in enumerate(sorted(gs, key=lambda g: g["depth"])):
            ep = np.arange(1, len(mean_curve(g, "test_acc")) + 1)
            ax.plot(ep, 100 - mean_curve(g, "train_acc"), "--", color=f"C{i}", lw=1.2,
                    label=f"{g['depth']}-layer train")
            ax.plot(ep, 100 - mean_curve(g, "test_acc"), "-", color=f"C{i}", lw=1.8,
                    label=f"{g['depth']}-layer test")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylim(0, 40)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("error (%)")
    fig.suptitle("Degradation problem (dashed = train error, bold = test error)", fontsize=10)
    save(fig, out, "degradation")


# --------------------------------------------------------------------------- table
def write_tables(groups, out):
    os.makedirs(out, exist_ok=True)
    order = {"densenet": 0, "resnet": 1, "plain": 2}
    rows = sorted(groups, key=lambda g: (order.get(g["arch"], 9), category(g), g["params"]))
    header = ["Run", "Model", "Depth", "k", "theta", "Params (M)", "Epochs", "Aug", "Seeds",
              "Final test err (%)", "Best-epoch err (%)", "Train min"]

    def model_name(g):
        return series_label(g).split(" (k=")[0] if g["arch"] == "densenet" else series_label(g)

    table = []
    for g in rows:
        dn = g["arch"] == "densenet"
        err = f"{g['err']:.2f}" + (f" ± {g['std']:.2f}" if g["n"] > 1 else "")
        table.append([g["name"], model_name(g), g["depth"], g["growth"] if dn else "-",
                      g["theta"] if dn else "-", f"{g['params']/1e6:.3f}", g["epochs"],
                      "yes" if g["aug"] else "no", g["n"], err, f"{g['best']:.2f}", f"{g['minutes']:.1f}"])
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(table)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in table]
    md_text = "\n".join(md)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(md_text + "\n\n*Final test err = error after the last epoch (mean ± std over seeds if >1). "
                "Best-epoch err is picked on the test set and is only informational.*\n")
    print(f"  saved {out}/results.md|csv\n")
    print(md_text)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="./runs", help="folder containing the run directories")
    ap.add_argument("--out", default="./figures")
    ap.add_argument("--curve-runs", nargs="*", default=None,
                    help="group names to show in the curves figure (see results.md 'Run' column)")
    ap.add_argument("--linear-x", action="store_true", help="linear x-axis (like the paper) instead of log")
    ap.add_argument("--no-paper-ref", action="store_true", help="hide the paper's reference points")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        raise SystemExit(f"No runs found in {args.runs} (expected {args.runs}/<name>/summary.json)")
    groups = list(group_seeds(runs).values())
    print(f"Loaded {len(runs)} runs -> {len(groups)} configurations from {args.runs}\n")

    logx = not args.linear_x
    fig_params_vs_error(groups, args.out, logx, not args.no_paper_ref)
    fig_ablation(groups, args.out)
    fig_curves(groups, args.out, args.curve_runs)
    fig_degradation(groups, args.out)
    fig_variants(groups, args.out, logx)
    write_tables(groups, args.out)


if __name__ == "__main__":
    main()
