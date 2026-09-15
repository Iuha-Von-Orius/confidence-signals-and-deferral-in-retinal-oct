"""Compare global and Mondrian prediction sets at alpha = 0.02 on validation."""

from paths import PROJECT

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = PROJECT / "cache"
RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "06_conformal"
TABDIR = RESULTS / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)

ALPHA = 0.02
SPLIT = "val"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})

CLASS_COLOURS = {"CNV": "#1f77b4", "DME": "#ff7f0e",
                 "DRUSEN": "#d62728", "NORMAL": "#2ca02c"}
COLOUR_LIST = [CLASS_COLOURS[c] for c in CLASSES]
SCHEME_COLOURS = {"global": "#2c3e50", "mondrian": "#b8560f"}


def load():
    probs = np.load(CACHE / f"{SPLIT}_probs.npy")
    labels = np.load(CACHE / f"{SPLIT}_labels.npy")
    with open(CACHE / "conformal_params.json") as fh:
        params = json.load(fh)
    q_global = params["q_global"][str(ALPHA)]
    q_mond = [params["q_mondrian"][str(ALPHA)][c] for c in CLASSES]
    return probs, labels, probs.argmax(1) == labels, q_global, q_mond


def build_sets(probs, q_global, q_mond):
    """Membership matrices under both schemes."""
    g = (1.0 - probs) <= q_global
    m = np.zeros_like(probs, dtype=bool)
    for k in range(N_CLASSES):
        m[:, k] = (1.0 - probs[:, k]) <= q_mond[k]
    return {"global": g, "mondrian": m}


# Figures
def fig_thresholds(q_global, q_mond):
    """Admission thresholds per class."""
    t_g = [1 - q_global] * N_CLASSES
    t_m = [1 - q for q in q_mond]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(N_CLASSES)
    ax.bar(x - 0.2, t_g, 0.38, label="Global", color=SCHEME_COLOURS["global"],
           alpha=0.85)
    bars = ax.bar(x + 0.2, t_m, 0.38, label="Mondrian",
                  color=SCHEME_COLOURS["mondrian"], alpha=0.85)

    for i, (bar, t) in enumerate(zip(bars, t_m)):
        if t < 1e-6:
            ax.annotate("t = 0\nadmitted\nunconditionally",
                        xy=(x[i] + 0.2, 0), xytext=(x[i] + 0.2, 0.28),
                        ha="center", fontsize=8.5, color="#8b1a1a",
                        arrowprops=dict(arrowstyle="->", color="#8b1a1a", lw=1.2))
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, t + 0.015,
                    f"{t:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Admission threshold  t = 1 − q̂")
    ax.set_title(f"A class enters the set when p̂ ≥ t   (α = {ALPHA}, {SPLIT})")
    ax.legend()
    fig.savefig(FIGDIR / "06b_scheme_thresholds.png")
    plt.close(fig)


def fig_setsize(sets):
    """Set-size distribution under each scheme."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (name, s) in zip(axes, sets.items()):
        sizes = s.sum(axis=1)
        counts = np.bincount(sizes, minlength=N_CLASSES + 1)
        frac = counts / counts.sum()
        bars = ax.bar(range(N_CLASSES + 1), frac,
                      color=SCHEME_COLOURS[name], alpha=0.85)
        for b, f, c in zip(bars, frac, counts):
            if f > 0.0005:
                ax.text(b.get_x() + b.get_width() / 2, f + 0.012,
                        f"{f*100:.1f}%\n(n={c:,})", ha="center", fontsize=8)
        p = frac[frac > 0]
        ax.set_xlabel("Set size |C(x)|")
        ax.set_ylabel("Proportion")
        ax.set_title(f"{name.capitalize()}\n"
                     f"mean {sizes.mean():.3f},  "
                     f"entropy {-(p*np.log2(p)).sum():.3f} bits", fontsize=10)
        ax.set_ylim(0, 1.12)
    fig.suptitle(f"Set-size distribution at α = {ALPHA} — {SPLIT}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06b_scheme_setsize.png")
    plt.close(fig)


def fig_membership(sets):
    """How often each class appears in a prediction set, irrespective of the truth."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(N_CLASSES)
    for i, (name, s) in enumerate(sets.items()):
        rates = s.mean(axis=0) * 100
        bars = ax.bar(x + (i - 0.5) * 0.38, rates, 0.38,
                      label=name.capitalize(), color=SCHEME_COLOURS[name],
                      alpha=0.85)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.1f}",
                    ha="center", fontsize=8)
    ax.axhline(100, color="#d62728", ls="--", lw=1.4,
               label="100% — present in every set")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Appears in % of prediction sets")
    ax.set_ylim(0, 118)
    ax.set_title(f"Class membership frequency (α = {ALPHA}, {SPLIT})")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "06b_scheme_membership.png")
    plt.close(fig)


def fig_coverage(sets, labels):
    """Per-class coverage under each scheme."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(N_CLASSES)
    for i, (name, s) in enumerate(sets.items()):
        cov = [s[labels == k, k].mean() for k in range(N_CLASSES)]
        bars = ax.bar(x + (i - 0.5) * 0.38, cov, 0.38,
                      label=name.capitalize(), color=SCHEME_COLOURS[name],
                      alpha=0.85)
        for b, c in zip(bars, cov):
            ax.text(b.get_x() + b.get_width() / 2, c + 0.006, f"{c:.3f}",
                    ha="center", fontsize=8)
    ax.axhline(1 - ALPHA, color="black", ls="--", lw=1.4,
               label=f"Nominal {1-ALPHA:.2f}")
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.85, 1.03)
    ax.set_title(f"Per-class coverage (α = {ALPHA}, {SPLIT})")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "06b_scheme_coverage.png")
    plt.close(fig)


def fig_error_by_size(sets, correct):
    """Error rate within each set size."""
    base = (~correct).mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (name, s) in zip(axes, sets.items()):
        sizes = s.sum(axis=1)
        present = [v for v in range(N_CLASSES + 1) if (sizes == v).sum() > 0]
        rates = [(~correct[sizes == v]).mean() * 100 for v in present]
        counts = [int((sizes == v).sum()) for v in present]
        bars = ax.bar([str(v) for v in present], rates,
                      color=SCHEME_COLOURS[name], alpha=0.85)
        for b, r, c in zip(bars, rates, counts):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.4,
                    f"{r:.1f}%\n(n={c:,})", ha="center", fontsize=8)
        ax.axhline(base, color="#d62728", ls="--", lw=1.4,
                   label=f"Overall {base:.2f}%")
        ax.set_xlabel("Set size")
        ax.set_ylabel("Error rate (%)")
        ax.set_title(name.capitalize(), fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle(f"Does set size predict error? (α = {ALPHA}, {SPLIT})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06b_scheme_error_by_size.png")
    plt.close(fig)


def fig_risk_coverage(sets, correct):
    """Set size used directly as a deferral rule: defer the largest sets first."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for name, s in sets.items():
        sizes = s.sum(axis=1)
        pts = []
        # Accept only sets at or below each size: the natural nesting of the
        # signal, and the only ordering a policy on a discrete variable can use.
        for cut in range(N_CLASSES + 1):
            keep = sizes <= cut
            if keep.sum() == 0:
                continue
            pts.append((keep.mean() * 100, (~correct[keep]).mean() * 100, cut))
        if pts:
            xs, ys, cuts = zip(*pts)
            ax.plot(xs, ys, "o-", lw=2, ms=8, label=f"{name.capitalize()}",
                    color=SCHEME_COLOURS[name])
            for xv, yv, c in pts:
                # Stagger vertically: at high coverage the points nearly
                # coincide and fixed offsets overlap illegibly.
                ax.annotate(f"|C|≤{c}", (xv, yv), textcoords="offset points",
                            xytext=(8, 6 + 11 * c), fontsize=7.5,
                            color=SCHEME_COLOURS[name])

    probs = np.load(CACHE / f"{SPLIT}_probs.npy")
    conf = probs.max(axis=1)
    order = np.argsort(-conf)
    cs = correct[order]
    covs = np.linspace(0.3, 1.0, 60)
    ax.plot(covs * 100,
            [(~cs[:max(1, int(round(c * len(cs))))]).mean() * 100 for c in covs],
            lw=1.6, ls=":", color="#7f8c8d",
            label="Softmax confidence (reference, not a condition)")

    ax.axhline((~correct).mean() * 100, color="#d62728", ls="--", lw=1.2,
               label=f"No deferral: {(~correct).mean()*100:.2f}%")
    ax.set_xlabel("Coverage (% auto-diagnosed)")
    ax.set_ylabel("Error rate on accepted cases (%)")
    ax.set_title(f"Set size as a deferral signal (α = {ALPHA}, {SPLIT})\n"
                 "few points means few usable operating points")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "06b_scheme_risk_coverage.png")
    plt.close(fig)


# Main
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    probs, labels, correct, q_global, q_mond = load()
    sets = build_sets(probs, q_global, q_mond)

    print(f"α = {ALPHA}, split = {SPLIT}, n = {len(labels):,}, "
          f"accuracy {correct.mean():.4f}\n")

    print(f"{'class':>8} {'q̂ global':>10} {'t global':>9} "
          f"{'q̂ mond':>9} {'t mond':>8}")
    print("-" * 50)
    for k, c in enumerate(CLASSES):
        flag = "  <-- unconditional" if (1 - q_mond[k]) < 1e-6 else ""
        print(f"{c:>8} {q_global:>10.4f} {1-q_global:>9.4f} "
              f"{q_mond[k]:>9.4f} {1-q_mond[k]:>8.4f}{flag}")

    fig_thresholds(q_global, q_mond)
    fig_setsize(sets)
    fig_membership(sets)
    fig_coverage(sets, labels)
    fig_error_by_size(sets, correct)
    fig_risk_coverage(sets, correct)

    rows = []
    for name, s in sets.items():
        sizes = s.sum(axis=1)
        counts = np.bincount(sizes, minlength=N_CLASSES + 1)
        p = counts[counts > 0] / counts.sum()
        row = {
            "scheme": name,
            "alpha": ALPHA,
            "marginal_coverage": float(s[np.arange(len(labels)), labels].mean()),
            "mean_setsize": float(sizes.mean()),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "n_distinct_sizes": int((counts > 0).sum()),
        }
        for k, c in enumerate(CLASSES):
            row[f"cov_{c}"] = float(s[labels == k, k].mean())
            row[f"member_{c}"] = float(s[:, k].mean())
            row[f"threshold_{c}"] = float(1 - (q_global if name == "global"
                                               else q_mond[k]))
        for v in range(N_CLASSES + 1):
            row[f"size_{v}_frac"] = float(counts[v] / counts.sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "06b_scheme_comparison.csv", index=False)

    print("\n" + "=" * 74)
    for name, s in sets.items():
        sizes = s.sum(axis=1)
        counts = np.bincount(sizes, minlength=N_CLASSES + 1)
        print(f"\n{name.upper()}")
        print(f"  marginal coverage   {s[np.arange(len(labels)), labels].mean():.4f}")
        print(f"  mean set size       {sizes.mean():.4f}")
        print(f"  distinct sizes      {(counts > 0).sum()}  "
              f"(sizes present: {[v for v in range(5) if counts[v] > 0]})")
        print("  per class:")
        for k, c in enumerate(CLASSES):
            t = 1 - (q_global if name == "global" else q_mond[k])
            flag = "  <-- in every set" if s[:, k].mean() > 0.999 else ""
            print(f"    {c:<7} coverage {s[labels == k, k].mean():.4f}   "
                  f"appears in {s[:, k].mean()*100:5.1f}% of sets   "
                  f"t={t:.4f}{flag}")

    print(f"\nFigures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("06b_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
