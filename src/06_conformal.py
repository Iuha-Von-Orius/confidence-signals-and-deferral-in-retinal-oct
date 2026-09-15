"""Compare global and class-conditional split conformal prediction.

Quantiles are fitted on calibration probabilities. The alpha sweep and
entropy-based plot are diagnostics; the final policy uses Mondrian sets
with alpha = 0.02. Writes conformal parameters and cached prediction sets.
"""

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

FIT_SPLIT = "calibration"
SELECT_SPLIT = "val"
SCORE_SPLITS = ["calibration", "val"]

ALPHAS = [0.10, 0.05, 0.02, 0.01, 0.005]

# Above this admission threshold the sum constraint permits only one class in
# the set, so |C(x)| reduces to an indicator on max softmax probability.
DEGENERACY_T = 0.5

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})

CLASS_COLOURS = {"CNV": "#1f77b4", "DME": "#ff7f0e",
                 "DRUSEN": "#d62728", "NORMAL": "#2ca02c"}
COLOUR_LIST = [CLASS_COLOURS[c] for c in CLASSES]
ALPHA_COLOURS = ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#fdd0a2"]


# Core
def nonconformity(probs, labels):
    """s(x, y_true) = 1 − p̂(y_true | x), evaluated at the *true* label."""
    return 1.0 - probs[np.arange(len(labels)), labels]


def conformal_quantile(scores, alpha):
    """Use the recorded ceil((n+1)*(1-alpha))/n level with method='higher'.

    Return infinity if the requested level exceeds one.
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level > 1:
        return np.inf
    return float(np.quantile(scores, level, method="higher"))


def fit_global(scores, alpha):
    return conformal_quantile(scores, alpha)


def fit_mondrian(scores, labels, alpha):
    """One quantile per class, each estimated only from that class's calibration samples."""
    return {k: conformal_quantile(scores[labels == k], alpha)
            for k in range(N_CLASSES)}


def build_sets_global(probs, q):
    """Boolean membership matrix (N, 4). A class is admitted when its nonconformity score falls at or below the threshold."""
    return (1.0 - probs) <= q


def build_sets_mondrian(probs, q_dict):
    """Admit each candidate class using its own calibration threshold."""
    out = np.zeros_like(probs, dtype=bool)
    for k in range(N_CLASSES):
        out[:, k] = (1.0 - probs[:, k]) <= q_dict[k]
    return out


def coverage_stats(sets, labels):
    """Marginal and per-class coverage, plus set-size summary."""
    covered = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)
    per_class = {CLASSES[k]: float(covered[labels == k].mean())
                 for k in range(N_CLASSES)}
    return {
        "coverage": float(covered.mean()),
        "per_class": per_class,
        "min_class_coverage": float(min(per_class.values())),
        "mean_size": float(sizes.mean()),
        "size_dist": {int(s): int((sizes == s).sum()) for s in range(N_CLASSES + 1)},
        "frac_empty": float((sizes == 0).mean()),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_multi": float((sizes >= 2).mean()),
        "size_entropy": size_entropy(sizes),
    }


def size_entropy(sizes):
    """Shannon entropy of the set-size distribution, in bits."""
    counts = np.bincount(sizes, minlength=N_CLASSES + 1)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# Figures
def fig_score_distribution(scores, labels, quantiles):
    """Calibration nonconformity scores, with each α's quantile marked."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax.hist(scores, bins=100, color="#2c3e50", alpha=0.85)
    ax.set_yscale("log")
    for (a, q), c in zip(quantiles.items(), ALPHA_COLOURS):
        if np.isfinite(q):
            ax.axvline(q, color=c, lw=1.8, label=f"α={a}: q̂={q:.4f}")
    ax.axvline(DEGENERACY_T, color="#d62728", ls="--", lw=2,
               label=f"q̂ = {DEGENERACY_T} (2-class floor)")
    ax.set_xlabel("Nonconformity score  s = 1 − p̂(y_true)")
    ax.set_ylabel("Count (log)")
    ax.set_title("Calibration scores and quantiles", fontsize=10)
    ax.legend(fontsize=7.5)

    for k, cls in enumerate(CLASSES):
        ax2.hist(scores[labels == k], bins=60, density=True, alpha=0.45,
                 label=f"{cls} (median {np.median(scores[labels == k]):.4f})",
                 color=CLASS_COLOURS[cls])
    ax2.set_yscale("log")
    ax2.set_xlabel("Nonconformity score")
    ax2.set_ylabel("Density (log)")
    ax2.set_title("By true class — the case for Mondrian", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle("Nonconformity scores on the calibration split")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_score_distribution.png")
    plt.close(fig)


def fig_quantile_vs_alpha(q_global, q_mondrian):
    """How the quantile and the admission threshold move with α."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a = list(q_global)

    ax.plot(a, [q_global[x] for x in a], "o-", lw=2, ms=6,
            color="#2c3e50", label="Global")
    for k, cls in enumerate(CLASSES):
        ax.plot(a, [q_mondrian[x][k] for x in a], "s--", lw=1.4, ms=4,
                alpha=0.8, color=CLASS_COLOURS[cls], label=cls)
    ax.axhline(DEGENERACY_T, color="#d62728", ls="--", lw=2,
               label="q̂ = 0.5 — 2-class floor")
    ax.set_xscale("log")
    ax.set_xlabel("α (log scale)")
    ax.set_ylabel("q̂")
    ax.set_title("Quantile against α", fontsize=10)
    ax.legend(fontsize=8)

    t = [1 - q_global[x] if np.isfinite(q_global[x]) else 0 for x in a]
    ax2.plot(a, t, "o-", lw=2, ms=6, color="#2c3e50")
    ax2.axhline(DEGENERACY_T, color="#d62728", ls="--", lw=2,
                label="t = 0.5")
    ax2.fill_between(a, DEGENERACY_T, 1.0, alpha=0.12, color="#d62728")
    ax2.text(a[0], 0.75, "degenerate:\nset size is\n1[conf ≥ t]",
             fontsize=8.5, color="#8b1a1a", va="center")
    ax2.set_xscale("log")
    ax2.set_xlabel("α (log scale)")
    ax2.set_ylabel("Admission threshold  t = 1 − q̂")
    ax2.set_title("Admission threshold against α", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle("Where the signal stops being a copy of softmax confidence")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_quantile_vs_alpha.png")
    plt.close(fig)


def fig_coverage_vs_alpha(records):
    """Empirical coverage against the nominal 1−α, on both splits."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, scheme in zip(axes, ["global", "mondrian"]):
        for split, marker in zip(SCORE_SPLITS, ["o-", "s--"]):
            sub = [r for r in records if r["scheme"] == scheme
                   and r["split"] == split]
            nominal = [1 - r["alpha"] for r in sub]
            ax.plot(nominal, [r["coverage"] for r in sub], marker, lw=2, ms=6,
                    label=f"{split} (marginal)")
            ax.plot(nominal, [r["min_class_coverage"] for r in sub], marker,
                    lw=1.2, ms=4, alpha=0.6,
                    label=f"{split} (worst class)")
        lo = min(1 - a for a in ALPHAS)
        ax.plot([lo, 1], [lo, 1], "k--", lw=1, alpha=0.6, label="nominal")
        ax.set_xlabel("Nominal coverage 1 − α")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"{scheme.capitalize()} threshold", fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle("Does coverage hold? Calibration is a check; val is the test")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_coverage_vs_alpha.png")
    plt.close(fig)


def fig_per_class_coverage(records, split=SELECT_SPLIT):
    """Per-class coverage under both schemes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, scheme in zip(axes, ["global", "mondrian"]):
        sub = sorted([r for r in records if r["scheme"] == scheme
                      and r["split"] == split], key=lambda r: -r["alpha"])
        x = np.arange(len(sub))
        width = 0.2
        for k, cls in enumerate(CLASSES):
            ax.bar(x + (k - 1.5) * width, [r["per_class"][cls] for r in sub],
                   width, label=cls, color=CLASS_COLOURS[cls], alpha=0.85)
        for i, r in enumerate(sub):
            ax.hlines(1 - r["alpha"], i - 0.45, i + 0.45,
                      color="black", ls="--", lw=1.4)
        ax.set_xticks(x, [f"α={r['alpha']}" for r in sub])
        ax.set_ylim(0.5, 1.02)
        ax.set_title(f"{scheme.capitalize()} threshold", fontsize=11)
        ax.set_ylabel("Empirical coverage")
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Per-class coverage — {split}   "
                 "(dashed line = nominal 1−α)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_per_class_coverage.png")
    plt.close(fig)


def fig_setsize_distribution(records, split=SELECT_SPLIT):
    """Set-size distribution at each α, both schemes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, scheme in zip(axes, ["global", "mondrian"]):
        sub = sorted([r for r in records if r["scheme"] == scheme
                      and r["split"] == split], key=lambda r: -r["alpha"])
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))
        shades = ["#d62728", "#2c3e50", "#4a7ba7", "#8fbcd4", "#cfe3ee"]
        for s in range(N_CLASSES + 1):
            frac = np.array([r["size_dist"].get(s, 0) / r["n"] for r in sub])
            ax.bar(x, frac, 0.6, bottom=bottom, label=f"|C| = {s}",
                   color=shades[s], alpha=0.9)
            bottom += frac
        ax.set_xticks(x, [f"α={r['alpha']}\nH={r['size_entropy']:.3f}"
                          for r in sub], fontsize=8)
        ax.set_ylabel("Proportion of images")
        ax.set_title(f"{scheme.capitalize()} threshold", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Set-size distribution — {split}   "
                 "(H = entropy in bits; near zero means a degenerate signal)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_setsize_distribution.png")
    plt.close(fig)


def fig_entropy_and_efficiency(records, split=SELECT_SPLIT):
    """Set-size entropy and mean set size against α."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for scheme, style in zip(["global", "mondrian"], ["o-", "s--"]):
        sub = sorted([r for r in records if r["scheme"] == scheme
                      and r["split"] == split], key=lambda r: r["alpha"])
        a = [r["alpha"] for r in sub]
        ax.plot(a, [r["size_entropy"] for r in sub], style, lw=2, ms=6,
                label=scheme)
        ax2.plot(a, [r["mean_size"] for r in sub], style, lw=2, ms=6,
                 label=scheme)
    ax.set_xscale("log"); ax.set_xlabel("α (log scale)")
    ax.set_ylabel("Set-size entropy (bits)")
    ax.set_title("Information content of the signal", fontsize=10)
    ax.legend(fontsize=8)
    ax2.set_xscale("log"); ax2.set_xlabel("α (log scale)")
    ax2.set_ylabel("Mean |C(x)|")
    ax2.set_title("Efficiency — smaller is better at equal coverage", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle(f"Choosing α — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_entropy_and_efficiency.png")
    plt.close(fig)


def fig_setsize_vs_confidence(probs, sizes, correct, alpha, split=SELECT_SPLIT):
    """Set size against softmax confidence, and the error rate within each size."""
    conf = probs.max(axis=1)
    present = sorted(set(sizes.tolist()))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for s in present:
        m = sizes == s
        if m.sum() == 0:
            continue
        ax.hist(conf[m], bins=50, alpha=0.55, label=f"|C|={s} (n={m.sum():,})")
    ax.set_yscale("log")
    ax.set_xlabel("Softmax confidence")
    ax.set_ylabel("Count (log)")
    ax.set_title("Confidence by set size", fontsize=10)
    ax.legend(fontsize=8)

    rates, labels_, counts = [], [], []
    for s in present:
        m = sizes == s
        if m.sum() == 0:
            continue
        rates.append((~correct[m]).mean() * 100)
        labels_.append(f"|C|={s}")
        counts.append(int(m.sum()))
    bars = ax2.bar(labels_, rates, color="#2c3e50", alpha=0.85)
    for bar, n in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax2.axhline((~correct).mean() * 100, color="#d62728", ls="--", lw=1.5,
                label=f"Overall {(~correct).mean()*100:.2f}%")
    ax2.set_ylabel("Error rate (%)")
    ax2.set_title("Does set size predict error?", fontsize=10)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Set size at α={alpha} — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_setsize_vs_confidence.png")
    plt.close(fig)


def fig_calibration_check(records):
    """Deviation of empirical from nominal coverage."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for scheme, style in zip(["global", "mondrian"], ["-", "--"]):
        for split, marker in zip(SCORE_SPLITS, ["o", "s"]):
            sub = sorted([r for r in records if r["scheme"] == scheme
                          and r["split"] == split], key=lambda r: r["alpha"])
            ax.plot([r["alpha"] for r in sub],
                    [(r["coverage"] - (1 - r["alpha"])) * 100 for r in sub],
                    marker + style, lw=1.8, ms=6, label=f"{scheme} / {split}")
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("α (log scale)")
    ax.set_ylabel("Empirical − nominal coverage (percentage points)")
    ax.set_title("Coverage deviation\n"
                 "calibration should be ~0 by construction; val is the real test")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "06_calibration_check.png")
    plt.close(fig)


def fig_mondrian_gain(records, split=SELECT_SPLIT):
    """What per-class calibration buys, per class and per α."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    alphas_sorted = sorted(ALPHAS, reverse=True)
    x = np.arange(len(alphas_sorted))
    width = 0.2
    for k, cls in enumerate(CLASSES):
        gains = []
        for a in alphas_sorted:
            g = next(r for r in records if r["scheme"] == "global"
                     and r["split"] == split and r["alpha"] == a)
            m = next(r for r in records if r["scheme"] == "mondrian"
                     and r["split"] == split and r["alpha"] == a)
            gains.append((m["per_class"][cls] - g["per_class"][cls]) * 100)
        ax.bar(x + (k - 1.5) * width, gains, width, label=cls,
               color=CLASS_COLOURS[cls], alpha=0.85)
    ax.axhline(0, color="black", lw=1.2)
    ax.set_xticks(x, [f"α={a}" for a in alphas_sorted])
    ax.set_ylabel("Coverage gain from Mondrian (percentage points)")
    ax.set_title(f"Mondrian minus global, per class — {split}")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIGDIR / "06_mondrian_gain.png")
    plt.close(fig)


# Main
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in SCORE_SPLITS:
        data[split] = {
            "probs": np.load(CACHE / f"{split}_probs.npy"),
            "labels": np.load(CACHE / f"{split}_labels.npy"),
        }
        d = data[split]
        d["correct"] = d["probs"].argmax(1) == d["labels"]
        print(f"{split:12s} {len(d['labels']):>6,}  "
              f"accuracy {d['correct'].mean():.4f}")

    cal = data[FIT_SPLIT]
    cal_scores = nonconformity(cal["probs"], cal["labels"])

    print(f"\nCalibration nonconformity scores, n = {len(cal_scores):,}")
    for p in [50, 75, 90, 95, 97.5, 99, 99.5]:
        print(f"  {p:>5.1f}th percentile: {np.percentile(cal_scores, p):.6f}")

    # Fit
    print(f"\n{'α':>7} {'q̂ global':>11} {'t = 1−q̂':>10}  per-class q̂")
    print("-" * 78)
    q_global, q_mondrian = {}, {}
    for a in ALPHAS:
        q_global[a] = fit_global(cal_scores, a)
        q_mondrian[a] = fit_mondrian(cal_scores, cal["labels"], a)
        t = 1 - q_global[a]
        flag = "  DEGENERATE" if t > DEGENERACY_T else "  2+ classes possible"
        per = "  ".join(f"{CLASSES[k][:3]} {q_mondrian[a][k]:.4f}"
                        for k in range(N_CLASSES))
        print(f"{a:>7} {q_global[a]:>11.6f} {t:>10.4f}  {per}{flag}")

    # Score
    records = []
    for a in ALPHAS:
        for split in SCORE_SPLITS:
            d = data[split]
            for scheme, sets in [
                ("global", build_sets_global(d["probs"], q_global[a])),
                ("mondrian", build_sets_mondrian(d["probs"], q_mondrian[a])),
            ]:
                stats = coverage_stats(sets, d["labels"])
                records.append({"alpha": a, "split": split, "scheme": scheme,
                                "n": len(d["labels"]), **stats})

                sizes = sets.sum(axis=1).astype(np.int8)
                tag = "" if scheme == "global" else "_mondrian"
                np.save(CACHE / f"{split}_cp_setsize{tag}_a{a}.npy", sizes)
                if scheme == "global":
                    np.save(CACHE / f"{split}_cp_sets_a{a}.npy", sets)

    with open(CACHE / "conformal_params.json", "w") as fh:
        json.dump({
            "alphas": ALPHAS,
            "fit_split": FIT_SPLIT,
            "n_calibration": len(cal_scores),
            "q_global": {str(a): q_global[a] for a in ALPHAS},
            "q_mondrian": {str(a): {CLASSES[k]: q_mondrian[a][k]
                                    for k in range(N_CLASSES)} for a in ALPHAS},
        }, fh, indent=2)

    # Figures
    print("\nGenerating figures...")
    fig_score_distribution(cal_scores, cal["labels"], q_global)
    fig_quantile_vs_alpha(q_global, q_mondrian)
    fig_coverage_vs_alpha(records)
    fig_per_class_coverage(records)
    fig_setsize_distribution(records)
    fig_entropy_and_efficiency(records)
    fig_calibration_check(records)
    fig_mondrian_gain(records)

    # The α with the most informative set-size distribution, chosen on the
    # selection split and on the global scheme.
    sel = max([r for r in records if r["split"] == SELECT_SPLIT
               and r["scheme"] == "global"], key=lambda r: r["size_entropy"])
    best_alpha = sel["alpha"]
    v = data[SELECT_SPLIT]
    fig_setsize_vs_confidence(
        v["probs"],
        build_sets_global(v["probs"], q_global[best_alpha]).sum(axis=1),
        v["correct"], best_alpha)

    # Tables
    rows = []
    for r in records:
        row = {k: r[k] for k in
               ["alpha", "split", "scheme", "n", "coverage",
                "min_class_coverage", "mean_size", "size_entropy",
                "frac_empty", "frac_singleton", "frac_multi"]}
        row["nominal"] = 1 - r["alpha"]
        row["deviation_pp"] = (r["coverage"] - (1 - r["alpha"])) * 100
        for c in CLASSES:
            row[f"cov_{c}"] = r["per_class"][c]
        for s in range(N_CLASSES + 1):
            row[f"size_{s}"] = r["size_dist"].get(s, 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "06_conformal_sweep.csv", index=False)

    pd.DataFrame([{"alpha": a, "q_global": q_global[a],
                   "admission_threshold": 1 - q_global[a],
                   "degenerate": bool(1 - q_global[a] > DEGENERACY_T),
                   **{f"q_{CLASSES[k]}": q_mondrian[a][k]
                      for k in range(N_CLASSES)}} for a in ALPHAS]
                 ).to_csv(TABDIR / "06_quantiles.csv", index=False)

    # Report
    print("\n" + "=" * 96)
    show = df[df.split == SELECT_SPLIT][
        ["alpha", "scheme", "nominal", "coverage", "deviation_pp",
         "min_class_coverage", "mean_size", "size_entropy",
         "frac_empty", "frac_multi"]]
    print(f"Selection split: {SELECT_SPLIT}\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nCalibration self-consistency (should be ~0 pp):")
    for r in [r for r in records if r["split"] == FIT_SPLIT]:
        dev = (r["coverage"] - (1 - r["alpha"])) * 100
        flag = "  <-- CHECK" if abs(dev) > 1.0 else ""
        print(f"  α={r['alpha']:<6} {r['scheme']:9s} {dev:+.2f} pp{flag}")

    print(f"\nDegeneracy: a set can hold two classes only when q̂ ≥ {DEGENERACY_T}")
    any_ok = False
    for a in ALPHAS:
        ok = q_global[a] >= DEGENERACY_T
        any_ok |= ok
        print(f"  α={a:<6} q̂={q_global[a]:.4f}  "
              f"{'yes' if ok else 'no — set size is 1[conf ≥ t]'}")
    if not any_ok:
        print("\nNo global quantile in the sweep permits multi-class prediction sets.")

    print(f"\nMaximum set-size entropy on {SELECT_SPLIT}: {best_alpha} "
          f"(set-size entropy {sel['size_entropy']:.4f} bits)")

    print(f"\nMondrian vs global, worst-class coverage on {SELECT_SPLIT}:")
    for a in ALPHAS:
        g = next(r for r in records if r["scheme"] == "global"
                 and r["split"] == SELECT_SPLIT and r["alpha"] == a)
        m = next(r for r in records if r["scheme"] == "mondrian"
                 and r["split"] == SELECT_SPLIT and r["alpha"] == a)
        worst_g = min(g["per_class"], key=g["per_class"].get)
        print(f"  α={a:<6} global {g['min_class_coverage']:.4f} ({worst_g})"
              f"   mondrian {m['min_class_coverage']:.4f}"
              f"   gain {(m['min_class_coverage']-g['min_class_coverage'])*100:+.2f} pp")

    print(f"\nFigures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("06_*.csv")):
        print(f"  {f.name}")
    print("\nFinal policy: Mondrian conformal sets at alpha = 0.02.")


if __name__ == "__main__":
    main()
