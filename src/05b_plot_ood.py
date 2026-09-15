"""Plot OOD-score diagnostics on the Kermany splits.

Error-detection AUROC here separates correct and incorrect predictions.
External-source OOD AUROC is computed in steps 08c and 13c.
"""

from paths import PROJECT


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

CACHE = PROJECT / "cache"
RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "05_ood"
TABDIR = RESULTS / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_STAGES = 6
ALL_SPLITS = ["train", "val", "calibration", "test"]

# Splits carrying softmax and MC Dropout output. train has features only, so it
# appears in the geometry figures but not in anything involving predictions.
DECISION_SPLITS = ["val", "calibration", "test"]

# Diagnostics are read off validation: it is the largest split whose signals are
# not consumed by any fitted component, and test must stay untouched until step 08.
REF = "val"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CLASS_COLOURS = {
    "CNV": "#1f77b4",
    "DME": "#ff7f0e",
    "DRUSEN": "#d62728",
    "NORMAL": "#2ca02c",
}
COLOUR_LIST = [CLASS_COLOURS[c] for c in CLASSES]
SPLIT_COLOURS = {"train": "#7f8c8d", "val": "#1f77b4",
                 "calibration": "#ff7f0e", "test": "#d62728"}


# Loading
def load_split(split):
    """Load cached scores; prediction-derived signals are omitted for training data."""
    d = {
        "labels": np.load(CACHE / f"{split}_labels.npy"),
        "cosine": np.load(CACHE / f"{split}_ood_cosine.npy"),
        "maha": np.load(CACHE / f"{split}_ood_maha_combined.npy"),
        "maha_z": [np.load(CACHE / f"{split}_ood_maha_z_stage{s}.npy")
                   for s in range(1, N_STAGES + 1)],
    }
    if split in DECISION_SPLITS:
        probs = np.load(CACHE / f"{split}_probs.npy")
        mc = np.load(CACHE / f"{split}_mc_probs.npy")
        mc_mean = mc.mean(axis=1)
        eps = 1e-12
        pred_ent = -(mc_mean * np.log(mc_mean + eps)).sum(axis=1)
        mean_ent = -(mc * np.log(mc + eps)).sum(axis=2).mean(axis=1)

        d["confidence"] = probs.max(axis=1)
        d["prediction"] = probs.argmax(axis=1)
        d["correct"] = probs.argmax(axis=1) == d["labels"]
        d["mutual_info"] = pred_ent - mean_ent
    return d


def error_auroc(score, correct, higher_means_ood=False):
    """Probability that a randomly drawn error scores as more suspect than a randomly drawn correct prediction."""
    s = score if higher_means_ood else -score
    return roc_auc_score(~correct, s)


# Figures
def fig_cosine_distribution(data):
    """Cosine score by split."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in ALL_SPLITS:
        ax.hist(data[split]["cosine"], bins=70, density=True, alpha=0.45,
                label=f"{split} (μ={data[split]['cosine'].mean():.3f})",
                color=SPLIT_COLOURS[split])
    ax.set_xlabel("Cosine similarity to nearest class centroid (stage 6)")
    ax.set_ylabel("Density")
    ax.set_title("Cosine OOD score by split\nhigher = more in-distribution")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "05_cosine_distribution.png")
    plt.close(fig)


def fig_maha_distribution(data):
    """Combined Mahalanobis z-score by split."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in ALL_SPLITS:
        v = data[split]["maha"]
        ax.hist(v, bins=70, density=True, alpha=0.45,
                label=f"{split} (μ={v.mean():+.3f}, σ={v.std():.3f})",
                color=SPLIT_COLOURS[split])
    ax.axvline(0, color="black", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Mahalanobis z-score, mean of six depths")
    ax.set_ylabel("Density")
    ax.set_title("Mahalanobis OOD score by split\n"
                 "higher = closer to the training distribution")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "05_maha_distribution.png")
    plt.close(fig)


def fig_maha_by_depth(data):
    """Per-depth z-score distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for s, ax in enumerate(axes.flat, start=1):
        for split in ALL_SPLITS:
            v = data[split]["maha_z"][s - 1]
            ax.hist(v, bins=60, density=True, alpha=0.42,
                    label=split if s == 1 else None, color=SPLIT_COLOURS[split])
        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        shift = data["test"]["maha_z"][s - 1].mean()
        ax.set_title(f"Stage {s}   (test shift {shift:+.3f})", fontsize=10)
        ax.set_xlabel("z-score")
    fig.legend(loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Mahalanobis z-score by depth", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIGDIR / "05_maha_by_depth.png")
    plt.close(fig)


def fig_scores_by_class(data, split=REF):
    """Both detectors, split by true class."""
    d = data[split]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ["cosine", "maha"],
                              ["Cosine similarity", "Mahalanobis z-score"]):
        groups = [d[key][d["labels"] == k] for k in range(len(CLASSES))]
        bp = ax.boxplot(groups, tick_labels=CLASSES, patch_artist=True,
                        showfliers=False)
        for patch, colour in zip(bp["boxes"], COLOUR_LIST):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by true class", fontsize=10)
    fig.suptitle(f"OOD scores by diagnostic class — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_scores_by_class.png")
    plt.close(fig)


def fig_error_separation(data, split=REF):
    """Correct against wrong predictions, for both detectors."""
    d = data[split]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in zip(axes, ["cosine", "maha"],
                              ["Cosine similarity", "Mahalanobis z-score"]):
        corr, err = d[key][d["correct"]], d[key][~d["correct"]]
        bins = np.linspace(d[key].min(), d[key].max(), 55)
        ax.hist(corr, bins=bins, density=True, alpha=0.6,
                label=f"Correct (n={len(corr):,})", color="#2ca02c")
        ax.hist(err, bins=bins, density=True, alpha=0.6,
                label=f"Wrong (n={len(err):,})", color="#d62728")
        auc = error_auroc(d[key], d["correct"])
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(f"{title}\nerror-detection AUROC {auc:.3f}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(f"Do OOD scores also flag misclassification? — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_error_separation.png")
    plt.close(fig)


def fig_detector_scatter(data, split=REF):
    """Cosine against Mahalanobis, errors drawn on top."""
    d = data[split]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ok, bad = d["correct"], ~d["correct"]
    ax.scatter(d["cosine"][ok], d["maha"][ok], s=4, alpha=0.22,
               c="#2ca02c", label="Correct", edgecolors="none")
    ax.scatter(d["cosine"][bad], d["maha"][bad], s=16, alpha=0.75,
               c="#d62728", label="Wrong", edgecolors="none")
    rho = spearmanr(d["cosine"], d["maha"]).statistic
    ax.set_xlabel("Cosine similarity (stage 6)")
    ax.set_ylabel("Mahalanobis z-score (six depths)")
    ax.set_title(f"Are the two detectors redundant? — {split}\n"
                 f"Spearman ρ = {rho:.3f}")
    ax.legend(markerscale=2)
    fig.savefig(FIGDIR / "05_detector_scatter.png")
    plt.close(fig)
    return rho


def fig_depth_correlation(data, split=REF):
    """Plot Pearson correlations between the six Mahalanobis depths."""
    z = np.stack(data[split]["maha_z"])
    mat = np.corrcoef(z)

    fig, ax = plt.subplots(figsize=(5.8, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [f"S{i}" for i in range(1, N_STAGES + 1)]
    ax.set_xticks(range(N_STAGES), names)
    ax.set_yticks(range(N_STAGES), names)
    ax.grid(False)
    for i in range(N_STAGES):
        for j in range(N_STAGES):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 0.6 else "#333333",
                    fontsize=8)
    off = mat[~np.eye(N_STAGES, dtype=bool)]
    ax.set_title(f"Correlation between Mahalanobis depths — {split}\n"
                 f"mean off-diagonal {off.mean():.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIGDIR / "05_depth_correlation.png")
    plt.close(fig)

    pd.DataFrame(mat, index=names, columns=names).to_csv(
        TABDIR / "05b_depth_correlation.csv")
    return mat


def fig_depth_auroc(data, split=REF):
    """Error-detection AUROC for each depth individually."""
    d = data[split]
    aucs = [error_auroc(d["maha_z"][s - 1], d["correct"])
            for s in range(1, N_STAGES + 1)]
    combined = error_auroc(d["maha"], d["correct"])
    cos_auc = error_auroc(d["cosine"], d["correct"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(1, N_STAGES + 1), aucs, color="#2c3e50", alpha=0.85)
    ax.axhline(combined, color="#d62728", ls="--", lw=1.5,
               label=f"Six-depth mean: {combined:.3f}")
    ax.axhline(cos_auc, color="#1f77b4", ls=":", lw=1.5,
               label=f"Cosine (stage 6): {cos_auc:.3f}")
    ax.axhline(0.5, color="grey", lw=1, alpha=0.6, label="Chance")
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("Mahalanobis depth")
    ax.set_ylabel("Error-detection AUROC")
    ax.set_ylim(0.4, max(max(aucs), combined, cos_auc) + 0.06)
    ax.set_title(f"Per-depth error detection — {split}")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGDIR / "05_depth_auroc.png")
    plt.close(fig)
    return aucs, combined, cos_auc


def fig_signal_correlation(data, split=REF):
    """All four candidate policy signals, pairwise."""
    d = data[split]
    sig = {
        "OOD: cosine": -d["cosine"],
        "OOD: Mahalanobis": -d["maha"],
        "MC: mutual info": d["mutual_info"],
        "Softmax: 1−conf": 1 - d["confidence"],
    }
    names = list(sig)
    mat = np.array([[spearmanr(sig[a], sig[b]).statistic for b in names]
                    for a in names])

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 0.6 else "#333333",
                    fontsize=9)
    ax.set_title(f"Candidate policy signals — {split}\n"
                 "all oriented so higher = more suspect")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIGDIR / "05_signal_correlation.png")
    plt.close(fig)

    pd.DataFrame(mat, index=names, columns=names).to_csv(
        TABDIR / "05b_signal_correlation.csv")
    return names, mat


def fig_signal_pairplot(data, split=REF):
    """Pairwise scatter matrix of the four signals."""
    d = data[split]
    sig = {
        "cosine": -d["cosine"],
        "Mahalanobis": -d["maha"],
        "mutual info": d["mutual_info"],
        "1−confidence": 1 - d["confidence"],
    }
    names = list(sig)
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(12, 11))
    ok, bad = d["correct"], ~d["correct"]

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(sig[names[i]], bins=50, color="#2c3e50", alpha=0.8)
                ax.set_yticks([])
            else:
                ax.scatter(sig[names[j]][ok], sig[names[i]][ok], s=2,
                           alpha=0.15, c="#2ca02c", edgecolors="none")
                ax.scatter(sig[names[j]][bad], sig[names[i]][bad], s=9,
                           alpha=0.7, c="#d62728", edgecolors="none")
            if i == n - 1:
                ax.set_xlabel(names[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(f"Signal pairwise structure — {split}   "
                 "(green correct, red wrong)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / "05_signal_pairplot.png")
    plt.close(fig)


def fig_score_vs_confidence(data, split=REF):
    """Each OOD score against softmax confidence."""
    d = data[split]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ok, bad = d["correct"], ~d["correct"]
    for ax, key, title in zip(axes, ["cosine", "maha"],
                              ["Cosine similarity", "Mahalanobis z-score"]):
        ax.scatter(d["confidence"][ok], d[key][ok], s=3, alpha=0.18,
                   c="#2ca02c", label="Correct", edgecolors="none")
        ax.scatter(d["confidence"][bad], d[key][bad], s=14, alpha=0.75,
                   c="#d62728", label="Wrong", edgecolors="none")
        rho = spearmanr(d["confidence"], d[key]).statistic
        ax.set_xlabel("Softmax confidence")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\nSpearman ρ = {rho:.3f}", fontsize=10)
        ax.legend(fontsize=8, markerscale=2)
    fig.suptitle(f"Do the OOD detectors restate softmax confidence? — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_score_vs_confidence.png")
    plt.close(fig)


def fig_split_shift(data):
    """Shift of each split relative to train, per depth and for both detectors."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    x = np.arange(1, N_STAGES + 1)
    width = 0.25
    for i, split in enumerate(["val", "calibration", "test"]):
        shifts = [data[split]["maha_z"][s - 1].mean() for s in range(1, N_STAGES + 1)]
        ax.bar(x + (i - 1) * width, shifts, width, label=split,
               color=SPLIT_COLOURS[split], alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [f"S{s}" for s in x])
    ax.set_xlabel("Mahalanobis depth")
    ax.set_ylabel("Mean z-score (train = 0)")
    ax.set_title("Shift by depth", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    base = data["train"]["cosine"].mean()
    for i, split in enumerate(["val", "calibration", "test"]):
        ax.bar(i, data[split]["cosine"].mean() - base,
               color=SPLIT_COLOURS[split], alpha=0.85, width=0.6)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(3), ["val", "calibration", "test"])
    ax.set_ylabel("Cosine score, difference from train")
    ax.set_title("Cosine shift", fontsize=10)

    fig.suptitle("Departure from the training distribution, by split")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_split_shift.png")
    plt.close(fig)


def fig_cdf_comparison(data):
    """Empirical CDFs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, key, title in zip(axes, ["cosine", "maha"],
                              ["Cosine similarity", "Mahalanobis z-score"]):
        for split in ALL_SPLITS:
            v = np.sort(data[split][key])
            ax.plot(v, np.linspace(0, 1, len(v)), lw=1.6,
                    label=split, color=SPLIT_COLOURS[split])
        ax.set_xlabel(title)
        ax.set_ylabel("Empirical CDF")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Cumulative distributions — tail behaviour is the operating region")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_cdf_comparison.png")
    plt.close(fig)


def fig_depth_by_class(data, split=REF):
    """Per-depth z-score by true class."""
    d = data[split]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for s, ax in enumerate(axes.flat, start=1):
        groups = [d["maha_z"][s - 1][d["labels"] == k] for k in range(len(CLASSES))]
        bp = ax.boxplot(groups, tick_labels=CLASSES, patch_artist=True,
                        showfliers=False)
        for patch, colour in zip(bp["boxes"], COLOUR_LIST):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(f"Stage {s}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Mahalanobis z-score by depth and class — {split}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_depth_by_class.png")
    plt.close(fig)


def fig_error_rate_by_decile(data, split=REF):
    """Observed error rate within each decile of each signal."""
    d = data[split]
    sig = {
        "OOD: cosine": -d["cosine"],
        "OOD: Mahalanobis": -d["maha"],
        "MC: mutual info": d["mutual_info"],
        "Softmax: 1−conf": 1 - d["confidence"],
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, v in sig.items():
        edges = np.percentile(v, np.linspace(0, 100, 11))
        edges[-1] += 1e-9
        idx = np.clip(np.digitize(v, edges) - 1, 0, 9)
        rates = [((~d["correct"])[idx == b].mean() * 100
                  if (idx == b).any() else np.nan) for b in range(10)]
        ax.plot(range(1, 11), rates, "o-", lw=1.8, ms=4, label=name)

    ax.set_xlabel("Decile of signal (1 = least suspect, 10 = most)")
    ax.set_ylabel("Error rate within decile (%)")
    ax.set_xticks(range(1, 11))
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title(f"Error rate against signal decile — {split}")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "05_error_rate_by_decile.png")
    plt.close(fig)


def fig_coverage_risk_preview(data, split=REF):
    """Risk-coverage curve for each signal used alone as a deferral rule."""
    d = data[split]
    sig = {
        "OOD: cosine": -d["cosine"],
        "OOD: Mahalanobis": -d["maha"],
        "MC: mutual info": d["mutual_info"],
        "Softmax: 1−conf": 1 - d["confidence"],
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    coverages = np.linspace(0.30, 1.0, 60)

    for name, v in sig.items():
        order = np.argsort(v)
        correct_sorted = d["correct"][order]
        risks = []
        for c in coverages:
            n_keep = max(1, int(round(c * len(v))))
            risks.append((~correct_sorted[:n_keep]).mean() * 100)
        ax.plot(coverages * 100, risks, lw=1.8, label=name)

    base = (~d["correct"]).mean() * 100
    ax.axhline(base, color="grey", ls="--", lw=1,
               label=f"No deferral: {base:.2f}%")
    ax.set_xlabel("Coverage (% of cases auto-diagnosed)")
    ax.set_ylabel("Error rate on accepted cases (%)")
    ax.set_title(f"Risk–coverage, single signals with a threshold — {split}\n"
                 "preview of step 08; in-distribution only")
    ax.legend(fontsize=8)
    fig.savefig(FIGDIR / "05_coverage_risk_preview.png")
    plt.close(fig)


def fig_score_qq(data):
    """Quantile-quantile plots of each split against train."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    q = np.linspace(1, 99, 99)
    for ax, key, title in zip(axes, ["cosine", "maha"],
                              ["Cosine similarity", "Mahalanobis z-score"]):
        ref = np.percentile(data["train"][key], q)
        for split in ["val", "calibration", "test"]:
            ax.plot(ref, np.percentile(data[split][key], q), "o", ms=3,
                    alpha=0.7, label=split, color=SPLIT_COLOURS[split])
        lo, hi = ref.min(), ref.max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="identical")
        ax.set_xlabel(f"train quantiles — {title}")
        ax.set_ylabel("split quantiles")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Q–Q against train")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_score_qq.png")
    plt.close(fig)


# Main
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached OOD scores...")
    data = {s: load_split(s) for s in ALL_SPLITS}
    for s in ALL_SPLITS:
        n = len(data[s]["labels"])
        extra = (f", accuracy {data[s]['correct'].mean():.4f}"
                 if s in DECISION_SPLITS else ", features only")
        print(f"  {s:12s} {n:>7,}{extra}")

    print(f"\nGenerating figures from '{REF}'...")

    print("  distributions...")
    fig_cosine_distribution(data)
    fig_maha_distribution(data)
    fig_maha_by_depth(data)
    fig_cdf_comparison(data)
    fig_score_qq(data)
    fig_split_shift(data)

    print("  class structure...")
    fig_scores_by_class(data)
    fig_depth_by_class(data)

    print("  error relationships...")
    fig_error_separation(data)
    fig_error_rate_by_decile(data)
    fig_coverage_risk_preview(data)

    print("  detector and signal structure...")
    rho_det = fig_detector_scatter(data)
    depth_corr = fig_depth_correlation(data)
    depth_aucs, maha_auc, cos_auc = fig_depth_auroc(data)
    sig_names, sig_corr = fig_signal_correlation(data)
    fig_signal_pairplot(data)
    fig_score_vs_confidence(data)

    # Tables
    rows = []
    for s in ALL_SPLITS:
        d = data[s]
        row = {"split": s, "n": len(d["labels"]),
               "cosine_mean": d["cosine"].mean(), "cosine_std": d["cosine"].std(),
               "maha_mean": d["maha"].mean(), "maha_std": d["maha"].std()}
        for st in range(1, N_STAGES + 1):
            row[f"maha_z_stage{st}_mean"] = d["maha_z"][st - 1].mean()
        if s in DECISION_SPLITS:
            row["accuracy"] = d["correct"].mean()
            row["cosine_error_auroc"] = error_auroc(d["cosine"], d["correct"])
            row["maha_error_auroc"] = error_auroc(d["maha"], d["correct"])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(TABDIR / "05b_score_summary.csv", index=False)

    auc_rows = [{"signal": f"Mahalanobis stage {s}", "error_auroc": depth_aucs[s - 1]}
                for s in range(1, N_STAGES + 1)]
    auc_rows += [
        {"signal": "Mahalanobis six-depth mean", "error_auroc": maha_auc},
        {"signal": "Cosine (stage 6)", "error_auroc": cos_auc},
        {"signal": "Mutual information", "error_auroc":
            error_auroc(data[REF]["mutual_info"], data[REF]["correct"], True)},
        {"signal": "Softmax confidence", "error_auroc":
            error_auroc(data[REF]["confidence"], data[REF]["correct"])},
    ]
    pd.DataFrame(auc_rows).to_csv(TABDIR / "05b_error_auroc.csv", index=False)

    cls_rows = []
    for s in ALL_SPLITS:
        d = data[s]
        for k, cname in enumerate(CLASSES):
            m = d["labels"] == k
            cls_rows.append({
                "split": s, "class": cname, "n": int(m.sum()),
                "cosine_mean": d["cosine"][m].mean(),
                "maha_mean": d["maha"][m].mean(),
            })
    pd.DataFrame(cls_rows).to_csv(TABDIR / "05b_class_scores.csv", index=False)

    # Report
    print("\n" + "=" * 74)
    cols = ["split", "n", "cosine_mean", "cosine_std", "maha_mean", "maha_std"]
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"))

    print(f"\nDetector redundancy: cosine vs Mahalanobis  ρ = {rho_det:+.3f}")
    off = depth_corr[~np.eye(N_STAGES, dtype=bool)]
    print(f"Mahalanobis depths:  mean off-diagonal ρ = {off.mean():.3f} "
          f"(range {off.min():.3f} to {off.max():.3f})")

    print(f"\nError-detection AUROC on '{REF}':")
    for r in auc_rows:
        flag = "  <-- at chance" if abs(r["error_auroc"] - 0.5) < 0.03 else ""
        print(f"  {r['signal']:30s} {r['error_auroc']:.3f}{flag}")

    print("\nSignal correlations (Spearman, all oriented higher = more suspect):")
    for i in range(len(sig_names)):
        for j in range(i + 1, len(sig_names)):
            flag = "  <-- redundant" if abs(sig_corr[i, j]) > 0.9 else ""
            print(f"  {sig_names[i]:20s} vs {sig_names[j]:20s} "
                  f"{sig_corr[i, j]:+.3f}{flag}")

    print(f"\nShift of test relative to train:")
    print(f"  cosine  {data['test']['cosine'].mean() - data['train']['cosine'].mean():+.4f}")
    print(f"  maha z  {data['test']['maha'].mean():+.4f}")

    print(f"\nFigures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("05b_*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
