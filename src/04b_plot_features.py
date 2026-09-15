"""Plot cached feature, confidence and uncertainty diagnostics."""

from paths import PROJECT

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

CACHE = PROJECT / "cache"
RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "04_features"
TABDIR = RESULTS / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_STAGES = 6
DECISION_SPLITS = ["val", "calibration", "test"]

TSNE_SAMPLE = 3000
SEED = 42

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


# Loading
def load_split(split, with_mc=True):
    """Load every cached array for one split into a dict."""
    d = {
        "labels": np.load(CACHE / f"{split}_labels.npy"),
        "features": [np.load(CACHE / f"{split}_feat_stage{i}.npy")
                     for i in range(1, N_STAGES + 1)],
    }
    if with_mc:
        d["probs"] = np.load(CACHE / f"{split}_probs.npy")
        d["mc_probs"] = np.load(CACHE / f"{split}_mc_probs.npy")
    return d


# Signal computation
def compute_signals(d):
    """Derive every candidate decision signal from the cached arrays."""
    probs = d["probs"]
    mc = d["mc_probs"]

    mc_mean = mc.mean(axis=1)
    eps = 1e-12

    pred_entropy = -(mc_mean * np.log(mc_mean + eps)).sum(axis=1)
    mean_entropy = -(mc * np.log(mc + eps)).sum(axis=2).mean(axis=1)

    return {
        "confidence": probs.max(axis=1),
        "prediction": probs.argmax(axis=1),
        "mc_variance": mc.var(axis=1).mean(axis=1),
        "pred_entropy": pred_entropy,
        "mutual_info": pred_entropy - mean_entropy,
        "correct": probs.argmax(axis=1) == d["labels"],
    }


def separability(features, labels):
    """A Fisher-style ratio: mean distance between class centroids divided by mean within-class spread."""
    dim = features.shape[1]
    centroids = np.stack([features[labels == k].mean(axis=0)
                          for k in range(len(CLASSES))])

    between = [np.linalg.norm(centroids[i] - centroids[j])
               for i in range(len(CLASSES)) for j in range(i + 1, len(CLASSES))]

    within = [np.linalg.norm(features[labels == k] - centroids[k], axis=1).mean()
              for k in range(len(CLASSES))]

    return (np.mean(between) / np.mean(within)) / np.sqrt(dim) * np.sqrt(dim)


def expected_calibration_error(conf, correct, n_bins=10):
    """ECE: mean gap between stated confidence and observed accuracy, weighted by bin population."""
    edges = np.linspace(0, 1, n_bins + 1)
    accs, confs, counts = [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
        else:
            accs.append(correct[mask].mean())
            confs.append(conf[mask].mean())
            counts.append(int(mask.sum()))

    accs, confs, counts = np.array(accs), np.array(confs), np.array(counts)
    valid = counts > 0
    ece = np.sum(counts[valid] / counts.sum() * np.abs(accs[valid] - confs[valid]))
    return ece, edges, accs, confs, counts


# Figures
def fig_tsne_stage6(d, split):
    """Penultimate-layer feature space, projected to two dimensions."""
    rng = np.random.default_rng(SEED)
    feats, labels = d["features"][5], d["labels"]
    idx = rng.choice(len(feats), size=min(TSNE_SAMPLE, len(feats)), replace=False)

    emb = TSNE(n_components=2, random_state=SEED, init="pca",
               perplexity=30).fit_transform(feats[idx])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for k, cls in enumerate(CLASSES):
        m = labels[idx] == k
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.55,
                   c=CLASS_COLOURS[cls], label=cls, edgecolors="none")

    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Penultimate features (stage 6, 1280-d) — {split}\n"
                 f"{len(idx):,} sampled points")
    ax.legend(markerscale=3, loc="best")
    fig.savefig(FIGDIR / "04_tsne_stage6.png")
    plt.close(fig)


def fig_tsne_all_stages(d, split):
    """The same projection at all six depths."""
    rng = np.random.default_rng(SEED)
    labels = d["labels"]
    n = min(TSNE_SAMPLE, len(labels))
    idx = rng.choice(len(labels), size=n, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for s, ax in enumerate(axes.flat):
        emb = TSNE(n_components=2, random_state=SEED, init="pca",
                   perplexity=30).fit_transform(d["features"][s][idx])
        for k, cls in enumerate(CLASSES):
            m = labels[idx] == k
            ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.5,
                       c=CLASS_COLOURS[cls], label=cls if s == 0 else None,
                       edgecolors="none")
        ax.set_title(f"Stage {s+1}  ({d['features'][s].shape[1]}-d)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    fig.legend(loc="lower center", ncol=4, markerscale=4, frameon=False)
    fig.suptitle(f"Feature space by depth — {split}", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(FIGDIR / "04_tsne_all_stages.png")
    plt.close(fig)


def fig_uncertainty_hist(sig, split):
    """Distribution of the three MC Dropout uncertainty measures."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
            axes,
            ["mc_variance", "pred_entropy", "mutual_info"],
            ["MC variance", "Predictive entropy (total)",
             "Mutual information (epistemic)"]):
        ax.hist(sig[key], bins=60, color="#2c3e50", alpha=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(title); ax.set_ylabel("Count (log)")
        ax.set_title(f"{title}\nmedian {np.median(sig[key]):.5f}, "
                     f"p99 {np.percentile(sig[key], 99):.5f}", fontsize=9)
    fig.suptitle(f"MC Dropout uncertainty distributions — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_uncertainty_hist.png")
    plt.close(fig)


def fig_uncertainty_by_correct(sig, split):
    """Uncertainty split by whether the deterministic prediction was correct."""
    from sklearn.metrics import roc_auc_score

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
            axes,
            ["confidence", "mc_variance", "mutual_info"],
            ["Softmax confidence", "MC variance", "Mutual information"]):
        corr, err = sig[key][sig["correct"]], sig[key][~sig["correct"]]

        bins = np.linspace(min(sig[key].min(), 0), sig[key].max(), 50)
        ax.hist(corr, bins=bins, alpha=0.6, density=True,
                label=f"Correct (n={len(corr):,})", color="#2ca02c")
        ax.hist(err, bins=bins, alpha=0.6, density=True,
                label=f"Wrong (n={len(err):,})", color="#d62728")

        # For confidence, low means uncertain, so the sign is flipped to keep
        # AUROC interpretable as "higher score = more likely wrong".
        score = -sig[key] if key == "confidence" else sig[key]
        auc = roc_auc_score(~sig["correct"], score)

        ax.set_xlabel(title); ax.set_ylabel("Density")
        ax.set_title(f"{title}\nAUROC for error detection: {auc:.3f}", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(f"Does uncertainty flag errors? — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_uncertainty_by_correct.png")
    plt.close(fig)


def fig_uncertainty_by_class(sig, labels, split):
    """Uncertainty per true class."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, key, title in zip(axes, ["mutual_info", "confidence"],
                              ["Mutual information", "Softmax confidence"]):
        data = [sig[key][labels == k] for k in range(len(CLASSES))]
        # matplotlib >=3.9 renamed this argument from `labels` to `tick_labels`;
        # the old name was removed in 3.11.
        bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True,
                        showfliers=False)
        for patch, colour in zip(bp["boxes"], COLOUR_LIST):
            patch.set_facecolor(colour); patch.set_alpha(0.6)
        ax.set_ylabel(title)
        ax.set_title(f"{title} by true class", fontsize=10)

    fig.suptitle(f"Uncertainty by diagnostic class — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_uncertainty_by_class.png")
    plt.close(fig)


def fig_confidence_hist(sig, split):
    """Softmax confidence distribution."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(sig["confidence"], bins=60, color="#2c3e50", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Max softmax probability")
    ax.set_ylabel("Count (log)")

    frac99 = (sig["confidence"] > 0.99).mean()
    acc99 = sig["correct"][sig["confidence"] > 0.99].mean()
    ax.set_title(f"Softmax confidence — {split}\n"
                 f"{frac99*100:.1f}% of images above 0.99 "
                 f"(their actual accuracy: {acc99*100:.1f}%)")
    fig.savefig(FIGDIR / "04_confidence_hist.png")
    plt.close(fig)


def fig_layer_separability(d, split):
    """Class separability by depth."""
    scores = [separability(d["features"][s], d["labels"]) for s in range(N_STAGES)]
    dims = [d["features"][s].shape[1] for s in range(N_STAGES)]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(range(1, N_STAGES + 1), scores, color="#2c3e50", alpha=0.85)
    for bar, dim in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{dim}-d", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Stage"); ax.set_ylabel("Between-class / within-class distance")
    ax.set_title(f"Class separability by feature depth — {split}")
    fig.savefig(FIGDIR / "04_layer_separability.png")
    plt.close(fig)

    pd.DataFrame({"stage": range(1, N_STAGES + 1), "dim": dims,
                  "separability": scores}).to_csv(
        TABDIR / "04_layer_separability.csv", index=False)
    return scores


def fig_signal_scatter(sig, split):
    """Confidence against uncertainty, coloured by correctness."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ok, bad = sig["correct"], ~sig["correct"]
    ax.scatter(sig["confidence"][ok], sig["mutual_info"][ok], s=4, alpha=0.25,
               c="#2ca02c", label="Correct", edgecolors="none")
    ax.scatter(sig["confidence"][bad], sig["mutual_info"][bad], s=14, alpha=0.75,
               c="#d62728", label="Wrong", edgecolors="none")

    rho, _ = spearmanr(sig["confidence"], sig["mutual_info"])
    ax.set_xlabel("Softmax confidence")
    ax.set_ylabel("Mutual information (epistemic uncertainty)")
    ax.set_title(f"Confidence vs uncertainty — {split}\n"
                 f"Spearman rho = {rho:.3f}")
    ax.legend(markerscale=2)
    fig.savefig(FIGDIR / "04_signal_scatter.png")
    plt.close(fig)


def fig_signal_correlation(sig, split):
    """Spearman correlation between every candidate decision signal."""
    keys = ["confidence", "mc_variance", "pred_entropy", "mutual_info"]
    names = ["Confidence", "MC variance", "Pred. entropy", "Mutual info"]
    mat = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            mat[i, j] = spearmanr(sig[a], sig[b]).statistic

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(keys)), names)
    ax.grid(False)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 0.6 else "#333333",
                    fontsize=9)
    ax.set_title(f"Signal correlation (Spearman) — {split}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(FIGDIR / "04_signal_correlation.png")
    plt.close(fig)

    pd.DataFrame(mat, index=names, columns=names).to_csv(
        TABDIR / "04_signal_correlation.csv")
    return mat


def fig_reliability(sig, split):
    """Reliability diagram with ECE."""
    ece, edges, accs, confs, counts = expected_calibration_error(
        sig["confidence"], sig["correct"])
    centres = (edges[:-1] + edges[1:]) / 2

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.5, 6),
                                  gridspec_kw={"height_ratios": [3, 1]},
                                  sharex=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    valid = counts > 0
    ax.bar(centres[valid], accs[valid], width=0.09, alpha=0.75,
           color="#2c3e50", edgecolor="black", lw=0.5, label="Observed accuracy")
    ax.plot(confs[valid], accs[valid], "o-", color="#d62728", ms=5,
            label="Accuracy vs confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram — {split}\nECE = {ece:.4f}")
    ax.legend(fontsize=8, loc="upper left")

    ax2.bar(centres[valid], counts[valid], width=0.09, color="#7f8c8d", alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count (log)")

    fig.tight_layout()
    fig.savefig(FIGDIR / "04_reliability.png")
    plt.close(fig)
    return ece


def fig_feature_norms(d, split):
    """L2 norm of the feature vector at each depth, by class."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for s, ax in enumerate(axes.flat):
        norms = np.linalg.norm(d["features"][s], axis=1)
        data = [norms[d["labels"] == k] for k in range(len(CLASSES))]
        bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True,
                        showfliers=False)
        for patch, colour in zip(bp["boxes"], COLOUR_LIST):
            patch.set_facecolor(colour); patch.set_alpha(0.6)
        ax.set_title(f"Stage {s+1} ({d['features'][s].shape[1]}-d)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Feature L2 norm by depth and class — {split}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_feature_norms.png")
    plt.close(fig)


def fig_split_comparison(sigs):
    """Signal distributions across val, calibration and test."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
            axes, ["confidence", "mutual_info", "mc_variance"],
            ["Softmax confidence", "Mutual information", "MC variance"]):
        for split, colour in zip(DECISION_SPLITS,
                                 ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            ax.hist(sigs[split][key], bins=50, density=True, alpha=0.45,
                    label=split, color=colour)
        ax.set_yscale("log")
        ax.set_xlabel(title); ax.set_ylabel("Density (log)")
        ax.legend(fontsize=8)
    fig.suptitle("Signal distributions across splits — exchangeability check")
    fig.tight_layout()
    fig.savefig(FIGDIR / "04_split_comparison.png")
    plt.close(fig)


# Main
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached arrays...")
    data = {s: load_split(s) for s in DECISION_SPLITS}
    sigs = {s: compute_signals(data[s]) for s in DECISION_SPLITS}

    for s in DECISION_SPLITS:
        print(f"  {s:12s} {len(data[s]['labels']):>6,} images, "
              f"accuracy {sigs[s]['correct'].mean():.4f}")

    ref = "val"
    d, sig = data[ref], sigs[ref]

    print(f"\nGenerating figures from '{ref}'...")

    print("  t-SNE stage 6 (slow)...")
    fig_tsne_stage6(d, ref)
    print("  t-SNE all stages (slowest — six projections)...")
    fig_tsne_all_stages(d, ref)

    print("  uncertainty distributions...")
    fig_uncertainty_hist(sig, ref)
    fig_uncertainty_by_correct(sig, ref)
    fig_uncertainty_by_class(sig, d["labels"], ref)
    fig_confidence_hist(sig, ref)

    print("  feature diagnostics...")
    sep = fig_layer_separability(d, ref)
    fig_feature_norms(d, ref)

    print("  signal relationships...")
    fig_signal_scatter(sig, ref)
    corr = fig_signal_correlation(sig, ref)
    ece = fig_reliability(sig, ref)

    print("  split comparison...")
    fig_split_comparison(sigs)

    # Summary table
    rows = []
    for s in DECISION_SPLITS:
        g = sigs[s]
        e, *_ = expected_calibration_error(g["confidence"], g["correct"])
        rows.append({
            "split": s,
            "n": len(g["confidence"]),
            "accuracy": g["correct"].mean(),
            "ece": e,
            "mean_confidence": g["confidence"].mean(),
            "frac_conf_above_099": (g["confidence"] > 0.99).mean(),
            "mean_mc_variance": g["mc_variance"].mean(),
            "mean_pred_entropy": g["pred_entropy"].mean(),
            "mean_mutual_info": g["mutual_info"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABDIR / "04_signal_summary.csv", index=False)

    # Console report
    print("\n" + "=" * 62)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nECE on {ref}: {ece:.4f}")
    print(f"  {(sig['confidence'] > 0.99).mean()*100:.1f}% of images claim "
          f">0.99 confidence; their true accuracy is "
          f"{sig['correct'][sig['confidence'] > 0.99].mean()*100:.1f}%")

    print(f"\nSeparability by stage: " +
          "  ".join(f"S{i+1} {v:.2f}" for i, v in enumerate(sep)))
    print(f"Best-separating depth: stage {int(np.argmax(sep)) + 1}")

    print("\nSignal correlations (Spearman):")
    names = ["Confidence", "MC variance", "Pred. entropy", "Mutual info"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            flag = "  <-- highly redundant" if abs(corr[i, j]) > 0.9 else ""
            print(f"  {names[i]:14s} vs {names[j]:14s} {corr[i, j]:+.3f}{flag}")

    print(f"\nFigures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("04_*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
