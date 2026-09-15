"""Plot training metrics and the recorded final-epoch validation confusion matrix."""

from paths import PROJECT, read_csv


import matplotlib.pyplot as plt
import numpy as np

RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "03_training"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

CONFUSION = np.array([
    [2192,    3,   32,    1],
    [   7,  661,    0,   12],
    [  27,    0,  476,   13],
    [   1,    9,    5, 1561],
])

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


def plot_loss(log):
    """Train vs validation loss."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log["epoch"], log["train_loss"], "o-", label="Train", lw=1.8, ms=4)
    ax.plot(log["epoch"], log["val_loss"], "s-", label="Validation", lw=1.8, ms=4)

    # Mark the minimum: past this point the model fits the training set at the
    # expense of calibration.
    best_ep = int(log.loc[log["val_loss"].idxmin(), "epoch"])
    ax.axvline(best_ep, color="grey", ls="--", lw=1, alpha=0.7)
    ax.annotate(f"min val loss\n(epoch {best_ep})",
                xy=(best_ep, log["val_loss"].min()),
                xytext=(best_ep + 1.5, log["val_loss"].min() + 0.02),
                fontsize=8, color="grey")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    fig.savefig(FIGDIR / "03_loss.png")
    plt.close(fig)


def plot_macro_f1(log):
    """Validation macro-F1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log["epoch"], log["val_macro_f1"], "o-", color="#2c3e50", lw=1.8, ms=4)

    best_idx = log["val_macro_f1"].idxmax()
    best_ep = int(log.loc[best_idx, "epoch"])
    best_f1 = log.loc[best_idx, "val_macro_f1"]
    ax.scatter([best_ep], [best_f1], s=90, facecolors="none",
               edgecolors="#e74c3c", lw=2, zorder=5)
    ax.annotate(f"best {best_f1:.4f}\n(epoch {best_ep})",
                xy=(best_ep, best_f1), xytext=(best_ep - 5, best_f1 - 0.008),
                fontsize=8, color="#e74c3c")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Validation macro-F1")
    fig.savefig(FIGDIR / "03_macro_f1.png")
    plt.close(fig)


def plot_per_class_f1(log):
    """Per-class F1."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls in CLASSES:
        ax.plot(log["epoch"], log[f"f1_{cls}"], "o-", label=cls,
                color=CLASS_COLOURS[cls], lw=1.8, ms=3.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("Per-class validation F1")
    ax.legend(loc="lower right", ncol=2)
    fig.savefig(FIGDIR / "03_per_class_f1.png")
    plt.close(fig)


def plot_confusion():
    """Plot the recorded final-epoch validation confusion matrix, normalised by row."""
    cm = CONFUSION
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation confusion matrix\n(row-normalised)")
    ax.grid(False)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            colour = "white" if cm_norm[i, j] > 0.5 else "#333333"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]:,})",
                    ha="center", va="center", color=colour, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, label="Proportion of true class")
    fig.savefig(FIGDIR / "03_confusion.png")
    plt.close(fig)


def main():
    # parents=True creates results/figures/ as well if it is missing.
    FIGDIR.mkdir(parents=True, exist_ok=True)
    log = read_csv(RESULTS / "train_log.csv")

    print(f"Loaded {len(log)} epochs from train_log.csv\n")

    plot_loss(log)
    plot_macro_f1(log)
    plot_per_class_f1(log)
    plot_confusion()

    print(f"Figures written to {FIGDIR.relative_to(PROJECT)}/:")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")

    # Numbers worth having to hand when writing the methods section.
    best_idx = log["val_macro_f1"].idxmax()
    total_errors = CONFUSION.sum() - np.trace(CONFUSION)
    cnv_drusen = CONFUSION[0, 2] + CONFUSION[2, 0]

    print(f"\nBest macro-F1  {log.loc[best_idx, 'val_macro_f1']:.4f} "
          f"at epoch {int(log.loc[best_idx, 'epoch'])}")
    print(f"Min val loss   {log['val_loss'].min():.4f} "
          f"at epoch {int(log.loc[log['val_loss'].idxmin(), 'epoch'])}")
    print(f"Final val loss {log['val_loss'].iloc[-1]:.4f}  "
          f"(rose {log['val_loss'].iloc[-1] - log['val_loss'].min():.4f} "
          f"from its minimum)")
    print(f"Total training time {log['minutes'].sum():.0f} min")
    print(f"\nValidation errors: {total_errors} total, "
          f"{cnv_drusen} of them CNV<->DRUSEN "
          f"({cnv_drusen / total_errors * 100:.0f}%)")


if __name__ == "__main__":
    main()
