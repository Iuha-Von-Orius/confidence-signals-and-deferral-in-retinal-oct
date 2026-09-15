"""Evaluate Kermany test, RETOUCH vendors and RASTI from per-image scores.

Report deferral, matched coverage and source-separation AUROC. RETOUCH
uses predicted NORMAL as a missed-disease error; labelled sources use
classification errors. Internal OCT5k labels refer to RASTI.
"""

from paths import PROJECT, read_csv


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TABDIR = PROJECT / "results" / "tables"
FIGDIR = PROJECT / "results" / "figures" / "08c_evaluate"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)
NORMAL_IDX = CLASSES.index("NORMAL")

COVERAGE_POINTS = [0.70, 0.80, 0.90]
COVERAGE_GRID = np.linspace(0.0, 1.0, 101)
CONDITIONS = [2, 3, 4, 5]

CONDITION_COLOURS = {1: "#7f8c8d", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd",
                     5: "#d62728", "B2": "#ff7f0e"}
CONDITION_LABELS = {1: "baseline", 2: "+OOD", 3: "+uncertainty",
                    4: "+conformal", 5: "+all signals", "B2": "softmax threshold"}

SOURCE_COLOURS = {"Kermany-test": "#2c3e50", "RETOUCH-Spectralis": "#d62728",
                  "RETOUCH-Topcon": "#1f77b4", "RETOUCH-Cirrus": "#2ca02c",
                  "OCT5k": "#9467bd"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})


# Source grouping
def add_source_column(df):
    """Assign Kermany test, RETOUCH vendor or RASTI source labels."""
    def source(row):
        if row["dataset"] == "Kermany-test":
            return "Kermany-test"
        if row["dataset"] == "RETOUCH":
            return f"RETOUCH-{row['vendor']}"
        return "OCT5k"
    df = df.copy()
    df["source"] = df.apply(source, axis=1)
    return df


# Physical-plausibility check
def plausibility_check(df):
    retouch = df[df.dataset == "RETOUCH"]
    means = retouch.groupby("vendor")[["ood_cosine", "ood_maha_combined"]].mean()
    print(means.to_string(float_format=lambda v: f"{v:.4f}"))

    spec_maha = means.loc["Spectralis", "ood_maha_combined"]
    topcon_maha = means.loc["Topcon", "ood_maha_combined"]
    cirrus_maha = means.loc["Cirrus", "ood_maha_combined"]
    spec_cos = means.loc["Spectralis", "ood_cosine"]
    topcon_cos = means.loc["Topcon", "ood_cosine"]
    cirrus_cos = means.loc["Cirrus", "ood_cosine"]

    maha_ok = spec_maha > topcon_maha and spec_maha > cirrus_maha
    cos_ok = spec_cos > topcon_cos and spec_cos > cirrus_cos

    print(f"\nMahalanobis-combined ordering (higher = more in-distribution): "
          f"Spectralis {spec_maha:.4f} vs Topcon {topcon_maha:.4f}, "
          f"Cirrus {cirrus_maha:.4f}  -- {'OK' if maha_ok else '[VIOLATED]'}")
    print(f"Cosine ordering: Spectralis {spec_cos:.4f} vs Topcon {topcon_cos:.4f}, "
          f"Cirrus {cirrus_cos:.4f}  -- {'OK' if cos_ok else '[note: violated]'}")

    return maha_ok, means


# Risk-coverage
def risk_coverage_curve(risk_score, correct, grid=COVERAGE_GRID):
    """Sort by increasing risk and interpolate selective risk onto the coverage grid.

    Retains the original sort-order treatment of ties for matched-coverage results.
    """
    order = np.argsort(risk_score)
    correct_sorted = correct[order]
    n = len(risk_score)
    cum_correct = np.concatenate([[0], np.cumsum(correct_sorted)])
    k = np.arange(n + 1)
    coverage = k / n
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(k > 0, 1.0 - cum_correct / np.maximum(k, 1), np.nan)
    if np.isnan(risk[0]) and len(risk) > 1:
        risk[0] = risk[1]
    return grid, np.interp(grid, coverage, risk)


def outcome_column(sub, source):
    """Use classification correctness for labelled data and non-NORMAL for RETOUCH."""
    if source == "RETOUCH-Spectralis" or source.startswith("RETOUCH"):
        # Missed-disease rate: a NORMAL prediction is the only confirmed
        # error, since every RETOUCH volume is from a diseased eye.
        return (sub["predicted_class"] != NORMAL_IDX).to_numpy()
    return (sub["predicted_class"] == sub["true_label"]).to_numpy()


def risk_score_for(sub, condition):
    if condition == "B2":
        return -sub["softmax_confidence"].to_numpy()
    if condition == 1:
        return np.zeros(len(sub))
    return sub[f"s_condition{condition}"].to_numpy()


# Confusion analyses
def pairwise_auroc(df, sources, signal_col):
    """AUROC for every ordered pair of sources, using -signal (higher = more OOD) as the score, source A as positive."""
    rows = []
    for a in sources:
        for b in sources:
            if a == b:
                continue
            sa = -df[df.source == a][signal_col].to_numpy()
            sb = -df[df.source == b][signal_col].to_numpy()
            y = np.concatenate([np.ones(len(sa)), np.zeros(len(sb))])
            s = np.concatenate([sa, sb])
            auroc = roc_auc_score(y, s)
            rows.append({"positive": a, "negative": b, "signal": signal_col,
                         "auroc": auroc})
    return pd.DataFrame(rows)


# Figures
def fig_risk_coverage_by_source(curves, source, baseline_risk):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for condition in CONDITIONS + ["B2"]:
        cov, risk = curves[(source, condition)]
        ax.plot(cov, risk, "-", lw=2.0, color=CONDITION_COLOURS[condition],
                label=f"{condition} {CONDITION_LABELS[condition]}")
    ax.axhline(baseline_risk, color=CONDITION_COLOURS[1], ls="--", lw=1.5,
               label=f"1 baseline ({baseline_risk:.4f})")
    ax.set_xlabel("Coverage")
    ylabel = "Missed-disease rate" if source.startswith("RETOUCH") else "Error rate"
    ax.set_ylabel(f"{ylabel} among accepted")
    ax.set_title(f"Risk-coverage: {source}", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    safe = source.replace(" ", "_")
    fig.savefig(FIGDIR / f"08c_risk_coverage_{safe}.png")
    plt.close(fig)


def fig_deferral_by_source(sweep_df):
    sources = sweep_df["source"].unique()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(sources))
    width = 0.13
    for i, condition in enumerate(CONDITIONS + ["B2"]):
        vals = [sweep_df[(sweep_df.source == s) & (sweep_df.condition == condition)]
                ["deferral_rate"].values[0] for s in sources]
        ax.bar(x + (i - 2) * width, vals, width,
               label=f"{condition} {CONDITION_LABELS[condition]}",
               color=CONDITION_COLOURS[condition], alpha=0.85)
    ax.set_xticks(x, sources, rotation=20, ha="right")
    ax.set_ylabel("Natural deferral rate")
    ax.set_title("Deferral rate by source and condition", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "08c_deferral_by_source.png")
    plt.close(fig)


def fig_auroc_heatmap(auroc_df, signal, sources, title, fname):
    mat = np.full((len(sources), len(sources)), np.nan)
    for i, a in enumerate(sources):
        for j, b in enumerate(sources):
            if a == b:
                continue
            row = auroc_df[(auroc_df.positive == a) & (auroc_df.negative == b)
                           & (auroc_df.signal == signal)]
            if len(row):
                mat[i, j] = row["auroc"].values[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(sources)), sources, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(sources)), sources)
    for i in range(len(sources)):
        for j in range(len(sources)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="AUROC (row = positive/OOD)")
    fig.tight_layout()
    fig.savefig(FIGDIR / fname)
    plt.close(fig)


# Main
def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    df = read_csv(TABDIR / "08b_slice_scores.csv")
    df = add_source_column(df)
    print(f"Loaded {len(df)} rows: " +
          ", ".join(f"{s} n={len(sub)}" for s, sub in df.groupby("source")))

    print("\n" + "=" * 90)
    print("PHYSICAL-PLAUSIBILITY CHECK (run before anything else)")
    print("=" * 90)
    maha_ok, vendor_means = plausibility_check(df)
    vendor_means.to_csv(TABDIR / "08c_retouch_vendor_ood_means.csv")

    if not maha_ok:
        print("\n[STOP] RETOUCH-Spectralis scores MORE out-of-distribution "
              "(lower Mahalanobis-combined) than Topcon or Cirrus. This is "
              "physically backwards for a near-OOD source and means the "
              "16-bit decode has not actually resolved the bit-depth issue, "
              "regardless of what 08a Phase 1's checks suggested. Not "
              "proceeding to risk-coverage curves or confusion analyses -- "
              "upstream assumptions (rescale method) need revisiting.")
        return
    print("\n  PASS -- Spectralis scores more in-distribution than Topcon/"
          "Cirrus on the far-OOD signal, as physically expected. Proceeding.")

    sources = ["Kermany-test", "RETOUCH-Spectralis", "RETOUCH-Topcon",
              "RETOUCH-Cirrus", "OCT5k"]

    # Risk-coverage curves + matched-coverage table, every source x condition
    print("\nBuilding risk-coverage curves...")
    curves = {}
    matched_rows = []
    sweep_rows = []
    baseline_risk = {}
    for source in sources:
        sub = df[df.source == source]
        outcome = outcome_column(sub, source)
        baseline_risk[source] = float((~outcome).mean())
        for condition in CONDITIONS + ["B2"]:
            score = risk_score_for(sub, condition)
            cov, risk = risk_coverage_curve(score, outcome)
            curves[(source, condition)] = (cov, risk)
            for c, r in zip(cov, risk):
                matched_rows.append({"source": source, "condition": condition,
                                     "coverage": float(c), "risk": float(r)})
            if condition != "B2":
                defer = sub[f"defer_condition{condition}"].to_numpy()
            else:
                defer = sub["defer_B2"].to_numpy()
            sweep_rows.append({"source": source, "condition": condition,
                               "deferral_rate": float(defer.mean()), "n": len(sub)})
        fig_risk_coverage_by_source(curves, source, baseline_risk[source])

    pd.DataFrame(matched_rows).to_csv(TABDIR / "08c_risk_coverage.csv", index=False)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(TABDIR / "08c_deferral_by_source.csv", index=False)
    fig_deferral_by_source(sweep_df)

    matched_summary = []
    for source in sources:
        for condition in CONDITIONS + ["B2"]:
            cov_grid, risk_grid = curves[(source, condition)]
            for cov in COVERAGE_POINTS:
                idx = int(round(cov * (len(COVERAGE_GRID) - 1)))
                matched_summary.append({"source": source, "condition": condition,
                                        "coverage": cov, "risk": float(risk_grid[idx])})
    matched_summary_df = pd.DataFrame(matched_summary)
    matched_summary_df.to_csv(TABDIR / "08c_matched_coverage_summary.csv", index=False)
    print("\nMatched-coverage summary (risk at 70/80/90% coverage):")
    print(matched_summary_df.pivot_table(index=["source", "condition"],
         columns="coverage", values="risk").to_string(float_format=lambda v: f"{v:.4f}"))

    # Confusion analysis (a): each source vs Kermany-test
    print("\nConfusion analysis (a): each cross-device source vs Kermany-test...")
    vs_kermany_rows = []
    for signal in ["ood_cosine", "ood_maha_combined"]:
        auroc_df = pairwise_auroc(df, sources, signal)
        vs_kermany = auroc_df[auroc_df.negative == "Kermany-test"]
        vs_kermany_rows.append(vs_kermany)
    vs_kermany_df = pd.concat(vs_kermany_rows, ignore_index=True)
    vs_kermany_df.to_csv(TABDIR / "08c_confusion_vs_kermany.csv", index=False)
    print(vs_kermany_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Confusion analysis (b): cross-device sources vs each other
    print("\nConfusion analysis (b): cross-device sources vs each other "
          "(isolates genuine device shift)...")
    cross_sources = [s for s in sources if s != "Kermany-test"]
    pairwise_rows = []
    for signal in ["ood_cosine", "ood_maha_combined"]:
        pairwise_rows.append(pairwise_auroc(df, cross_sources, signal))
    pairwise_df = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_df.to_csv(TABDIR / "08c_confusion_cross_device.csv", index=False)
    print(pairwise_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    all_auroc = pd.concat([pairwise_auroc(df, sources, s)
                           for s in ["ood_cosine", "ood_maha_combined"]], ignore_index=True)
    for signal in ["ood_cosine", "ood_maha_combined"]:
        fig_auroc_heatmap(all_auroc, signal, sources,
                          f"AUROC (positive=row, {signal})",
                          f"08c_auroc_heatmap_{signal}.png")

    # Confounder note
    print("\nConfounder note: every image (Kermany, RETOUCH, OCT5k) is "
          "resized to the identical 260x260 before the backbone sees it "
          "(build_backbone_transform, 08a), which controls for native-"
          "resolution differences by construction rather than requiring a "
          "separate statistical adjustment. Vendor identity itself is not a "
          "confounder to control for here -- it is the variable of interest "
          "(source separation is a supplementary detector diagnostic).")

    print("\n" + "=" * 90)
    print(f"Figures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("08c_*.csv")):
        print(f"  {f.name}")
    print("\n08c complete.")


if __name__ == "__main__":
    main()
