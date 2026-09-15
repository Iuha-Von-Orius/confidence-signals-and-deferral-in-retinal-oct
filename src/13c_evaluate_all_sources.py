"""Evaluate all nine sources from the two per-image score tables.

RASTI is the dataset stored under oct5k in earlier steps. Labelled sources
use classification errors; RETOUCH and OCTDL ambiguous/novel-class tiers
use missed-disease errors. Sources with different error definitions stay separate.
"""

from paths import PROJECT, read_csv

import importlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

TABDIR = PROJECT / "results" / "tables"
FIGDIR = PROJECT / "results" / "figures" / "13c"

sys.path.insert(0, str(Path(__file__).resolve().parent))
_ood08c = importlib.import_module("08c_evaluate")
_aurc09 = importlib.import_module("09_aurc")

add_source_column = _ood08c.add_source_column
outcome_column = _ood08c.outcome_column
risk_score_for = _ood08c.risk_score_for
risk_coverage_curve = _ood08c.risk_coverage_curve
pairwise_auroc = _ood08c.pairwise_auroc
CLASSES = _ood08c.CLASSES
N_CLASSES = _ood08c.N_CLASSES
NORMAL_IDX = _ood08c.NORMAL_IDX

per_sample_curve = _aurc09.per_sample_curve
aurc_trapezoid = _aurc09.aurc_trapezoid
AURC_SCALE = _aurc09.AURC_SCALE
EPS = np.finfo(float).eps

SOURCES_ORDER = ["Kermany-test", "RETOUCH-Spectralis", "RETOUCH-Cirrus", "RETOUCH-Topcon",
                 "Rasti", "NEH", "OCTDL-in-label", "OCTDL-ambiguous", "OCTDL-novel-class"]
TRUTH_SOURCES = ["Kermany-test", "Rasti", "NEH", "OCTDL-in-label"]
OCTDL_TIERS = ["OCTDL-in-label", "OCTDL-ambiguous", "OCTDL-novel-class"]

CONDITIONS_ALL = [(1, "No deferral"), (2, "OOD signals"), (3, "Uncertainty"),
                  (4, "Conformal"), (5, "All signals"), ("B2", "Softmax confidence")]
CONDITIONS_AURC = [2, 3, 4, 5, "B2"]

SIGNAL_COLS = ["ood_cosine", "ood_maha_combined", "softmax_confidence", "mutual_info", "cp_setsize"]
HIGHER_IS_OOD = {"ood_cosine": False, "ood_maha_combined": False, "softmax_confidence": False,
                 "mutual_info": True, "cp_setsize": True}

IDENTIFIER_COL = {"Kermany-test": None, "RETOUCH-Spectralis": "volume_id",
                  "RETOUCH-Cirrus": "volume_id", "RETOUCH-Topcon": "volume_id",
                  "Rasti": "volume_id", "NEH": "global_patient_id",
                  "OCTDL-in-label": "patient_id", "OCTDL-ambiguous": "patient_id",
                  "OCTDL-novel-class": "patient_id"}

SOURCE_COLOURS_13C = {
    "Kermany-test": "#2c3e50", "RETOUCH-Spectralis": "#d62728",
    "RETOUCH-Cirrus": "#2ca02c", "RETOUCH-Topcon": "#1f77b4", "Rasti": "#9467bd",
    "NEH": "#ff7f0e", "OCTDL-in-label": "#17becf",
    "OCTDL-ambiguous": "#bcbd22", "OCTDL-novel-class": "#8c564b",
}
CONDITION_COLOURS_13C = {1: "#7f8c8d", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd",
                         5: "#d62728", "B2": "#ff7f0e"}

VENDOR_BACKFILL = {"NEH": "Spectralis", "OCTDL": "Optovue"}
VENDOR_CITATIONS = {
    "NEH": ('Sotoudeh-Paima et al., arXiv 2110.03002 -- "the Heidelberg '
           'SD-OCT imaging system at Noor Eye Hospital, Tehran, Iran"'),
    "OCTDL": ("OCTDL, Scientific Data 2024, doi:10.1038/s41597-024-03182-7 "
             "-- acquisition device Optovue Avanti RTVue XR"),
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3,
})


# Data loading
def load_merged_df():
    old = read_csv(TABDIR / "08b_slice_scores.csv")
    old = add_source_column(old)
    old["source"] = old["source"].replace({"OCT5k": "Rasti"})

    new = read_csv(TABDIR / "13b_new_slice_scores.csv")
    new = new.copy()
    new["source"] = new.apply(
        lambda r: "NEH" if r["dataset"] == "NEH" else f"OCTDL-{r['label_group']}", axis=1)

    merged = pd.concat([old, new], ignore_index=True)

    def backfill_vendor(row):
        if row["source"] == "NEH":
            return VENDOR_BACKFILL["NEH"]
        if row["source"].startswith("OCTDL-"):
            return VENDOR_BACKFILL["OCTDL"]
        return row["vendor"]
    merged["vendor"] = merged.apply(backfill_vendor, axis=1)
    return merged


def outcome_source_arg(source):
    """Route OCTDL ambiguous and novel-class tiers to the missed-disease outcome rule."""
    if source in ("OCTDL-ambiguous", "OCTDL-novel-class"):
        return "RETOUCH"
    return source


def correct_for(sub, source):
    return outcome_column(sub, outcome_source_arg(source))


def risk_definition_for(source):
    """Return the error definition used for the source."""
    return "missed_disease" if outcome_source_arg(source).startswith("RETOUCH") else "four_class_accuracy"


def signal_for_auroc(df, signal):
    """Orient the signal so the pairwise AUROC helper treats higher values as OOD."""
    if HIGHER_IS_OOD[signal]:
        col = f"__neg_{signal}"
        df2 = df.copy()
        df2[col] = -df[signal]
        return df2, col
    return df, signal


# T01
def build_t01(merged):
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        has_truth = source in TRUTH_SOURCES
        id_col = IDENTIFIER_COL[source]
        n_ids = int(sub[id_col].nunique()) if id_col else None

        acc = float((sub["predicted_class_name"] == sub["true_label_name"]).mean()) if has_truth else None

        diseased = sub[sub["known_diseased"] == True]
        sensitivity = float((diseased["predicted_class_name"] != "NORMAL").mean()) if len(diseased) else None
        not_diseased = sub[sub["known_diseased"] == False]
        specificity = float((not_diseased["predicted_class_name"] == "NORMAL").mean()) if len(not_diseased) else None

        notes = []
        if not has_truth:
            notes.append("no per-class ground truth; missed-disease semantics only")
        if len(not_diseased) == 0:
            notes.append("no known-normal rows in this source; specificity undefined")
        if id_col is None:
            notes.append("no patient/volume identifier for this source")

        row = {"source": source, "n": len(sub), "n_identifiers": n_ids,
              "vendor": sub["vendor"].iloc[0] if len(sub) else None,
              "has_4class_truth": has_truth,
              "accuracy_condition1": acc, "sensitivity": sensitivity,
              "specificity": specificity, "note": "; ".join(notes) if notes else ""}
        for sig in SIGNAL_COLS:
            row[f"{sig}_mean"] = float(sub[sig].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T01_source_summary.csv", index=False)
    return df


# T02 / T03 / T04
def build_t02(merged):
    tables = {}
    for source in TRUTH_SOURCES:
        sub = merged[merged["source"] == source]
        cm = pd.crosstab(sub["true_label_name"], sub["predicted_class_name"])
        cm = cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
        cm_pct = (cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(2)
        tables[source] = (cm, cm_pct)
        out = cm.copy()
        out.columns = [f"pred_{c}_n" for c in out.columns]
        out_pct = cm_pct.copy()
        out_pct.columns = [f"pred_{c}_pct" for c in out_pct.columns]
        combined = pd.concat([out, out_pct], axis=1)
        combined.index.name = "true_label_name"
        safe = source.replace(" ", "_")
        combined.to_csv(TABDIR / f"13c_T02_confusion_{safe}.csv")
    return tables


def build_t03(merged):
    rows = []
    for source in TRUTH_SOURCES:
        sub = merged[merged["source"] == source]
        true_d = sub["true_label_name"].ne("NORMAL").map({True: "Disease", False: "Normal"})
        pred_d = sub["predicted_class_name"].ne("NORMAL").map({True: "Disease", False: "Normal"})
        cm = pd.crosstab(true_d, pred_d).reindex(index=["Disease", "Normal"],
                                                  columns=["Disease", "Normal"], fill_value=0)
        cm.index.name = "source"
        for true_state in ["Disease", "Normal"]:
            for pred_state in ["Disease", "Normal"]:
                rows.append({"source": source, "true_state": true_state,
                            "pred_state": pred_state, "n": int(cm.loc[true_state, pred_state])})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T03_disease_vs_normal.csv", index=False)
    return df


def build_t04(merged):
    rows = []
    for c in CLASSES:
        row = {"class": c}
        for source in TRUTH_SOURCES:
            sub = merged[(merged["source"] == source) & (merged["true_label_name"] == c)]
            row[source] = float((sub["predicted_class_name"] == c).mean()) if len(sub) else None
            row[f"{source}_n"] = len(sub)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T04_per_class_accuracy.csv", index=False)
    return df


# T05
def build_t05(merged):
    g = merged.groupby("source")[SIGNAL_COLS].agg(["mean", "std"]).reindex(SOURCES_ORDER)
    g.to_csv(TABDIR / "13c_T05_signal_stats.csv")
    return g


# T06
def build_t06(merged):
    rows = []
    for signal in SIGNAL_COLS:
        df2, col = signal_for_auroc(merged, signal)
        auroc_df = pairwise_auroc(df2, SOURCES_ORDER, col)
        vs_kermany = auroc_df[auroc_df["negative"] == "Kermany-test"].copy()
        vs_kermany["signal"] = signal
        rows.append(vs_kermany[["positive", "negative", "signal", "auroc"]])
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(TABDIR / "13c_T06_auroc_vs_kermany.csv", index=False)
    return df


# T07
def build_t07(merged):
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        for cond, label in CONDITIONS_ALL:
            if cond == 1:
                rate = 0.0
            elif cond == "B2":
                rate = float(sub["defer_B2"].mean())
            else:
                rate = float(sub[f"defer_condition{cond}"].mean())
            rows.append({"source": source, "condition": cond, "condition_label": label,
                        "deferral_rate": rate, "n": len(sub)})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T07_deferral_rate.csv", index=False)
    return df


# T08
def build_t08(merged):
    """Compute supplementary AURC; retain the recorded undefined marker for zero errors."""
    undefined_rows = []
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source].reset_index(drop=True)
        correct = correct_for(sub, source)
        n_errors = int((~correct).sum())
        base_error = 1.0 - correct.mean()
        optimal = base_error + (1.0 - base_error) * np.log(1.0 - base_error + EPS)
        risk_def = risk_definition_for(source)
        for condition in CONDITIONS_AURC:
            if n_errors == 0:
                aurc = "undefined (0 errors)"
                e_aurc = "undefined (0 errors)"
                undefined_rows.append((source, condition))
            else:
                risk_score = risk_score_for(sub, condition)
                coverage, risk = per_sample_curve(risk_score, correct)
                aurc = aurc_trapezoid(coverage, risk)
                e_aurc = aurc - optimal * AURC_SCALE
            rows.append({"source": source, "condition": condition, "n": len(sub),
                        "base_error": base_error, "aurc": aurc, "e_aurc": e_aurc,
                        "risk_definition": risk_def})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T08_aurc.csv", index=False)
    return df, undefined_rows


# T09
def build_t09(merged):
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        has_truth = source in TRUTH_SOURCES
        for cond, label in CONDITIONS_ALL:
            if cond == 1:
                accepted = sub
            elif cond == "B2":
                accepted = sub[~sub["defer_B2"]]
            else:
                accepted = sub[~sub[f"defer_condition{cond}"]]
            accuracy = (float((accepted["predicted_class_name"] == accepted["true_label_name"]).mean())
                       if has_truth and len(accepted) else None)
            diseased_accepted = accepted[accepted["known_diseased"] == True]
            miss_rate = (float((diseased_accepted["predicted_class_name"] == "NORMAL").mean())
                        if len(diseased_accepted) else None)
            rows.append({"source": source, "condition": cond, "condition_label": label,
                        "n_accepted": len(accepted), "accuracy": accuracy, "miss_rate": miss_rate})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T09_accepted_set_stats.csv", index=False)
    return df


# T10
def build_t10(merged, t06):
    stats = merged[merged["source"].isin(OCTDL_TIERS)].groupby("source")[SIGNAL_COLS].agg(["mean", "std"])
    stats.to_csv(TABDIR / "13c_T10a_octdl_tier_signal_stats.csv")

    vs_kermany = t06[t06["positive"].isin(OCTDL_TIERS)]
    vs_kermany.to_csv(TABDIR / "13c_T10b_octdl_tier_vs_kermany_auroc.csv", index=False)

    pairwise_rows = []
    for signal in SIGNAL_COLS:
        df2, col = signal_for_auroc(merged, signal)
        pw = pairwise_auroc(df2, OCTDL_TIERS, col)
        pw = pw.copy()
        pw["signal"] = signal
        pairwise_rows.append(pw[["positive", "negative", "signal", "auroc"]])
    pairwise_df = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_df.to_csv(TABDIR / "13c_T10c_octdl_tier_pairwise_auroc.csv", index=False)
    return stats, vs_kermany, pairwise_df


# T11
def build_t11(merged):
    rows = []
    hist_lines = []
    for source in SOURCES_ORDER:
        id_col = IDENTIFIER_COL[source]
        if id_col is None:
            continue
        sub = merged[merged["source"] == source].copy()
        correct = correct_for(sub, source)
        sub["_error"] = ~correct
        per_id = sub.groupby(id_col)["_error"].sum()
        rows.append({"source": source, "n_identifiers": int(len(per_id)),
                    "n_identifiers_with_error": int((per_id > 0).sum()),
                    "min_errors": int(per_id.min()), "median_errors": float(per_id.median()),
                    "max_errors": int(per_id.max())})
        vc = per_id.value_counts().sort_index()
        hist_lines.append(f"  {source}: " + ", ".join(f"{int(k)} errors x{int(v)} identifiers"
                                                      for k, v in vc.items()))
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "13c_T11_error_concentration.csv", index=False)
    return df, hist_lines


# T12
def build_t12(merged, t10_pairwise):
    groups = ["OCTDL-in-label", "OCTDL-ambiguous"]
    stats = merged[merged["source"].isin(groups)].groupby("source")[SIGNAL_COLS].agg(["mean", "std"])

    defer_rows = []
    for source in groups:
        sub = merged[merged["source"] == source]
        for cond, label in CONDITIONS_ALL:
            if cond == 1:
                rate = 0.0
            elif cond == "B2":
                rate = float(sub["defer_B2"].mean())
            else:
                rate = float(sub[f"defer_condition{cond}"].mean())
            defer_rows.append({"source": source, "condition": cond, "condition_label": label,
                              "deferral_rate": rate})
    defer_df = pd.DataFrame(defer_rows)

    auroc_df = t10_pairwise[(t10_pairwise["positive"] == "OCTDL-ambiguous")
                            & (t10_pairwise["negative"] == "OCTDL-in-label")]

    stats_out = stats.copy()
    stats_out.to_csv(TABDIR / "13c_T12a_ambiguous_vs_inlabel_signals.csv")
    defer_df.to_csv(TABDIR / "13c_T12b_ambiguous_vs_inlabel_deferral.csv", index=False)
    auroc_df.to_csv(TABDIR / "13c_T12c_ambiguous_vs_inlabel_auroc.csv", index=False)
    return stats, defer_df, auroc_df


# Figures
def fig_f01(t02_tables):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    for ax, source in zip(axes.flat, TRUTH_SOURCES):
        cm, cm_pct = t02_tables[source]
        im = ax.imshow(cm_pct.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(N_CLASSES), CLASSES)
        ax.set_yticks(range(N_CLASSES), CLASSES)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                n = cm.iloc[i, j]
                pct = cm_pct.iloc[i, j]
                text = f"{n}" if pd.isna(pct) else f"{n}\n{pct:.1f}%"
                ax.text(j, i, text, ha="center", va="center", fontsize=10,
                       color="white" if (not pd.isna(pct) and pct > 50) else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(source)
    fig.suptitle("Confusion matrices (count and row %)", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F01_confusion_matrices.png")
    plt.close(fig)


def fig_f02(merged):
    for signal, fname in [("ood_maha_combined", "13c_F02_maha.png"),
                          ("ood_cosine", "13c_F02_cosine.png")]:
        fig, ax = plt.subplots(figsize=(15, 7))
        data = [merged.loc[merged["source"] == s, signal].dropna().to_numpy() for s in SOURCES_ORDER]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(SOURCE_COLOURS_13C[SOURCES_ORDER[i]])
            pc.set_alpha(0.75)
        ax.set_xticks(range(1, len(SOURCES_ORDER) + 1), SOURCES_ORDER, rotation=30, ha="right")
        ax.set_ylabel(signal)
        ax.set_title(f"Distribution of {signal} by source")
        fig.tight_layout()
        fig.savefig(FIGDIR / fname)
        plt.close(fig)


def fig_f03(t06):
    sources = [s for s in SOURCES_ORDER if s != "Kermany-test"]
    mat = np.full((len(sources), len(SIGNAL_COLS)), np.nan)
    for i, s in enumerate(sources):
        for j, sig in enumerate(SIGNAL_COLS):
            row = t06[(t06["positive"] == s) & (t06["signal"] == sig)]
            if len(row):
                mat[i, j] = row["auroc"].values[0]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SIGNAL_COLS)), SIGNAL_COLS, rotation=30, ha="right")
    ax.set_yticks(range(len(sources)), sources)
    for i in range(len(sources)):
        for j in range(len(SIGNAL_COLS)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=11)
    ax.set_title("OOD AUROC vs Kermany-test, by source and signal")
    fig.colorbar(im, ax=ax, label="AUROC (positive = row source, oriented higher=more OOD)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F03_auroc_heatmap.png")
    plt.close(fig)


def fig_f04(merged):
    """Plot risk-coverage curves separately for each new source and OCTDL tier."""
    sources = ["NEH", "OCTDL-in-label", "OCTDL-ambiguous", "OCTDL-novel-class"]
    for source in sources:
        sub = merged[merged["source"] == source].reset_index(drop=True)
        correct = correct_for(sub, source)
        risk_def = risk_definition_for(source)
        n_errors = int((~correct).sum())

        fig, ax = plt.subplots(figsize=(9, 6.3))
        for condition in [2, 3, 4, 5, "B2"]:
            score = risk_score_for(sub, condition)
            cov, risk = risk_coverage_curve(score, correct)
            label = "B2 Softmax confidence" if condition == "B2" else dict(CONDITIONS_ALL)[condition]
            ax.plot(cov, risk, "-", lw=2.2, color=CONDITION_COLOURS_13C[condition], label=f"{condition} {label}")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk among accepted")
        ax.set_title(f"Risk-coverage: {source}\ncorrect-definition: {risk_def}", fontsize=13)
        if n_errors == 0:
            ax.text(0.5, 0.5, "This group has 0 errors: risk is 0 at every coverage "
                   "level for every condition. The curve carries no ranking information.",
                   transform=ax.transAxes, ha="center", va="center", fontsize=11,
                   wrap=True, bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))
        ax.legend(fontsize=10)
        fig.tight_layout()
        safe = source.replace(" ", "_")
        fig.savefig(FIGDIR / f"13c_F04_risk_coverage_{safe}.png")
        plt.close(fig)


def fig_f05(t07):
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(SOURCES_ORDER))
    width = 0.13
    conds = [c for c, _ in CONDITIONS_ALL]
    for i, (cond, label) in enumerate(CONDITIONS_ALL):
        vals = [t07[(t07["source"] == s) & (t07["condition"] == cond)]["deferral_rate"].values[0]
               for s in SOURCES_ORDER]
        ax.bar(x + (i - 2.5) * width, vals, width, label=f"{cond} {label}",
              color=CONDITION_COLOURS_13C[cond], alpha=0.88)
    ax.set_xticks(x, SOURCES_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Deferral rate")
    ax.set_title("Deferral rate by source and condition")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F05_deferral_by_source.png")
    plt.close(fig)


def fig_f06():
    inten = read_csv(TABDIR / "11_explore_new_data.csv")
    inten = inten.copy()
    inten["display"] = inten["dataset"].replace({"OCT5k": "Rasti"})

    def wmean(group, col):
        w = group["n_intensity_sample"].to_numpy(dtype=float)
        return float(np.average(group[col].to_numpy(dtype=float), weights=w))

    rows = []
    for display, group in inten.groupby("display"):
        rows.append({"display": display, "p1": wmean(group, "p1"),
                    "p50": wmean(group, "p50"), "p99": wmean(group, "p99")})
    pooled = pd.DataFrame(rows)
    order = ["Kermany-train", "NEH", "OCTDL", "RETOUCH-Spectralis", "RETOUCH-Cirrus",
            "RETOUCH-Topcon", "Rasti"]
    pooled = pooled.set_index("display").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, stat in zip(axes, ["p1", "p50", "p99"]):
        colours = [SOURCE_COLOURS_13C.get(d, "#999999") for d in pooled["display"]]
        ax.bar(range(len(pooled)), pooled[stat], color=colours)
        ax.set_xticks(range(len(pooled)), pooled["display"], rotation=35, ha="right")
        ax.set_ylabel(f"{stat} (pixel intensity, 0-255)")
        ax.set_title(stat)
    fig.suptitle("Raw-pixel intensity percentiles by dataset (from 11_explore_new_data.csv, "
                "n_intensity_sample-weighted mean across classes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F06_intensity_percentiles.png")
    plt.close(fig)


def fig_f07(merged):
    groups = ["OCTDL-in-label", "OCTDL-novel-class"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, sig in zip(axes, SIGNAL_COLS):
        for g in groups:
            vals = merged.loc[merged["source"] == g, sig].dropna()
            ax.hist(vals, bins=40, density=True, alpha=0.55, color=SOURCE_COLOURS_13C[g], label=g)
        ax.set_xlabel(sig)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
    fig.suptitle("OCTDL in-label vs novel-class: signal distributions", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F07_octdl_inlabel_vs_novel.png")
    plt.close(fig)


def fig_f08(t01):
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in t01.iterrows():
        s = row["source"]
        sens = row["sensitivity"]
        if sens is None or pd.isna(sens):
            continue
        x = row["ood_maha_combined_mean"]
        y = 1.0 - sens
        ax.scatter(x, y, s=140, color=SOURCE_COLOURS_13C[s], edgecolor="black", zorder=3)
        ax.annotate(s, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=11)
    ax.set_xlabel("Mean ood_maha_combined (higher = more in-distribution)")
    ax.set_ylabel("Miss rate at no-deferral (1 - sensitivity)")
    ax.set_title("Source-level OOD score vs no-deferral miss rate")
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F08_maha_vs_missrate.png")
    plt.close(fig)


def fig_f09(merged):
    positive = merged[merged["source"] == "OCTDL-novel-class"]
    negative = merged[merged["source"] == "OCTDL-in-label"]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for sig in SIGNAL_COLS:
        sa = positive[sig].to_numpy()
        sb = negative[sig].to_numpy()
        if HIGHER_IS_OOD[sig]:
            sa, sb = sa, sb
        else:
            sa, sb = -sa, -sb
        y = np.concatenate([np.ones(len(sa)), np.zeros(len(sb))])
        s = np.concatenate([sa, sb])
        fpr, tpr, _ = roc_curve(y, s)
        df2, col = signal_for_auroc(merged, sig)
        auroc_row = pairwise_auroc(df2, ["OCTDL-novel-class", "OCTDL-in-label"], col)
        auroc = auroc_row[(auroc_row["positive"] == "OCTDL-novel-class")
                          & (auroc_row["negative"] == "OCTDL-in-label")]["auroc"].values[0]
        ax.plot(fpr, tpr, lw=2.2, label=f"{sig} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: OCTDL novel-class (positive) vs OCTDL in-label (negative)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F09_roc_novel_vs_inlabel.png")
    plt.close(fig)


def fig_f10(merged, t12_defer):
    groups = ["OCTDL-in-label", "OCTDL-ambiguous"]
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1])

    gs_left = gs[0].subgridspec(1, 3)
    for i, sig in enumerate(["softmax_confidence", "mutual_info", "cp_setsize"]):
        ax = fig.add_subplot(gs_left[i])
        for g in groups:
            vals = merged.loc[merged["source"] == g, sig].dropna()
            ax.hist(vals, bins=35, density=True, alpha=0.55, color=SOURCE_COLOURS_13C[g], label=g)
        ax.set_xlabel(sig)
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    conds = [c for c, _ in CONDITIONS_ALL]
    x = np.arange(len(conds))
    width = 0.35
    for i, g in enumerate(groups):
        vals = [t12_defer[(t12_defer["source"] == g) & (t12_defer["condition"] == c)]["deferral_rate"].values[0]
               for c in conds]
        ax2.bar(x + (i - 0.5) * width, vals, width, label=g, color=SOURCE_COLOURS_13C[g], alpha=0.88)
    ax2.set_xticks(x, [str(c) for c in conds])
    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Deferral rate")
    ax2.legend(fontsize=10)

    fig.suptitle("OCTDL ambiguous vs in-label: signal distributions and deferral rate", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGDIR / "13c_F10_ambiguous_vs_inlabel.png")
    plt.close(fig)


# Main
def main():
    t0 = time.time()
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 78)
    report.append("13c_evaluate_all_sources.py -- nine-source evaluation")
    report.append("=" * 78)
    report.append("")
    report.append("outcome_column dispatch: OCTDL-ambiguous/novel-class (no class label, "
                  "known_diseased=True on every row) are passed to outcome_column as the "
                  "literal string \"RETOUCH\" (dispatch only, never relabels the row's own "
                  "source column) so its missed-disease branch applies -- used for T08 and "
                  "T11 only; T09's miss-rate is instead computed directly from "
                  "known_diseased (a more general, row-level formula, not via this "
                  "dispatch), and T01/T02/T04 correctly leave these two tiers blank.")
    report.append("")
    report.append("AUROC sign handling (pairwise_auroc always computes -signal_col "
                  "internally, correct only for higher=in-distribution signals):")
    for sig in SIGNAL_COLS:
        if HIGHER_IS_OOD[sig]:
            report.append(f"  {sig:<20s} raw: higher=more OOD -> pre-negated before calling "
                          f"pairwise_auroc, so its internal negation cancels -> net: no sign change")
        else:
            report.append(f"  {sig:<20s} raw: higher=more in-distribution -> passed as-is "
                          f"-> net: pairwise_auroc's negation orients it higher=more OOD")
    report.append("")
    report.append("Vendor backfill (13c's in-memory copy only, never written back to "
                  "13b_new_slice_scores.csv):")
    for k, v in VENDOR_CITATIONS.items():
        report.append(f"  {k} -> \"{VENDOR_BACKFILL[k]}\"  ({v})")

    print("Loading and merging 08b_slice_scores.csv + 13b_new_slice_scores.csv...")
    merged = load_merged_df()
    print(f"  {len(merged)} rows: " + ", ".join(f"{s} n={len(sub)}" for s, sub in merged.groupby("source")))

    print("Building tables T01-T12...")
    t01 = build_t01(merged)
    t02 = build_t02(merged)
    t03 = build_t03(merged)
    t04 = build_t04(merged)
    t05 = build_t05(merged)
    t06 = build_t06(merged)
    t07 = build_t07(merged)
    t08, t08_undefined = build_t08(merged)
    t09 = build_t09(merged)
    t10_stats, t10_vs_kermany, t10_pairwise = build_t10(merged, t06)
    t11, t11_hist_lines = build_t11(merged)
    t12_stats, t12_defer, t12_auroc = build_t12(merged, t10_pairwise)

    print("Building figures F01-F10...")
    fig_f01(t02)
    fig_f02(merged)
    fig_f03(t06)
    fig_f04(merged)
    fig_f05(t07)
    fig_f06()
    fig_f07(merged)
    fig_f08(t01)
    fig_f09(merged)
    fig_f10(merged, t12_defer)

    elapsed = time.time() - t0

    report.append("")
    report.append("-" * 78)
    report.append("Tables")
    report.append("-" * 78)
    report.append("T01 13c_T01_source_summary.csv -- one row per source: n, patient/volume "
                  "count, vendor, has_4class_truth, accuracy/sensitivity/specificity, "
                  "5-signal means.")
    report.append("T02 13c_T02_confusion_<source>.csv (x4) -- 4x4 confusion matrix, "
                  "count and row %, for the four sources with class-level truth.")
    report.append("T03 13c_T03_disease_vs_normal.csv -- 2x2 disease-vs-normal confusion, "
                  "same four sources.")
    report.append("T04 13c_T04_per_class_accuracy.csv -- per-class accuracy, rows=CLASSES, "
                  "columns=the four truth-bearing sources.")
    report.append("T05 13c_T05_signal_stats.csv -- mean/std of the five signals, all nine sources.")
    report.append("T06 13c_T06_auroc_vs_kermany.csv -- AUROC of each non-Kermany source vs "
                  "Kermany-test, per signal.")
    report.append("T07 13c_T07_deferral_rate.csv -- deferral rate, source x six conditions.")
    report.append("T08 13c_T08_aurc.csv -- AURC/E-AURC, source x [2,3,4,5,B2], via 09's "
                  "per_sample_curve/aurc_trapezoid; risk_definition column states which "
                  "correct-definition each row used (derived from the same predicate "
                  "outcome_column itself dispatches on, not a separately-inferred check).")
    if t08_undefined:
        report.append("  (source, condition) pairs with zero errors under their own "
                      "correct-definition -- aurc/e_aurc written as the string "
                      "\"undefined (0 errors)\", not 0.0 or NaN (a zero-error group has "
                      "no ranking to evaluate, and E-AURC's closed form would otherwise "
                      "return a spurious tiny negative value):")
        for source, condition in t08_undefined:
            report.append(f"    {source}, condition {condition}")
    else:
        report.append("  No (source, condition) pair had zero errors.")
    report.append("T09 13c_T09_accepted_set_stats.csv -- accuracy and literal miss-rate "
                  "among accepted samples at the natural operating point, source x six conditions.")
    report.append("T10a/b/c 13c_T10*_octdl_tier_*.csv -- OCTDL's three tiers' signal stats, "
                  "AUROC vs Kermany-test, and pairwise AUROC between tiers.")
    report.append("T11 13c_T11_error_concentration.csv -- for the eight identifier-bearing "
                  "sources: identifier count, count with >=1 error, min/median/max errors "
                  "per identifier. Full errors-per-identifier histogram (not in the CSV):")
    report += t11_hist_lines
    report.append("T12a/b/c 13c_T12*_ambiguous_vs_inlabel_*.csv -- OCTDL ambiguous vs "
                  "in-label: signal stats, deferral rate at six conditions, pairwise AUROC.")

    report.append("")
    report.append("-" * 78)
    report.append("Figures")
    report.append("-" * 78)
    report.append("F01 13c_F01_confusion_matrices.png -- 2x2 grid of T02's four confusion matrices.")
    report.append("F02 13c_F02_maha.png / 13c_F02_cosine.png -- violin plots, all nine "
                  "sources, one signal per figure.")
    report.append("F03 13c_F03_auroc_heatmap.png -- AUROC heatmap, source x signal (T06's data).")
    report.append("F04 13c_F04_risk_coverage_NEH.png / _OCTDL-in-label.png / "
                  "_OCTDL-ambiguous.png / _OCTDL-novel-class.png -- one panel per source "
                  "(not pooled -- OCTDL's three tiers use two different correct-"
                  "definitions, so mixing them into one curve would not be an "
                  "interpretable quantity), five condition curves each, title states "
                  "which correct-definition that panel uses. OCTDL-ambiguous (0 errors) "
                  "still plotted, flat at risk=0, with an on-figure note that the curve "
                  "carries no ranking information.")
    report.append("F05 13c_F05_deferral_by_source.png -- grouped bar, deferral rate, "
                  "source x condition (T07's data).")
    report.append("F06 13c_F06_intensity_percentiles.png -- p1/p50/p99 bar charts, seven "
                  "physical datasets, from 11_explore_new_data.csv (n_intensity_sample-"
                  "weighted mean across each dataset's own classes, not recomputed from pixels).")
    report.append("F07 13c_F07_octdl_inlabel_vs_novel.png -- five-panel overlaid histograms, "
                  "OCTDL in-label vs novel-class.")
    report.append("F08 13c_F08_maha_vs_missrate.png -- scatter, mean ood_maha_combined vs "
                  "no-deferral miss rate, one point per source (T01's data).")
    report.append("F09 13c_F09_roc_novel_vs_inlabel.png -- five ROC curves, OCTDL "
                  "novel-class vs in-label, one per signal, legend states AUROC.")
    report.append("F10 13c_F10_ambiguous_vs_inlabel.png -- OCTDL ambiguous vs in-label: "
                  "three signal-distribution panels plus a six-condition deferral-rate bar chart.")

    report.append("")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    (TABDIR / "13c_evaluation.txt").write_text("\n".join(str(l) for l in report) + "\n")
    print(f"\nWrote results/tables/13c_evaluation.txt")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
