"""Report per-disease confusion, missed disease and acceptance for all policies.

Use the nine-source score table without refitting models or policies.
"""

from paths import PROJECT, read_csv

import importlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABDIR = PROJECT / "results" / "tables"
FIGDIR = PROJECT / "results" / "figures" / "17"

sys.path.insert(0, str(Path(__file__).resolve().parent))
_e13c = importlib.import_module("13c_evaluate_all_sources")

load_merged_df = _e13c.load_merged_df
SOURCES_ORDER = _e13c.SOURCES_ORDER
TRUTH_SOURCES = _e13c.TRUTH_SOURCES
CONDITIONS_ALL = _e13c.CONDITIONS_ALL
CLASSES = _e13c.CLASSES
CONDITION_COLOURS_13C = _e13c.CONDITION_COLOURS_13C

LABELLESS_SOURCES = [s for s in SOURCES_ORDER if s not in TRUTH_SOURCES]


def accepted_mask(sub, condition):
    """Return the acceptance mask for a condition, including the no-deferral baseline."""
    if condition == 1:
        return pd.Series(True, index=sub.index)
    if condition == "B2":
        return ~sub["defer_B2"]
    return ~sub[f"defer_condition{condition}"]


def deferral_rate_for(sub, condition):
    if condition == 1:
        return 0.0
    accepted = accepted_mask(sub, condition)
    return float((~accepted).mean())


# Table A
def build_table_a(merged):
    rows = []
    for source in TRUTH_SOURCES:
        sub = merged[merged["source"] == source]
        class_totals = sub["true_label_name"].value_counts()
        for condition, _ in CONDITIONS_ALL:
            accepted = accepted_mask(sub, condition)
            acc_df = sub[accepted]
            def_df = sub[~accepted]
            for true_class in CLASSES:
                total = int(class_totals.get(true_class, 0))
                acc_tc = acc_df[acc_df["true_label_name"] == true_class]
                def_tc = def_df[def_df["true_label_name"] == true_class]
                for predicted_class in CLASSES:
                    count = int((acc_tc["predicted_class_name"] == predicted_class).sum())
                    row_pct = round(count / total * 100, 2) if total else None
                    rows.append({"source": source, "condition": condition,
                                "true_class": true_class, "predicted_class": predicted_class,
                                "count": count, "row_pct": row_pct})
                count = len(def_tc)
                row_pct = round(count / total * 100, 2) if total else None
                rows.append({"source": source, "condition": condition,
                            "true_class": true_class, "predicted_class": "deferred",
                            "count": count, "row_pct": row_pct})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "17_T17_confusion_by_disease.csv", index=False)
    return df


# Table B
def build_table_b(merged):
    rows = []
    for source in TRUTH_SOURCES:
        sub = merged[merged["source"] == source].copy()
        sub["_true_state"] = sub["true_label_name"].ne("NORMAL").map({True: "Disease", False: "Normal"})
        sub["_pred_state"] = sub["predicted_class_name"].ne("NORMAL").map({True: "Disease", False: "Normal"})
        state_totals = sub["_true_state"].value_counts()
        for condition, _ in CONDITIONS_ALL:
            accepted = accepted_mask(sub, condition)
            acc_df = sub[accepted]
            def_df = sub[~accepted]
            for true_state in ["Disease", "Normal"]:
                total = int(state_totals.get(true_state, 0))
                acc_ts = acc_df[acc_df["_true_state"] == true_state]
                def_ts = def_df[def_df["_true_state"] == true_state]
                for predicted_state in ["Disease", "Normal"]:
                    count = int((acc_ts["_pred_state"] == predicted_state).sum())
                    row_pct = round(count / total * 100, 2) if total else None
                    rows.append({"source": source, "condition": condition,
                                "true_state": true_state, "predicted_state": predicted_state,
                                "count": count, "row_pct": row_pct})
                count = len(def_ts)
                row_pct = round(count / total * 100, 2) if total else None
                rows.append({"source": source, "condition": condition,
                            "true_state": true_state, "predicted_state": "deferred",
                            "count": count, "row_pct": row_pct})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "17_T18_disease_vs_normal.csv", index=False)
    return df


# Table C
def build_table_c(merged):
    rows = []
    for source in LABELLESS_SOURCES:
        sub = merged[merged["source"] == source]
        n = len(sub)
        for condition, _ in CONDITIONS_ALL:
            accepted = accepted_mask(sub, condition)
            acc_df = sub[accepted]
            n_accepted = len(acc_df)
            n_deferred = n - n_accepted
            n_accepted_normal = int((acc_df["predicted_class_name"] == "NORMAL").sum())
            pct = round(n_accepted_normal / n_accepted * 100, 2) if n_accepted else None
            rows.append({"source": source, "condition": condition, "n": n,
                        "n_deferred": n_deferred, "n_accepted": n_accepted,
                        "n_accepted_normal": n_accepted_normal,
                        "pct_accepted_normal": pct})
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "17_T19_missed_disease_labelless.csv", index=False)
    return df


# Regression checks
def run_checks(merged, table_a):
    mismatches = []
    report = []

    t01 = read_csv(TABDIR / "13c_T01_source_summary.csv").set_index("source")
    t07 = read_csv(TABDIR / "13c_T07_deferral_rate.csv")

    report.append("Check 1: accepted + deferred == n, and n == "
                  "13c_T01_source_summary.csv's own 'n' column (54 combinations: "
                  "9 sources x 6 conditions)")
    check1_fail = 0
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        n = len(sub)
        n_t01 = int(t01.loc[source, "n"])
        if n != n_t01:
            mismatches.append(f"Check 1: {source} n={n} != T01 n={n_t01}")
            check1_fail += 1
        for condition, _ in CONDITIONS_ALL:
            accepted = accepted_mask(sub, condition)
            n_acc, n_def = int(accepted.sum()), int((~accepted).sum())
            if n_acc + n_def != n:
                mismatches.append(f"Check 1: {source} condition={condition} "
                                  f"accepted({n_acc})+deferred({n_def}) != n({n})")
                check1_fail += 1
    report.append(f"  {'PASS' if check1_fail == 0 else f'{check1_fail} FAILURES'}")

    report.append("")
    report.append("Check 2: deferral rate == 13c_T07_deferral_rate.csv's "
                  "'deferral_rate' column (same 54 combinations). T07's 'condition' "
                  "column round-trips through CSV as strings ('1'..'5','B2'), not "
                  "native int/str -- this script's own condition value is cast via "
                  "str(condition) before matching, the same CSV round-trip dtype "
                  "trap already documented in this project's 15_augrc.py fix.")
    check2_fail = 0
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        for condition, _ in CONDITIONS_ALL:
            rate = deferral_rate_for(sub, condition)
            t07_row = t07[(t07["source"] == source) & (t07["condition"] == str(condition))]
            if len(t07_row) != 1:
                mismatches.append(f"Check 2: {source} condition={condition}: "
                                  f"expected exactly 1 matching T07 row, found {len(t07_row)}")
                check2_fail += 1
                continue
            t07_rate = float(t07_row["deferral_rate"].iloc[0])
            if abs(rate - t07_rate) > 1e-9:
                mismatches.append(f"Check 2: {source} condition={condition}: "
                                  f"computed={rate} T07={t07_rate}")
                check2_fail += 1
    report.append(f"  {'PASS' if check2_fail == 0 else f'{check2_fail} FAILURES'}")

    report.append("")
    report.append("Check 3: condition-1 confusion matrix == "
                  "13c_T02_confusion_<source>.csv cell by cell, for the four "
                  "TRUTH_SOURCES ('pred_{CLASS}_n' columns)")
    check3_fail = 0
    for source in TRUTH_SOURCES:
        safe = source.replace(" ", "_")
        path = TABDIR / f"13c_T02_confusion_{safe}.csv"
        if not path.exists():
            mismatches.append(f"Check 3: {path.name} MISSING -- check 3 not run for {source}")
            check3_fail += 1
            continue
        t02 = read_csv(path).set_index("true_label_name")
        sub_a = table_a[(table_a["source"] == source) & (table_a["condition"] == 1)
                        & (table_a["predicted_class"] != "deferred")]
        for true_class in CLASSES:
            for predicted_class in CLASSES:
                mine = int(sub_a[(sub_a["true_class"] == true_class)
                                 & (sub_a["predicted_class"] == predicted_class)]["count"].iloc[0])
                theirs = int(t02.loc[true_class, f"pred_{predicted_class}_n"])
                if mine != theirs:
                    mismatches.append(f"Check 3: {source} true={true_class} pred={predicted_class}: "
                                      f"this script={mine} T02={theirs}")
                    check3_fail += 1
    report.append(f"  {'PASS' if check3_fail == 0 else f'{check3_fail} FAILURES'}")

    return report, mismatches


# Check 5
def run_check5(table_a, table_c):
    """Compare accepted classification errors and missed-disease rates with step 13c."""
    mismatches = []
    report = []
    t09 = read_csv(TABDIR / "13c_T09_accepted_set_stats.csv")

    report.append("Check 5: T17 diagonal-sum / accepted-total == 13c_T09_accepted_set_"
                  "stats.csv's 'accuracy' (4 TRUTH_SOURCES); T19 'pct_accepted_normal' "
                  "== T09's 'miss_rate' (5 label-less sources)")
    fail = 0
    for source in TRUTH_SOURCES:
        sub_a = table_a[table_a["source"] == source]
        for condition, _ in CONDITIONS_ALL:
            sub_c = sub_a[sub_a["condition"] == condition]
            pred_rows = sub_c[sub_c["predicted_class"] != "deferred"]
            accepted_total = int(pred_rows["count"].sum())
            correct = int(pred_rows.loc[pred_rows["true_class"] == pred_rows["predicted_class"],
                                        "count"].sum())
            computed = correct / accepted_total if accepted_total else None
            t09_row = t09[(t09["source"] == source) & (t09["condition"] == str(condition))]
            if len(t09_row) != 1:
                mismatches.append(f"Check 5: {source} condition={condition}: expected 1 "
                                  f"matching T09 row, found {len(t09_row)}")
                fail += 1
                continue
            t09_val = t09_row["accuracy"].iloc[0]
            t09_val = None if pd.isna(t09_val) else float(t09_val)
            ok = (computed is None and t09_val is None) or \
                (computed is not None and t09_val is not None and abs(computed - t09_val) <= 1e-9)
            if not ok:
                mismatches.append(f"Check 5: {source} condition={condition}: "
                                  f"computed accuracy={computed} T09 accuracy={t09_val}")
                fail += 1

    for source in LABELLESS_SOURCES:
        sub_c19 = table_c[table_c["source"] == source]
        for condition, _ in CONDITIONS_ALL:
            row = sub_c19[sub_c19["condition"] == condition]
            n_acc = int(row["n_accepted"].iloc[0])
            n_acc_normal = int(row["n_accepted_normal"].iloc[0])
            computed = n_acc_normal / n_acc if n_acc else None
            t09_row = t09[(t09["source"] == source) & (t09["condition"] == str(condition))]
            if len(t09_row) != 1:
                mismatches.append(f"Check 5: {source} condition={condition}: expected 1 "
                                  f"matching T09 row, found {len(t09_row)}")
                fail += 1
                continue
            t09_val = t09_row["miss_rate"].iloc[0]
            t09_val = None if pd.isna(t09_val) else float(t09_val)
            ok = (computed is None and t09_val is None) or \
                (computed is not None and t09_val is not None and abs(computed - t09_val) <= 1e-9)
            if not ok:
                mismatches.append(f"Check 5: {source} condition={condition}: computed "
                                  f"pct_accepted_normal/100={computed} T09 miss_rate={t09_val}")
                fail += 1
    report.append(f"  {'PASS' if fail == 0 else f'{fail} FAILURES'}")
    return report, mismatches


# Check 6
def run_check6(merged):
    """Cross-check NEH per-image labels against the Label field in the source metadata."""
    mismatches = []
    report = []
    neh = merged[merged["source"] == "NEH"]
    ct = pd.crosstab(neh["true_label_name"], neh["Label"])
    is_diagonal = True
    for row_label in ct.index:
        nonzero_cols = [c for c in ct.columns if ct.loc[row_label, c] != 0]
        if nonzero_cols != [row_label]:
            is_diagonal = False
    report.append("Check 6: NEH true_label_name x Label cross-tab must be diagonal "
                  "(exactly one non-zero cell per row)")
    report.append(f"  {'PASS (diagonal)' if is_diagonal else 'MISMATCH -- not diagonal'}")
    if not is_diagonal:
        mismatches.append("Check 6: NEH true_label_name x Label cross-tab is not "
                          "diagonal -- full table:")
        for line in ct.to_string().splitlines():
            mismatches.append(f"  {line}")
    return report, mismatches


# Figures
def fig_f26():
    """Plot row-normalised confusion matrices including deferred cases for each condition."""
    df = read_csv(TABDIR / "17_T17_confusion_by_disease.csv")
    pred_cols = CLASSES + ["deferred"]
    cmap = plt.cm.Blues.with_extremes(bad="lightgrey")
    for source in TRUTH_SOURCES:
        sub_s = df[df["source"] == source]
        n_total = int(sub_s[sub_s["condition"].astype(str) == "1"]["count"].sum())

        fig, axes = plt.subplots(1, len(CONDITIONS_ALL), figsize=(3.1 * len(CONDITIONS_ALL), 5.5))
        for ax, (condition, label) in zip(axes, CONDITIONS_ALL):
            sub_c = sub_s[sub_s["condition"].astype(str) == str(condition)]
            mat_count = np.zeros((len(CLASSES), len(pred_cols)), dtype=int)
            mat_pct = np.full((len(CLASSES), len(pred_cols)), np.nan)
            for i, true_class in enumerate(CLASSES):
                for j, predicted_class in enumerate(pred_cols):
                    cell = sub_c[(sub_c["true_class"] == true_class)
                                & (sub_c["predicted_class"] == predicted_class)]
                    mat_count[i, j] = int(cell["count"].iloc[0])
                    pct = cell["row_pct"].iloc[0]
                    mat_pct[i, j] = np.nan if pd.isna(pct) else float(pct)

            masked = np.ma.masked_invalid(mat_pct)
            ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
            for i in range(len(CLASSES)):
                for j in range(len(pred_cols)):
                    n = mat_count[i, j]
                    pct = mat_pct[i, j]
                    text = f"{n}" if np.isnan(pct) else f"{n}\n({pct:.1f}%)"
                    colour = "white" if (not np.isnan(pct) and pct > 50) else "black"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color=colour)

            diag = sum(mat_count[i, i] for i in range(len(CLASSES)))
            accepted_total = int(mat_count[:, :len(CLASSES)].sum())
            acc = diag / accepted_total if accepted_total else float("nan")
            ax.set_xticks(range(len(pred_cols)))
            ax.set_xticklabels(pred_cols, rotation=30, ha="right", fontsize=9)
            if ax is axes[0]:
                row_totals = mat_count.sum(axis=1)
                yticklabels = [f"{c} (n=0)" if row_totals[i] == 0 else c
                              for i, c in enumerate(CLASSES)]
                ax.set_yticks(range(len(CLASSES)))
                ax.set_yticklabels(yticklabels, fontsize=9)
            else:
                ax.set_yticks([])
            title_label = "B2 Softmax confidence" if condition == "B2" else label
            ax.set_title(f"{condition} {title_label}\naccepted acc={acc:.3f}", fontsize=9)

        fig.suptitle(f"{source}  (n={n_total})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        safe = source.replace(" ", "_")
        fig.savefig(FIGDIR / f"17_F26_confusion_{safe}.png")
        plt.close(fig)


def fig_f27():
    """Plot accepted NORMAL rates on sources known to contain disease."""
    df = read_csv(TABDIR / "17_T19_missed_disease_labelless.csv")
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(LABELLESS_SOURCES))
    width = 0.13
    for i, (condition, label) in enumerate(CONDITIONS_ALL):
        vals = []
        for s in LABELLESS_SOURCES:
            row = df[(df["source"] == s) & (df["condition"].astype(str) == str(condition))]
            v = row["pct_accepted_normal"].iloc[0]
            vals.append(0.0 if pd.isna(v) else float(v))
        bars = ax.bar(x + (i - 2.5) * width, vals, width,
                      label=f"{condition} {label}", color=CONDITION_COLOURS_13C[condition],
                      alpha=0.88)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom",
                   fontsize=7)
    ax.set_xticks(x, LABELLESS_SOURCES, rotation=30, ha="right")
    ax.set_ylabel("pct_accepted_normal (%)")
    ax.set_title("Missed-disease proxy (accepted rows predicted NORMAL), by source and condition")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "17_F27_missed_disease_accepted.png")
    plt.close(fig)


def fig_f28():
    """Plot accepted classification error by true class and condition."""
    df = read_csv(TABDIR / "17_T17_confusion_by_disease.csv")
    cond_codes = [c for c, _ in CONDITIONS_ALL]
    x = np.arange(len(cond_codes))
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, source in zip(axes.flat, TRUTH_SOURCES):
        sub_s = df[df["source"] == source]
        for true_class in CLASSES:
            y = []
            for condition in cond_codes:
                sub_cell = sub_s[(sub_s["condition"].astype(str) == str(condition))
                                 & (sub_s["true_class"] == true_class)]
                pred_rows = sub_cell[sub_cell["predicted_class"] != "deferred"]
                accepted_for_class = int(pred_rows["count"].sum())
                if accepted_for_class == 0:
                    y.append(np.nan)
                else:
                    correct = int(pred_rows.loc[pred_rows["predicted_class"] == true_class,
                                                "count"].iloc[0])
                    y.append(1.0 - correct / accepted_for_class)
            if all(np.isnan(v) for v in y):
                continue
            ax.plot(x, y, marker="o", label=true_class)
        ax.set_xticks(x, [str(c) for c in cond_codes])
        ax.set_xlabel("Condition")
        ax.set_ylabel("Error rate among accepted")
        ax.set_title(source)
        ax.legend(fontsize=9)
    fig.suptitle("Per-class error rate among accepted rows, by condition", fontsize=14)
    fig.text(0.5, 0.005, "Denominator is accepted rows of that true class, not the "
             "class's full row count (accepted + deferred).", ha="center", fontsize=9,
             color="dimgray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(FIGDIR / "17_F28_per_class_accepted_error.png")
    plt.close(fig)


def fig_f29():
    """Plot correct, incorrect and deferred fractions by true class and condition."""
    df = read_csv(TABDIR / "17_T17_confusion_by_disease.csv")
    cond_codes = [c for c, _ in CONDITIONS_ALL]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    width = 0.8 / len(cond_codes)
    seen_labels = set()

    def maybe_label(name):
        if name in seen_labels:
            return None
        seen_labels.add(name)
        return name

    for ax, source in zip(axes.flat, TRUTH_SOURCES):
        sub_s = df[df["source"] == source]
        group_classes = []
        for true_class in CLASSES:
            total = int(sub_s[(sub_s["condition"].astype(str) == "1")
                              & (sub_s["true_class"] == true_class)]["count"].sum())
            if total > 0:
                group_classes.append(true_class)
        x_base = np.arange(len(group_classes))
        for gi, true_class in enumerate(group_classes):
            for ci, condition in enumerate(cond_codes):
                sub_cell = sub_s[(sub_s["condition"].astype(str) == str(condition))
                                 & (sub_s["true_class"] == true_class)]
                total = int(sub_cell["count"].sum())
                correct = int(sub_cell.loc[sub_cell["predicted_class"] == true_class,
                                           "count"].iloc[0])
                deferred = int(sub_cell.loc[sub_cell["predicted_class"] == "deferred",
                                            "count"].iloc[0])
                incorrect = total - correct - deferred
                pct_correct = correct / total * 100
                pct_incorrect = incorrect / total * 100
                pct_deferred = deferred / total * 100
                xpos = x_base[gi] + (ci - (len(cond_codes) - 1) / 2) * width
                ax.bar(xpos, pct_correct, width, color="tab:green",
                      label=maybe_label("accepted, correct"))
                ax.bar(xpos, pct_incorrect, width, bottom=pct_correct, color="tab:red",
                      label=maybe_label("accepted, incorrect"))
                ax.bar(xpos, pct_deferred, width, bottom=pct_correct + pct_incorrect,
                      color="lightgrey", label=maybe_label("deferred"))
        ax.set_xticks(x_base, group_classes)
        ax.set_ylabel("% of true-class total")
        ax.set_ylim(0, 100)
        ax.set_title(source)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.suptitle("Accepted-correct / accepted-incorrect / deferred, as % of each true "
                "class's total, by condition", fontsize=13, y=1.02)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3,
              fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIGDIR / "17_F29_accept_defer_breakdown.png")
    plt.close(fig)


def main():
    t0 = time.time()
    report = []
    report.append("=" * 78)
    report.append("17_confusion_by_disease.py -- by-disease confusion, nine sources, "
                  "six conditions")
    report.append("=" * 78)
    report.append("")
    report.append("'Accepted' definition, used throughout: condition 1 = every row in "
                  "the source (nothing deferred); conditions 2/3/4/5 = "
                  "~defer_condition{N}; condition B2 = ~defer_B2.")

    print("Building nine-source merged dataframe (13c's load_merged_df)...")
    merged = load_merged_df()
    print(f"  {len(merged)} rows across {merged['source'].nunique()} sources")

    print("Building Table A (17_T17_confusion_by_disease.csv)...")
    table_a = build_table_a(merged)
    print("Building Table B (17_T18_disease_vs_normal.csv)...")
    table_b = build_table_b(merged)
    print("Building Table C (17_T19_missed_disease_labelless.csv)...")
    table_c = build_table_c(merged)

    report.append("")
    report.append("-" * 78)
    report.append("Tables")
    report.append("-" * 78)
    report.append(f"17_T17_confusion_by_disease.csv ({len(table_a)} rows) -- "
                  "source, condition, true_class, predicted_class, count, row_pct. "
                  "4 TRUTH_SOURCES x 6 conditions x 4 true_class x 5 predicted_class "
                  "(4 CLASSES + 'deferred'). true_label_name/predicted_class_name read "
                  "from the merged 08b/13b slice-score rows (via load_merged_df); "
                  "row_pct denominator is the source's own true_class total, matching "
                  "13c_T02's own row-pct convention (round to 2dp).")
    report.append(f"17_T18_disease_vs_normal.csv ({len(table_b)} rows) -- "
                  "source, condition, true_state, predicted_state, count, row_pct. "
                  "Same 4 sources x 6 conditions, true_label_name/predicted_class_name "
                  "collapsed to Disease/Normal via .ne(\"NORMAL\"), the same mapping "
                  "13c_T03's build_t03 uses ().")
    report.append(f"17_T19_missed_disease_labelless.csv ({len(table_c)} rows) -- "
                  "source, condition, n, n_deferred, n_accepted, n_accepted_normal, "
                  "pct_accepted_normal. The 5 sources with no class-level truth. "
                  "n_accepted_normal/n_accepted is not known_diseased-filtered; "
                  "confirmed this run (see below) that every row in all 5 sources has "
                  "known_diseased==True, so no filter would change this ratio.")

    print("Confirming known_diseased==True on every row of the 5 label-less sources...")
    kd_lines = []
    for source in LABELLESS_SOURCES:
        sub = merged[merged["source"] == source]
        all_true = bool((sub["known_diseased"] == True).all())
        kd_lines.append(f"  {source}: known_diseased==True on {int((sub['known_diseased'] == True).sum())} "
                        f"of {len(sub)} rows -- {'all True' if all_true else 'NOT all True'}")
    report.append("")
    report.append("-" * 78)
    report.append("known_diseased check, the 5 label-less sources")
    report.append("-" * 78)
    report += kd_lines

    print("Running regression checks 1-3...")
    check_report, mismatches = run_checks(merged, table_a)

    print("Running regression checks 5-6...")
    check5_report, check5_mismatches = run_check5(table_a, table_c)
    check6_report, check6_mismatches = run_check6(merged)
    check_report = check_report + [""] + check5_report + [""] + check6_report
    mismatches = mismatches + check5_mismatches + check6_mismatches

    report.append("")
    report.append("-" * 78)
    report.append("Regression checks")
    report.append("-" * 78)
    report += check_report

    report.append("")
    report.append("-" * 78)
    if mismatches:
        report.append("MISMATCHES FOUND -- printed verbatim, not adjusted:")
        report += [f"  {m}" for m in mismatches]
    else:
        report.append("No mismatches found in any of the 5 regression checks.")
    report.append("-" * 78)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("Building figures F26-F29 (read fresh from 17_T17/17_T19, no in-memory reuse)...")
    fig_f26()
    fig_f27()
    fig_f28()
    fig_f29()

    report.append("")
    report.append("-" * 78)
    report.append("Figures (all read fresh from the CSVs listed, not from any "
                  "in-memory dataframe from the table-building step above)")
    report.append("-" * 78)
    report.append("F26 17_F26_confusion_<source>.png (x4, one per TRUTH_SOURCE) -- "
                  "17_T17_confusion_by_disease.csv's true_class, predicted_class, count, "
                  "row_pct columns, one 4x5 heat map per condition.")
    report.append("F27 17_F27_missed_disease_accepted.png -- "
                  "17_T19_missed_disease_labelless.csv's pct_accepted_normal column, "
                  "grouped bar, 5 label-less sources x 6 conditions.")
    report.append("F28 17_F28_per_class_accepted_error.png -- "
                  "17_T17_confusion_by_disease.csv's true_class/predicted_class/count "
                  "columns (predicted_class != 'deferred' rows only), 1 - correct/"
                  "accepted_for_class per true_class per condition, 4 TRUTH_SOURCES.")
    report.append("F29 17_F29_accept_defer_breakdown.png -- "
                  "17_T17_confusion_by_disease.csv's true_class/predicted_class/count "
                  "columns, all 3 predicted_class categories (own class = correct, "
                  "other 3 classes = incorrect, 'deferred'), normalised to each true "
                  "class's own total, 4 TRUTH_SOURCES.")

    elapsed = time.time() - t0
    report.append("")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    report_path = TABDIR / "17_confusion_by_disease.txt"
    report_path.write_text("\n".join(str(l) for l in report) + "\n")
    print(f"\nWrote {report_path.relative_to(PROJECT)}")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
