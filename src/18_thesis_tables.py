"""Format the final thesis tables from the recorded evaluation results.

Includes classifier performance, AUGRC, matched coverage, window risk and
operating points. Checks source metrics and ranks before writing the tables.
"""

from paths import PROJECT, read_csv

import importlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TABDIR = PROJECT / "results" / "tables"

sys.path.insert(0, str(Path(__file__).resolve().parent))
_ev13c = importlib.import_module("13c_evaluate_all_sources")

SOURCES_ORDER = _ev13c.SOURCES_ORDER
CONDITIONS_AURC = _ev13c.CONDITIONS_AURC

CONDITION_DISPLAY = {
    1: "Baseline", 2: "OOD detection", 3: "Uncertainty",
    4: "Conformal prediction", 5: "All signals", "B2": "Softmax reference",
}

CLASSES = _ev13c.CLASSES

T02_SOURCES = _ev13c.TRUTH_SOURCES
T02_FILES = [f"13c_T02_confusion_{s.replace(' ', '_')}.csv" for s in T02_SOURCES]

REQUIRED_INPUTS = [
    "13c_T01_source_summary.csv", "13c_T09_accepted_set_stats.csv",
    "15_augrc.csv", "15b_augrc_window.csv",
    "14_T16_matched_coverage.csv", "14_T13_expected_cost.csv",
    "15_augrc.txt", "15b_grc.txt",
] + T02_FILES


def check_inputs_exist():
    missing = [f for f in REQUIRED_INPUTS if not (TABDIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"required input(s) missing from {TABDIR}: {missing} -- "
            f"rerun the scripts that produce them before this one.")


def load_csv(name):
    """Load a result table, treating all condition codes as strings."""
    df = read_csv(TABDIR / name)
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str)
    return df


def one_row(df, **filters):
    """Return one matching row; raise if the match is absent or ambiguous."""
    mask = pd.Series(True, index=df.index)
    for k, v in filters.items():
        mask &= (df[k] == v)
    sub = df[mask]
    if len(sub) != 1:
        raise ValueError(f"expected exactly 1 row for {filters}, found {len(sub)}")
    return sub.iloc[0]


def dense_rank(df, group_col, value_col):
    """Compute ascending dense ranks within each group."""
    return df.groupby(group_col)[value_col].rank(method="dense", ascending=True)


# Regression-check target parsing
def parse_augrc_targets(path):
    """Read reference mean ranks from the saved full-range AUGRC report."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "(b) like-for-like" in l)
    pattern = re.compile(r"condition (\S+): mean rank_augrc=([\d.]+)")
    targets = {}
    for line in lines[start + 1:]:
        m = pattern.search(line)
        if m is None:
            if targets:
                break
            continue
        targets[m.group(1)] = float(m.group(2))
    if not targets:
        raise ValueError(f"no '(b) like-for-like' targets parsed from {path}")
    return targets


def parse_window_targets(path):
    """Read reference mean ranks from the saved coverage-window report."""
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if "Full-range vs window mean rank" in l)
    pattern = re.compile(
        r"condition (\S+): mean rank_augrc \(full range\)=[\d.]+\s+"
        r"mean rank_augrc_window \(\[0\.7,0\.9\]\)=([\d.]+)")
    targets = {}
    for line in lines[start + 1:]:
        m = pattern.search(line)
        if m is None:
            if targets:
                break
            continue
        targets[m.group(1)] = float(m.group(2))
    if not targets:
        raise ValueError(f"no window-rank targets parsed from {path}")
    return targets


# Mean-rank helper
def mean_rank_row(df, rank_cols, base_error_col):
    """Average ranks over sources with nonzero baseline error and list excluded sources."""
    included = df[df[base_error_col] > 0]
    excluded = sorted(set(df["source"]) - set(included["source"]))
    row = {"source": "Mean rank"}
    for col in rank_cols:
        row[col] = float(included[col].mean())
    return row, excluded


# T20
def load_confusion(source):
    """Load the recorded four-class confusion matrix for a source."""
    path = TABDIR / f"13c_T02_confusion_{source.replace(' ', '_')}.csv"
    return read_csv(path).set_index("true_label_name")


def per_class_f1(cm):
    """Compute F1 for classes with ground-truth support; exclude absent classes from the mean."""
    pred_cols = [f"pred_{c}_n" for c in CLASSES]
    f1 = {}
    present = []
    for k in CLASSES:
        row_total = cm.loc[k, pred_cols].sum()
        if row_total == 0:
            continue
        present.append(k)
        tp = cm.loc[k, f"pred_{k}_n"]
        fp = cm[f"pred_{k}_n"].sum() - tp
        fn = row_total - tp
        f1[k] = 2 * tp / (2 * tp + fp + fn)
    return f1, present


def check_f1_against_sklearn(source, cm, f1, present):
    """Expand confusion counts into label pairs and cross-check F1 with scikit-learn."""
    pred_cols = [f"pred_{c}_n" for c in CLASSES]
    y_true, y_pred = [], []
    for true_cls in CLASSES:
        for pred_cls in CLASSES:
            count = int(cm.loc[true_cls, f"pred_{pred_cls}_n"])
            y_true += [true_cls] * count
            y_pred += [pred_cls] * count
    sklearn_macro = f1_score(y_true, y_pred, labels=present, average="macro")
    hand_macro = float(np.mean([f1[k] for k in present]))
    if abs(sklearn_macro - hand_macro) > 1e-9:
        raise AssertionError(
            f"{source}: macro-F1 self-check failed -- hand-computed "
            f"{hand_macro:.9f} vs sklearn {sklearn_macro:.9f} "
            f"(classes scored: {present}).")
    return hand_macro


def build_t20(t01, t09, t13):
    rows = []
    f1_report_lines = []
    for source in SOURCES_ORDER:
        r01 = one_row(t01, source=source)
        r09 = one_row(t09, source=source, condition="1")
        r13 = one_row(t13, source=source, condition="1")
        row = {
            "source": source, "n": int(r01["n"]), "vendor": r01["vendor"],
            "macro_f1": np.nan,
            "f1_CNV": np.nan, "f1_DME": np.nan, "f1_DRUSEN": np.nan, "f1_NORMAL": np.nan,
            "miss_rate": r09["miss_rate"],
            "risk_definition": r13["risk_definition"],
        }
        if source in T02_SOURCES:
            cm = load_confusion(source)
            f1, present = per_class_f1(cm)
            macro = check_f1_against_sklearn(source, cm, f1, present)
            row["macro_f1"] = macro
            for k, v in f1.items():
                row[f"f1_{k}"] = v
            f1_report_lines.append(
                f"  {source}: classes scored = {present} (row total > 0 in "
                f"13c_T02_confusion_{source.replace(' ', '_')}.csv); "
                f"f1 = {{{', '.join(f'{k}: {v:.4f}' for k, v in f1.items())}}}; "
                f"macro_f1 = {macro:.4f} (hand-computed == sklearn, checked above)")
        rows.append(row)
    df = pd.DataFrame(rows, columns=["source", "n", "vendor", "macro_f1",
                                     "f1_CNV", "f1_DME", "f1_DRUSEN", "f1_NORMAL",
                                     "miss_rate", "risk_definition"])
    return df, f1_report_lines


# T21
def build_t21(augrc):
    check_rank_reuse_augrc(augrc)

    rows = []
    for source in SOURCES_ORDER:
        row = {"source": source,
               "base_error": one_row(augrc, source=source,
                                     condition=str(CONDITIONS_AURC[0]))["base_error"]}
        for c in CONDITIONS_AURC:
            r = one_row(augrc, source=source, condition=str(c))
            row[f"augrc_{c}_x1000"] = r["augrc_x1000"]
            row[f"rank_{c}"] = r["rank_augrc"]
        rows.append(row)
    df = pd.DataFrame(rows)

    rank_cols = [f"rank_{c}" for c in CONDITIONS_AURC]
    mean_row, excluded = mean_rank_row(df, rank_cols, "base_error")
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df, excluded


def check_rank_reuse_augrc(augrc):
    """Verify saved full-range ranks against recomputed dense ranks."""
    recomputed = dense_rank(augrc, "source", "augrc").astype(int)
    existing = augrc["rank_augrc"].astype(int)
    mismatches = int((recomputed.values != existing.values).sum())
    if mismatches:
        raise AssertionError(
            f"rank_augrc reuse check failed: {mismatches} of {len(augrc)} rows "
            f"disagree between 15_augrc.csv's own rank_augrc and a fresh "
            f"dense_rank(source, augrc) recomputation.")
    return mismatches


# T22
def build_t22(t16):
    coverages = sorted(t16["coverage"].unique())
    expected = [0.70, 0.80, 0.90]
    if [round(c, 2) for c in coverages] != expected:
        raise ValueError(
            f"14_T16_matched_coverage.csv's own coverage values {coverages} "
            f"do not match the three levels this table expects ({expected}).")

    blocks = []
    per_cov_means = {}
    for cov in coverages:
        sub_cov = t16[t16["coverage"] == cov]
        ranked_long = sub_cov[sub_cov["condition"].isin([str(c) for c in CONDITIONS_AURC])].copy()
        ranked_long["rank"] = dense_rank(ranked_long, "source", "accepted_error_rate")

        rows = []
        for source in SOURCES_ORDER:
            row = {"coverage": cov, "source": source}
            base = one_row(sub_cov, source=source,
                           condition=str(CONDITIONS_AURC[0]))["baseline_error"]
            row["baseline_error"] = base
            for c in CONDITIONS_AURC:
                r = one_row(sub_cov, source=source, condition=str(c))
                row[f"accepted_error_rate_{c}"] = r["accepted_error_rate"]
                row[f"rank_{c}"] = one_row(ranked_long, source=source,
                                           condition=str(c))["rank"]
            rows.append(row)
        block_df = pd.DataFrame(rows)
        rank_cols = [f"rank_{c}" for c in CONDITIONS_AURC]
        mean_row, excluded = mean_rank_row(block_df, rank_cols, "baseline_error")
        mean_row["coverage"] = cov
        per_cov_means[cov] = {col: mean_row[col] for col in rank_cols}
        block_df = pd.concat([block_df, pd.DataFrame([mean_row])], ignore_index=True)
        blocks.append(block_df)

    overall_row = {"coverage": "all", "source": "Mean rank (all 3 coverages)"}
    for c in CONDITIONS_AURC:
        col = f"rank_{c}"
        overall_row[col] = float(np.mean([per_cov_means[cov][col] for cov in coverages]))
    blocks.append(pd.DataFrame([overall_row]))

    df = pd.concat(blocks, ignore_index=True)
    return df, excluded


# T23
def build_t23(window):
    check_rank_reuse_window(window)

    rows = []
    for source in SOURCES_ORDER:
        row = {"source": source,
               "base_error": one_row(window, source=source,
                                     condition=str(CONDITIONS_AURC[0]))["base_error"]}
        for c in CONDITIONS_AURC:
            r = one_row(window, source=source, condition=str(c))
            row[f"augrc_window_{c}_x1000"] = r["augrc_window_x1000"]
            row[f"rank_{c}"] = r["rank_augrc_window"]
        rows.append(row)
    df = pd.DataFrame(rows)

    rank_cols = [f"rank_{c}" for c in CONDITIONS_AURC]
    mean_row, excluded = mean_rank_row(df, rank_cols, "base_error")
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df, excluded


def check_rank_reuse_window(window):
    """Verify non-missing saved window ranks against recomputed dense ranks."""
    recomputed = dense_rank(window, "source", "augrc_window")
    has_existing = window["rank_augrc_window"].notna()
    existing = window.loc[has_existing, "rank_augrc_window"].astype(int)
    recomputed_sub = recomputed.loc[has_existing].astype(int)
    mismatches = int((recomputed_sub.values != existing.values).sum())
    if mismatches:
        raise AssertionError(
            f"rank_augrc_window reuse check failed: {mismatches} of "
            f"{has_existing.sum()} non-blank rows disagree between "
            f"15b_augrc_window.csv's own rank_augrc_window and a fresh "
            f"dense_rank(source, augrc_window) recomputation.")
    return mismatches


# T24
def build_t24(t13):
    ranked_long = t13[t13["condition"].isin([str(c) for c in CONDITIONS_AURC])].copy()
    ranked_long["rank"] = dense_rank(ranked_long, "source", "accepted_error_rate")

    rows = []
    for source in SOURCES_ORDER:
        row = {"source": source}
        for c in [1] + CONDITIONS_AURC:
            r = one_row(t13, source=source, condition=str(c))
            row[f"coverage_{c}"] = 1.0 - r["deferral_rate"]
            row[f"selective_risk_{c}"] = r["accepted_error_rate"]
            if c != 1:
                row[f"rank_{c}"] = one_row(ranked_long, source=source,
                                           condition=str(c))["rank"]
        rows.append(row)
    df = pd.DataFrame(rows)

    rank_cols = [f"rank_{c}" for c in CONDITIONS_AURC]
    mean_row, excluded = mean_rank_row(df, rank_cols, "selective_risk_1")
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return df, excluded


# Main
def main():
    t0 = time.time()
    report = []
    report.append("=" * 78)
    report.append("18_thesis_tables.py -- thesis-format result tables (source x condition)")
    report.append("=" * 78)
    report.append("")
    report.append("Pure re-shape: every value below is read from an existing table, not "
                  "recomputed. No metric value or regression-check target is a Python "
                  "literal in this script -- the two mean-rank targets are parsed out of "
                  "15_augrc.txt / 15b_grc.txt at runtime.")

    print("Checking required inputs exist...")
    check_inputs_exist()

    print("Loading source tables...")
    t01 = load_csv("13c_T01_source_summary.csv")
    t09 = load_csv("13c_T09_accepted_set_stats.csv")
    augrc = load_csv("15_augrc.csv")
    window = load_csv("15b_augrc_window.csv")
    t16 = load_csv("14_T16_matched_coverage.csv")
    t13 = load_csv("14_T13_expected_cost.csv")

    # Regression checks (before any file is written)
    report.append("")
    report.append("-" * 78)
    report.append("Regression checks (run before any output file is written)")
    report.append("-" * 78)

    print("Rank-reuse checks (T21 vs 15_augrc.csv, T23 vs 15b_augrc_window.csv)...")
    n_mis_augrc = check_rank_reuse_augrc(augrc)
    report.append(f"1a. rank_augrc reuse: dense_rank(source, augrc) recomputed on "
                  f"15_augrc.csv vs its own rank_augrc column -- "
                  f"{'PASS' if n_mis_augrc == 0 else 'FAIL'} ({n_mis_augrc} mismatches "
                  f"of {len(augrc)} rows)")
    n_mis_window = check_rank_reuse_window(window)
    non_blank = int(window["rank_augrc_window"].notna().sum())
    report.append(f"1b. rank_augrc_window reuse: dense_rank(source, augrc_window) "
                  f"recomputed on 15b_augrc_window.csv vs its own rank_augrc_window "
                  f"column, non-blank rows only -- "
                  f"{'PASS' if n_mis_window == 0 else 'FAIL'} ({n_mis_window} mismatches "
                  f"of {non_blank} non-blank rows)")

    print("Parsing mean-rank targets from 15_augrc.txt and 15b_grc.txt...")
    augrc_targets = parse_augrc_targets(TABDIR / "15_augrc.txt")
    window_targets = parse_window_targets(TABDIR / "15b_grc.txt")

    # T21/T23 built here so their mean_rank rows exist for the comparison below.
    t21, t21_excluded = build_t21(augrc)
    t23, t23_excluded = build_t23(window)

    t21_mean_row = t21[t21["source"] == "Mean rank"].iloc[0]
    t23_mean_row = t23[t23["source"] == "Mean rank"].iloc[0]

    report.append("")
    report.append("2a. T21 mean_rank vs 15_augrc.txt's '(b) like-for-like' targets:")
    check2a_fail = 0
    for c in CONDITIONS_AURC:
        computed = float(t21_mean_row[f"rank_{c}"])
        target = augrc_targets[str(c)]
        ok = abs(computed - target) < 1e-9
        check2a_fail += 0 if ok else 1
        report.append(f"    condition {c}: computed={computed:.4f}  target={target:.4f}  "
                      f"{'PASS' if ok else 'FAIL'}")

    report.append("2b. T23 mean_rank vs 15b_grc.txt's window-rank targets:")
    check2b_fail = 0
    for c in CONDITIONS_AURC:
        computed = float(t23_mean_row[f"rank_{c}"])
        target = window_targets[str(c)]
        ok = abs(computed - target) < 1e-9
        check2b_fail += 0 if ok else 1
        report.append(f"    condition {c}: computed={computed:.4f}  target={target:.4f}  "
                      f"{'PASS' if ok else 'FAIL'}")

    if n_mis_augrc or n_mis_window or check2a_fail or check2b_fail:
        report.append("")
        report.append("MISMATCH -- stopping before any output file is written.")
        (TABDIR / "18_thesis_tables.txt").write_text("\n".join(report) + "\n")
        raise AssertionError("one or more regression checks failed; see "
                             "results/tables/18_thesis_tables.txt")

    # Remaining tables
    print("Building T20, T22, T24...")
    t20, t20_f1_lines = build_t20(t01, t09, t13)
    t22, t22_excluded = build_t22(t16)
    t24, t24_excluded = build_t24(t13)

    # Write outputs
    print("Writing tables...")
    t20.to_csv(TABDIR / "18_T20_backbone_no_deferral.csv", index=False)
    t21.to_csv(TABDIR / "18_T21_augrc.csv", index=False)
    t22.to_csv(TABDIR / "18_T22_matched_coverage.csv", index=False)
    t23.to_csv(TABDIR / "18_T23_augrc_window.csv", index=False)
    t24.to_csv(TABDIR / "18_T24_operating_point.csv", index=False)

    # Report: provenance
    report.append("")
    report.append("-" * 78)
    report.append("Provenance")
    report.append("-" * 78)
    report.append("T20 18_T20_backbone_no_deferral.csv -- n/vendor from "
                  "13c_T01_source_summary.csv; macro_f1/f1_{class} computed from "
                  "13c_T02_confusion_<source>.csv's own TP/FP/FN per class (row total "
                  "> 0 only), cross-checked against sklearn.metrics.f1_score on an "
                  "expanded sample list (see per-source detail below); blank for the "
                  "five sources with no T02 file; miss_rate from 13c_T09_accepted_set_"
                  "stats.csv's condition==1 row; risk_definition from 14_T13_expected_"
                  "cost.csv's condition==1 row.")
    report += t20_f1_lines
    report.append("T21 18_T21_augrc.csv -- base_error/augrc_x1000 from 15_augrc.csv; "
                  "rank_{c} is that file's own rank_augrc column, reused after check 1a.")
    report.append("T22 18_T22_matched_coverage.csv -- baseline_error/accepted_error_rate "
                  "from 14_T16_matched_coverage.csv; rank_{c} computed fresh here "
                  "(dense_rank), not reused, since no matched-coverage rank exists "
                  "anywhere else in the pipeline.")
    report.append("T23 18_T23_augrc_window.csv -- base_error/augrc_window_x1000 from "
                  "15b_augrc_window.csv; rank_{c} is that file's own rank_augrc_window "
                  "column, reused after check 1b.")
    report.append("T24 18_T24_operating_point.csv -- coverage_{c} (=1-deferral_rate) and "
                  "selective_risk_{c} (=accepted_error_rate) from 14_T13_expected_"
                  "cost.csv; rank_{c} computed fresh here (dense_rank on "
                  "selective_risk), condition 1 excluded from ranking as the baseline.")

    report.append("")
    report.append("-" * 78)
    report.append("Mean-rank exclusions (derived from each table's own data)")
    report.append("-" * 78)
    for name, excl, col in [("T21", t21_excluded, "15_augrc.csv base_error"),
                            ("T22", t22_excluded, "14_T16_matched_coverage.csv baseline_error"),
                            ("T23", t23_excluded, "15b_augrc_window.csv base_error"),
                            ("T24", t24_excluded, "14_T13_expected_cost.csv condition==1 "
                                                  "accepted_error_rate")]:
        report.append(f"  {name}: excluded {excl} (that source's {col} is 0)")

    elapsed = time.time() - t0
    report.append("")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    (TABDIR / "18_thesis_tables.txt").write_text("\n".join(report) + "\n")
    print(f"\nWrote results/tables/18_T20..T24_*.csv and 18_thesis_tables.txt")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
