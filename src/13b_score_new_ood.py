"""Score NEH and OCTDL using the frozen parameters from steps 05-07.

Check earlier-source scores and record per-image predictions, signals and decisions.
"""

from paths import PROJECT, read_csv

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# import 08b (digit-prefixed filename, so importlib not `import`)
sys.path.insert(0, str(Path(__file__).resolve().parent))
_ood08b = importlib.import_module("08b_score_ood")

CLASSES = _ood08b.CLASSES
N_CLASSES = _ood08b.N_CLASSES
CACHE = _ood08b.CACHE
TABDIR = _ood08b.TABDIR
cosine_score = _ood08b.cosine_score
mahalanobis_score = _ood08b.mahalanobis_score
score_detectors = _ood08b.score_detectors
mutual_information = _ood08b.mutual_information
compute_cp_setsize = _ood08b.compute_cp_setsize
apply_policy = _ood08b.apply_policy
apply_b2 = _ood08b.apply_b2
load_frozen_params = _ood08b.load_frozen_params
regression_test = _ood08b.regression_test
build_slice_table = _ood08b.build_slice_table
build_retouch_index = _ood08b.build_retouch_index
build_oct5k_index = _ood08b.build_oct5k_index

NUMERIC_DIFF_COLS = ["predicted_class", "softmax_confidence", "mutual_info",
                     "ood_cosine", "ood_maha_combined", "cp_setsize",
                     "s_condition2", "s_condition3", "s_condition4", "s_condition5"]
BOOL_DIFF_COLS = ["defer_condition2", "defer_condition3", "defer_condition4",
                  "defer_condition5", "defer_B2"]
STATS_COLS = ["ood_cosine", "ood_maha_combined", "softmax_confidence",
             "mutual_info", "cp_setsize"]

NEH_META_COLS = ["global_patient_id", "Class", "Label", "Eye", "B-scan", "ext"]
OCTDL_META_COLS = ["patient_id", "disease", "subcategory", "condition",
                   "format", "eye", "sex", "year", "file_name"]


# Part 4: NEH/OCTDL index frames
def build_index_frames():
    """Load NEH and OCTDL cache indices in the schema used by the scoring functions."""
    neh = read_csv(CACHE / "neh_index.csv")
    neh["volume_id"] = None
    neh["slice_index"] = None
    neh["vendor"] = None
    neh["set"] = "n/a"
    neh["known_diseased"] = neh["true_label_name"] != "NORMAL"
    neh = neh[["path", "volume_id", "slice_index", "vendor", "set",
              "true_label", "true_label_name", "known_diseased", "label_group"]
              + NEH_META_COLS]

    octdl = read_csv(CACHE / "octdl_index.csv")
    octdl["volume_id"] = None
    octdl["slice_index"] = None
    octdl["vendor"] = None
    octdl["set"] = "n/a"
    octdl["known_diseased"] = octdl["true_label_name"] != "NORMAL"
    octdl = octdl[["path", "volume_id", "slice_index", "vendor", "set",
                   "true_label", "true_label_name", "known_diseased", "label_group"]
                   + OCTDL_META_COLS]
    return neh, octdl


# Part 3: regression test
def diff_against_reference(computed_df, reference_df, label):
    lines = [f"{label}: n_computed={len(computed_df)}  n_reference={len(reference_df)}"]
    if len(computed_df) != len(reference_df):
        lines.append("  [ROW COUNT MISMATCH] cannot diff column-by-column.")
        return lines, False
    computed_df = computed_df.reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)
    path_mismatch = int((computed_df["path"].astype(str) != reference_df["path"].astype(str)).sum())
    lines.append(f"  path alignment: {path_mismatch} of {len(computed_df)} rows mismatched"
                + ("  [ROW ORDER MISMATCH]" if path_mismatch else "  (all aligned)"))
    all_zero = (path_mismatch == 0)
    for col in NUMERIC_DIFF_COLS:
        d = (computed_df[col].to_numpy(dtype=float) - reference_df[col].to_numpy(dtype=float))
        max_abs = float(np.abs(d).max())
        all_zero &= (max_abs < 1e-9)
        lines.append(f"  {col:<22s} max_abs_diff = {max_abs:.3e}")
    for col in BOOL_DIFF_COLS:
        mismatch = int((computed_df[col].to_numpy() != reference_df[col].to_numpy()).sum())
        all_zero &= (mismatch == 0)
        lines.append(f"  {col:<22s} mismatched rows = {mismatch} of {len(computed_df)}")
    return lines, all_zero


def run_regression_tests(ood_params, conformal_params, policy_params, selected_ratio):
    lines = []
    lines.append("Layer 1: 08b's own test-split gate (imported regression_test)")
    reg_df, reg_ok, test_cos, test_maha = regression_test(ood_params)
    lines.append(reg_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append(f"  PASS={reg_ok}")

    lines.append("")
    lines.append("Layer 2: RETOUCH/OCT5k rebuilt via this script's own call to "
                 "build_slice_table, diffed against 08b_slice_scores.csv's "
                 "already-recorded rows for those sources. Reported for the "
                 "record, not a pass/fail gate (unlike Layer 1) -- see note below.")
    ref_all = read_csv(TABDIR / "08b_slice_scores.csv")

    retouch_index = build_retouch_index()
    retouch_computed = build_slice_table("retouch_linear_fixed", "RETOUCH", retouch_index,
                                         ood_params, conformal_params, policy_params, selected_ratio)
    retouch_ref = ref_all[ref_all["dataset"] == "RETOUCH"]
    r_lines, r_exact = diff_against_reference(retouch_computed, retouch_ref, "RETOUCH")
    lines += ["  " + l for l in r_lines]

    oct5k_index = build_oct5k_index()
    oct5k_computed = build_slice_table("oct5k", "OCT5k", oct5k_index, ood_params,
                                       conformal_params, policy_params, selected_ratio)
    oct5k_ref = ref_all[ref_all["dataset"] == "OCT5k"]
    o_lines, o_exact = diff_against_reference(oct5k_computed, oct5k_ref, "OCT5k")
    lines += ["  " + l for l in o_lines]

    lines.append(f"  Layer 2 bitwise-exact (diff < 1e-9 on every float column, 0 "
                f"row/decision mismatches everywhere): RETOUCH={r_exact}  OCT5k={o_exact}")
    return lines, reg_ok, ref_all


# Part 5: grouped stats + predictions
def format_grouped_stats(df, group_cols, title):
    lines = [title]
    g = df.groupby(group_cols)[STATS_COLS].agg(["mean", "std"])
    lines.append(g.to_string(float_format=lambda v: f"{v:.4f}"))
    return lines


def prediction_verdicts(new_df, ref_all):
    lines = []

    octdl = new_df[new_df["dataset"] == "OCTDL"]

    in_label_mean = float(octdl.loc[octdl["label_group"] == "in-label", "ood_maha_combined"].mean())
    novel_mean = float(octdl.loc[octdl["label_group"] == "novel-class", "ood_maha_combined"].mean())
    p1 = novel_mean < in_label_mean
    lines.append("Prediction 1: OCTDL novel-class ood_maha_combined mean < OCTDL in-label mean "
                "(lower = further OOD, since higher = more in-distribution)")
    lines.append(f"  in-label mean={in_label_mean:.4f}   novel-class mean={novel_mean:.4f}")
    lines.append(f"  VERDICT: {p1}")

    neh_mean_for_span = float(new_df.loc[new_df["dataset"] == "NEH", "ood_maha_combined"].mean())
    octdl_overall_mean_for_span = float(octdl["ood_maha_combined"].mean())
    seven_source_means = {"Kermany-test": float(ref_all.loc[ref_all["dataset"] == "Kermany-test", "ood_maha_combined"].mean())}
    for vendor in ["Spectralis", "Cirrus", "Topcon"]:
        sub = ref_all[(ref_all["dataset"] == "RETOUCH") & (ref_all["vendor"] == vendor)]
        seven_source_means[f"RETOUCH-{vendor}"] = float(sub["ood_maha_combined"].mean())
    seven_source_means["Rasti"] = float(ref_all.loc[ref_all["dataset"] == "OCT5k", "ood_maha_combined"].mean())
    seven_source_means["NEH"] = neh_mean_for_span
    seven_source_means["OCTDL"] = octdl_overall_mean_for_span
    max_source = max(seven_source_means, key=seven_source_means.get)
    min_source = min(seven_source_means, key=seven_source_means.get)
    span = seven_source_means[max_source] - seven_source_means[min_source]
    lines.append(f"  Gap between the two group means: {abs(novel_mean - in_label_mean):.4f}. "
                f"For reference, across this project's seven sources, "
                f"ood_maha_combined means range from {max_source} "
                f"({seven_source_means[max_source]:+.4f}) to {min_source} "
                f"({seven_source_means[min_source]:+.4f}), a span of {span:.4f}.")

    # -- reference RETOUCH vendor means (re-read from 08b_slice_scores.csv) --
    retouch = ref_all[ref_all["dataset"] == "RETOUCH"]
    topcon_mean = float(retouch.loc[retouch["vendor"] == "Topcon", "ood_maha_combined"].mean())
    spectralis_mean = float(retouch.loc[retouch["vendor"] == "Spectralis", "ood_maha_combined"].mean())

    # -- Prediction 2 --
    octdl_overall_mean = float(octdl["ood_maha_combined"].mean())
    lines.append("")
    lines.append("Prediction 2: OCTDL overall ood_maha_combined mean (no inequality judged)")
    lines.append(f"  OCTDL overall mean={octdl_overall_mean:.4f}   "
                f"RETOUCH-Topcon mean={topcon_mean:.4f}   "
                f"RETOUCH-Spectralis mean={spectralis_mean:.4f}")

    # -- Prediction 3 --
    neh = new_df[new_df["dataset"] == "NEH"]
    neh_mean = float(neh["ood_maha_combined"].mean())
    p3a = neh_mean > topcon_mean
    p3b = abs(neh_mean - spectralis_mean) < abs(neh_mean - topcon_mean)
    lines.append("")
    lines.append("Prediction 3: NEH ood_maha_combined mean close to RETOUCH-Spectralis, "
                "far from RETOUCH-Topcon")
    lines.append(f"  NEH mean={neh_mean:.4f}   RETOUCH-Topcon mean={topcon_mean:.4f}   "
                f"RETOUCH-Spectralis mean={spectralis_mean:.4f}")
    lines.append(f"  VERDICT (NEH_mean > Topcon_mean, i.e. materially less OOD than Topcon): {p3a}")
    lines.append(f"  VERDICT (|NEH_mean - Spectralis_mean| < |NEH_mean - Topcon_mean|, "
                f"i.e. closer to Spectralis than to Topcon): {p3b}")

    # -- Prediction 4 --
    oct5k = ref_all[ref_all["dataset"] == "OCT5k"]
    oct5k_drusen = oct5k[oct5k["true_label_name"] == "DRUSEN"]
    oct5k_drusen_acc = float((oct5k_drusen["predicted_class_name"] == oct5k_drusen["true_label_name"]).mean())

    neh_cm = pd.crosstab(neh["true_label_name"], neh["predicted_class_name"])
    neh_cm = neh_cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)

    neh_class_acc = {}
    for c in CLASSES:
        sub = neh[neh["true_label_name"] == c]
        if len(sub) == 0:
            neh_class_acc[c] = None
        else:
            neh_class_acc[c] = float((sub["predicted_class_name"] == sub["true_label_name"]).mean())

    p4 = (neh_class_acc["DRUSEN"] is not None) and (neh_class_acc["DRUSEN"] > oct5k_drusen_acc)
    lines.append("")
    lines.append("Prediction 4: NEH DRUSEN accuracy > OCT5k/Rasti DRUSEN accuracy")
    lines.append(f"  OCT5k/Rasti DRUSEN accuracy = {oct5k_drusen_acc:.4f} "
                f"({int((oct5k_drusen['predicted_class_name']==oct5k_drusen['true_label_name']).sum())}"
                f"/{len(oct5k_drusen)})")
    lines.append(f"  NEH DRUSEN accuracy = {neh_class_acc['DRUSEN']:.4f}")
    lines.append(f"  VERDICT: {p4}")
    lines.append("")
    lines.append("  NEH per-class accuracy (NEH has no DME ground truth -- row is "
                "structurally empty, not silently absent):")
    for c in CLASSES:
        acc = neh_class_acc[c]
        lines.append(f"    {c:<8s} n={int((neh['true_label_name']==c).sum()):<6d} "
                    f"accuracy={'n/a (no ground truth)' if acc is None else f'{acc:.4f}'}")
    lines.append("")
    lines.append("  NEH confusion matrix (rows=true_label_name, columns=predicted_class_name):")
    lines.append(neh_cm.to_string())

    return lines


def main():
    TABDIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 78)
    report.append("13b_score_new_ood.py -- NEH + OCTDL scoring")
    report.append("=" * 78)
    report.append("")
    ood_params, conformal_params, policy_params = load_frozen_params()
    selected_ratio = policy_params["selected_cost_ratio"]

    report.append("")
    report.append("-" * 78)
    report.append("Frozen parameter files")
    report.append("-" * 78)
    for name in ["ood_params.npz", "conformal_params.json", "policy_params.json"]:
        p = CACHE / name
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
        report.append(f"  {p}   mtime={mtime}")
    report.append(f"  q_mondrian[0.02] = {conformal_params['q_mondrian']['0.02']}")
    report.append(f"  selected_cost_ratio = {selected_ratio}")
    for cond in ["2", "3", "4", "5"]:
        spec = policy_params["conditions"][cond]
        w = spec["weights"][str(selected_ratio)]
        report.append(f"  condition {cond}: features={spec['feature_names']}  "
                      f"w={w['w']}  b={w['b']:.6f}")
    report.append(f"  B2 threshold @ ratio {selected_ratio} = "
                  f"{policy_params['B2']['thresholds'][str(selected_ratio)]:.6f}")
    report.append("  NEH/OCTDL never enter any fit/whitening/quantile-estimation/"
                  "threshold-search step -- pure application-time inputs.")

    print("Running regression tests (Layer 1: 08b's own test-split gate; "
          "Layer 2: RETOUCH/OCT5k diff against 08b_slice_scores.csv, "
          "reported not gated)...")
    reg_lines, gate_ok, ref_all = run_regression_tests(
        ood_params, conformal_params, policy_params, selected_ratio)
    report.append("")
    report.append("-" * 78)
    report.append("Regression tests")
    report.append("-" * 78)
    report += reg_lines
    print(f"  Layer 1 (gating) PASS={gate_ok}")
    if not gate_ok:
        print("\n[STOP] Layer 1 regression test failed -- see report for measured "
              "differences. Not proceeding to NEH/OCTDL scoring.")
        (TABDIR / "13b_new_scoring.txt").write_text("\n".join(report) + "\n")
        return

    print("\nBuilding NEH/OCTDL index frames...")
    neh_index, octdl_index = build_index_frames()

    print(f"Scoring NEH ({len(neh_index)} rows)...")
    neh_df = build_slice_table("neh", "NEH", neh_index, ood_params,
                               conformal_params, policy_params, selected_ratio)

    print(f"Scoring OCTDL ({len(octdl_index)} rows)...")
    octdl_df = build_slice_table("octdl", "OCTDL", octdl_index, ood_params,
                                 conformal_params, policy_params, selected_ratio)

    new_df = pd.concat([neh_df, octdl_df], ignore_index=True)
    out_csv = TABDIR / "13b_new_slice_scores.csv"
    new_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.relative_to(PROJECT)} ({len(new_df)} rows)")

    report.append("")
    report.append("-" * 78)
    report.append("Grouped statistics: mean/std of ood_cosine, ood_maha_combined, "
                  "softmax_confidence, mutual_info, cp_setsize")
    report.append("-" * 78)
    report += format_grouped_stats(new_df, ["dataset", "label_group"],
                                   "NEH / OCTDL, by (dataset, label_group):")
    report.append("")
    report += format_grouped_stats(ref_all, ["dataset", "vendor"],
                                   "Reference -- existing 08b sources, by (dataset, vendor):")

    report.append("")
    report.append("-" * 78)
    report.append("Exploratory comparisons of external-source signals")
    report.append("-" * 78)
    report += prediction_verdicts(new_df, ref_all)

    report_path = TABDIR / "13b_new_scoring.txt"
    report_path.write_text("\n".join(report) + "\n")
    print(f"Wrote {report_path.relative_to(PROJECT)}")
    print("\n13b complete.")


if __name__ == "__main__":
    main()
