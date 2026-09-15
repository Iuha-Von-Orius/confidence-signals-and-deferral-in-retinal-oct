"""Evaluate operating costs, matched coverage and conformal coverage by source.

Also retain cost-ratio sensitivity and restricted-class prediction diagnostics.
"""

from paths import PROJECT, read_csv

import importlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = PROJECT / "cache"
TABDIR = PROJECT / "results" / "tables"
FIGDIR = PROJECT / "results" / "figures" / "14"

sys.path.insert(0, str(Path(__file__).resolve().parent))
_pol07 = importlib.import_module("07_policy")
_conf06 = importlib.import_module("06_conformal")
_ood08b = importlib.import_module("08b_score_ood")
_ood08c = importlib.import_module("08c_evaluate")
_aurc09 = importlib.import_module("09_aurc")
_ev13c = importlib.import_module("13c_evaluate_all_sources")

C_DEFER = _pol07.C_DEFER
COST_RATIOS = _pol07.COST_RATIOS
natural_stats = _pol07.natural_stats

apply_policy = _ood08b.apply_policy
load_frozen_params = _ood08b.load_frozen_params

risk_coverage_curve = _ood08c.risk_coverage_curve
COVERAGE_GRID = _ood08c.COVERAGE_GRID

build_sets_mondrian = _conf06.build_sets_mondrian
coverage_stats = _conf06.coverage_stats

per_sample_curve = _aurc09.per_sample_curve

load_merged_df = _ev13c.load_merged_df
SOURCES_ORDER = _ev13c.SOURCES_ORDER
TRUTH_SOURCES = _ev13c.TRUTH_SOURCES
OCTDL_TIERS = _ev13c.OCTDL_TIERS
CONDITIONS_ALL = _ev13c.CONDITIONS_ALL
CONDITIONS_AURC = _ev13c.CONDITIONS_AURC
outcome_source_arg = _ev13c.outcome_source_arg
correct_for = _ev13c.correct_for
risk_definition_for = _ev13c.risk_definition_for
SOURCE_COLOURS_13C = _ev13c.SOURCE_COLOURS_13C
IDENTIFIER_COL = _ev13c.IDENTIFIER_COL
CLASSES = _ev13c.CLASSES
N_CLASSES = _ev13c.N_CLASSES
NORMAL_IDX = _ev13c.NORMAL_IDX
risk_score_for = _ev13c.risk_score_for

CONDITIONS_T13T14 = [1, 2, 3, 4, 5, "B2"]
SELECTED_RATIO = 20
TARGET_RISKS = [0.10, 0.05, 0.02, 0.01]

PROBS_CACHE_PREFIX = {"Kermany-test": "test", "Rasti": "oct5k", "NEH": "neh"}

RASTI_PRESENT = ["DME", "DRUSEN", "NORMAL"]
NEH_PRESENT = ["CNV", "DRUSEN", "NORMAL"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3,
})
CONDITION_COLOURS_14 = {1: "#7f8c8d", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd",
                        5: "#d62728", "B2": "#ff7f0e"}


def condition_label(c):
    return dict(CONDITIONS_ALL)[c]


# Part 3 helper: Case A/B check
def check_cost_ratio_case(policy_params):
    lines = []
    cond2 = policy_params["conditions"]["2"]
    weight_keys = sorted(cond2["weights"].keys(), key=int)
    b2_keys = sorted(policy_params["B2"]["thresholds"].keys(), key=int)
    lines.append(f'policy_params.json["conditions"]["2"]["weights"] keys: {weight_keys}')
    lines.append(f'policy_params.json["B2"]["thresholds"] keys: {b2_keys}')
    expected = sorted([str(r) for r in COST_RATIOS], key=int)
    case_a = (weight_keys == expected) and (b2_keys == expected)
    lines.append(f"Expected COST_RATIOS as strings: {expected}")
    if case_a:
        lines.append("CASE A: every cost ratio in COST_RATIOS has its own fitted "
                     "{w,b} (and B2 threshold). T14 uses each ratio's own parameters "
                     "to both decide and cost -- the scientifically correct procedure.")
    else:
        lines.append("CASE B: not every cost ratio has its own fitted parameters. "
                     "T14 can only vary the evaluation weighting on one fixed decision "
                     "-- this tests the metric's sensitivity to cost ratio, NOT the "
                     "policy's sensitivity to cost ratio. These are different claims.")
    return case_a, lines


# T13
def build_t13(merged, policy_params):
    b2_threshold_20 = policy_params["B2"]["thresholds"][str(SELECTED_RATIO)]
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        correct = correct_for(sub, source)
        risk_def = risk_definition_for(source)
        for condition in CONDITIONS_T13T14:
            if condition == 1:
                stats = natural_stats(np.zeros(len(sub)), correct, np.inf, SELECTED_RATIO, C_DEFER)
            elif condition == "B2":
                stats = natural_stats(-sub["softmax_confidence"].to_numpy(), correct,
                                      -b2_threshold_20, SELECTED_RATIO, C_DEFER)
            else:
                stats = natural_stats(sub[f"s_condition{condition}"].to_numpy(), correct,
                                      0.0, SELECTED_RATIO, C_DEFER)
            rows.append({"source": source, "condition": condition, "ratio": SELECTED_RATIO,
                        "risk_definition": risk_def, **stats})
    df = pd.DataFrame(rows)

    baseline = df.loc[df["condition"] == 1].set_index("source")["expected_cost"]

    def rel_cost(row):
        b = baseline[row["source"]]
        if b == 0:
            return "undefined (zero baseline cost)"
        return row["expected_cost"] / b

    df["relative_cost"] = df.apply(rel_cost, axis=1)
    df.to_csv(TABDIR / "14_T13_expected_cost.csv", index=False)
    return df


# T13b
def build_t13b_robustness(t13):
    """Summarise relative cost, excluding sources with zero baseline cost."""
    rows = []
    for condition in CONDITIONS_T13T14:
        sub = t13[t13["condition"] == condition]
        undefined_mask = sub["relative_cost"].apply(lambda v: isinstance(v, str))
        excluded = sorted(sub.loc[undefined_mask, "source"].tolist())
        defined = sub.loc[~undefined_mask].copy()
        defined["relative_cost"] = defined["relative_cost"].astype(float)
        if len(defined) == 0:
            rows.append({"condition": condition, "worst_case_relative_cost": None,
                        "worst_case_source": None, "n_sources_above_one": None,
                        "sources_above_one": None, "mean_relative_cost": None,
                        "median_relative_cost": None,
                        "excluded_sources": "; ".join(excluded),
                        "exclusion_reason": "relative_cost undefined (zero baseline cost) "
                                            "for all sources at this condition"})
            continue
        worst_idx = defined["relative_cost"].idxmax()
        above_one = defined.loc[defined["relative_cost"] > 1.0]
        rows.append({
            "condition": condition,
            "worst_case_relative_cost": float(defined.loc[worst_idx, "relative_cost"]),
            "worst_case_source": defined.loc[worst_idx, "source"],
            "n_sources_above_one": int(len(above_one)),
            "sources_above_one": "; ".join(sorted(above_one["source"].tolist())),
            "mean_relative_cost": float(defined["relative_cost"].mean()),
            "median_relative_cost": float(defined["relative_cost"].median()),
            "excluded_sources": "; ".join(excluded) if excluded else "",
            "exclusion_reason": ("relative_cost undefined (zero baseline cost)"
                                if excluded else ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "14_T13b_robustness.csv", index=False)
    return df


# T14
def build_t14(merged, policy_params):
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        correct = correct_for(sub, source)
        risk_def = risk_definition_for(source)
        feature_dict = {"ood_cosine": sub["ood_cosine"].to_numpy(),
                        "ood_maha_combined": sub["ood_maha_combined"].to_numpy(),
                        "mutual_info": sub["mutual_info"].to_numpy(),
                        "cp_setsize": sub["cp_setsize"].to_numpy()}
        for ratio in COST_RATIOS:
            for condition in CONDITIONS_T13T14:
                if condition == 1:
                    stats = natural_stats(np.zeros(len(sub)), correct, np.inf, ratio, C_DEFER)
                elif condition == "B2":
                    t = policy_params["B2"]["thresholds"][str(ratio)]
                    stats = natural_stats(-sub["softmax_confidence"].to_numpy(), correct,
                                          -t, ratio, C_DEFER)
                else:
                    s = apply_policy(condition, feature_dict, policy_params, ratio)
                    stats = natural_stats(s, correct, 0.0, ratio, C_DEFER)
                rows.append({"source": source, "condition": condition, "ratio": ratio,
                            "risk_definition": risk_def, **stats})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby(["source", "ratio"])["expected_cost"].rank(method="min").astype(int)
    df.to_csv(TABDIR / "14_T14_cost_ratio_sweep.csv", index=False)

    change_rows = []
    for source in SOURCES_ORDER:
        for condition in CONDITIONS_T13T14:
            sub = df[(df["source"] == source) & (df["condition"] == condition)]
            ranks = sub.sort_values("ratio")["rank"].tolist()
            changed = len(set(ranks)) > 1
            change_rows.append({"source": source, "condition": condition,
                                "ranks_by_ratio": ranks, "rank_changes": changed})
    change_df = pd.DataFrame(change_rows)
    change_df.to_csv(TABDIR / "14_T14_rank_changes.csv", index=False)
    return df, change_df


# T15
def max_coverage_at_risk(risk_score, correct, target):
    coverage, risk = per_sample_curve(risk_score, correct)
    valid = (coverage[1:] > 0) & (risk[1:] <= target)
    if not valid.any():
        return None
    idx = np.where(valid)[0]
    return float(coverage[1:][idx.max()])


def build_t15(merged):
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        for risk_def, correct in [
            ("missed_disease", correct_for(sub, source)),
        ] + ([("four_class_accuracy",
               (sub["predicted_class_name"] == sub["true_label_name"]).to_numpy())]
             if source in TRUTH_SOURCES else []):
            for condition in CONDITIONS_AURC:
                risk_score = risk_score_for(sub, condition)
                for target in TARGET_RISKS:
                    mc = max_coverage_at_risk(risk_score, correct, target)
                    rows.append({
                        "source": source, "condition": condition, "risk_definition": risk_def,
                        "target_risk": target,
                        "max_coverage": mc if mc is not None else "not reachable",
                        "required_deferral_rate": (1.0 - mc) if mc is not None else "not reachable",
                    })
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "14_T15_coverage_at_target_risk.csv", index=False)
    return df


# T16
MATCHED_COVERAGE_POINTS = [0.70, 0.80, 0.90]
MATCHED_COVERAGE_CONDITIONS = [2, 3, 4, 5, "B2"]


def build_t16(merged):
    """Read matched-coverage risk from the original 101-point interpolated curves."""
    rows = []
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        correct = correct_for(sub, source)
        baseline_error = 1.0 - correct.mean()
        for condition in MATCHED_COVERAGE_CONDITIONS:
            risk_score = risk_score_for(sub, condition)
            cov_grid, risk_grid = risk_coverage_curve(risk_score, correct)
            for coverage in MATCHED_COVERAGE_POINTS:
                idx = int(round(coverage * (len(COVERAGE_GRID) - 1)))
                accepted_error_rate = float(risk_grid[idx])
                if baseline_error == 0:
                    relative = "undefined (zero baseline)"
                    worse = None
                else:
                    relative = accepted_error_rate / baseline_error
                    worse = bool(relative > 1.0)
                rows.append({
                    "source": source, "condition": condition, "coverage": coverage,
                    "accepted_error_rate": accepted_error_rate,
                    "baseline_error": baseline_error,
                    "relative_to_baseline": relative,
                    "worse_than_baseline": worse,
                })
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "14_T16_matched_coverage.csv", index=False)
    return df


# Part 5: conformal coverage
def load_probs_for_source(merged, source):
    """Load probabilities in source row order, joining OCTDL paths to cache indices."""
    sub = merged[merged["source"] == source].reset_index(drop=True)
    if source == "OCTDL-in-label":
        full = np.load(CACHE / "octdl_probs.npy")
        octdl_idx = read_csv(CACHE / "octdl_index.csv")[["path", "cache_index"]]
        joined = sub[["path"]].merge(octdl_idx, on="path", how="left")
        assert joined["cache_index"].notna().all(), (
            "OCTDL-in-label: some rows failed to match cache/octdl_index.csv by path")
        idx = joined["cache_index"].to_numpy().astype(int)
        probs = full[idx]
    else:
        prefix = PROBS_CACHE_PREFIX[source]
        probs = np.load(CACHE / f"{prefix}_probs.npy")
    labels = sub["true_label"].to_numpy().astype(int)
    return probs, labels, sub


def build_conformal_coverage(merged, conformal_params):
    q_mondrian_002 = conformal_params["q_mondrian"]["0.02"]
    q_dict = {k: q_mondrian_002[CLASSES[k]] for k in range(N_CLASSES)}

    verify_lines = []
    coverage_rows = []
    per_class_rows = []
    for source in TRUTH_SOURCES:
        probs, labels, sub = load_probs_for_source(merged, source)
        fresh_pred = probs.argmax(axis=1)
        recorded_pred = sub["predicted_class"].to_numpy().astype(int)
        mismatch = int((fresh_pred != recorded_pred).sum())
        verify_lines.append(f"  {source}: {mismatch} of {len(sub)} rows mismatch "
                            f"between freshly-loaded argmax and merged's recorded predicted_class")
        if mismatch:
            verify_lines.append(f"    [STOP] correspondence for {source} is not verified -- "
                                "coverage numbers for this source are not computed.")
            continue

        sets = build_sets_mondrian(probs, q_dict)
        stats = coverage_stats(sets, labels)
        coverage_rows.append({"source": source, "n": len(labels),
                              "measured_coverage": stats["coverage"],
                              "nominal_coverage": 0.98,
                              "deviation": stats["coverage"] - 0.98,
                              "mean_set_size": stats["mean_size"]})
        for cls, cov in stats["per_class"].items():
            per_class_rows.append({"source": source, "class": cls, "coverage": cov})

    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(TABDIR / "14_conformal_coverage.csv", index=False)
    pc_df = pd.DataFrame(per_class_rows)
    pc_df.to_csv(TABDIR / "14_conformal_per_class_coverage.csv", index=False)
    return cov_df, pc_df, verify_lines


# Part 6: restricted argmax
def restricted_argmax_diagnostic(merged):
    rows = []
    cm_lines = []
    for source, present in [("Rasti", RASTI_PRESENT), ("NEH", NEH_PRESENT)]:
        probs, labels, sub = load_probs_for_source(merged, source)
        allowed_idx = [CLASSES.index(c) for c in present]
        unrestricted_pred = probs.argmax(axis=1)
        restricted_local = probs[:, allowed_idx].argmax(axis=1)
        restricted_pred = np.array(allowed_idx)[restricted_local]

        known_diseased = sub["known_diseased"].to_numpy()
        for label, pred in [("unrestricted", unrestricted_pred), ("restricted", restricted_pred)]:
            pred_names = np.array(CLASSES)[pred]
            true_names = sub["true_label_name"].to_numpy()
            overall_acc = float((pred_names == true_names).mean())
            miss_rate = float((pred_names[known_diseased] == "NORMAL").mean()) if known_diseased.any() else None
            for c in CLASSES:
                m = true_names == c
                acc = float((pred_names[m] == c).mean()) if m.sum() else None
                rows.append({"source": source, "variant": label, "class": c,
                            "n": int(m.sum()), "accuracy": acc})
            rows.append({"source": source, "variant": label, "class": "OVERALL",
                        "n": len(sub), "accuracy": overall_acc, "miss_rate": miss_rate})
            cm = pd.crosstab(pd.Series(true_names, name="true"),
                             pd.Series(pred_names, name="predicted"))
            cm = cm.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
            cm_lines.append(f"{source} {label} confusion matrix:\n{cm.to_string()}")
    df = pd.DataFrame(rows)
    df.to_csv(TABDIR / "14_restricted_argmax.csv", index=False)
    return df, cm_lines


# Figures
def fig_f11():
    t08 = read_csv(TABDIR / "13c_T08_aurc.csv")
    for col, fname in [("aurc", "14_F11_aurc.png"), ("e_aurc", "14_F11_eaurc.png")]:
        conditions = [c for c in [2, 3, 4, 5, "B2"]]
        mat = np.full((len(SOURCES_ORDER), len(conditions)), np.nan)
        for i, s in enumerate(SOURCES_ORDER):
            for j, c in enumerate(conditions):
                row = t08[(t08["source"] == s) & (t08["condition"].astype(str) == str(c))]
                if len(row):
                    v = row[col].values[0]
                    if isinstance(v, str):
                        continue
                    mat[i, j] = v
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(conditions)), [str(c) for c in conditions])
        ax.set_yticks(range(len(SOURCES_ORDER)), SOURCES_ORDER)
        for i in range(len(SOURCES_ORDER)):
            for j in range(len(conditions)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=10)
        ax.set_xlabel("Condition")
        ax.set_title(f"{col} (blank = undefined, 0 errors)")
        fig.colorbar(im, ax=ax, label=col)
        fig.tight_layout()
        fig.savefig(FIGDIR / fname)
        plt.close(fig)


def fig_f12(t13):
    """Plot policy cost relative to each source's no-deferral cost."""
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(SOURCES_ORDER))
    width = 0.13
    undefined_sources = []
    for i, condition in enumerate(CONDITIONS_T13T14):
        vals = []
        for s in SOURCES_ORDER:
            v = t13[(t13["source"] == s) & (t13["condition"] == condition)]["relative_cost"].values[0]
            if isinstance(v, str):
                vals.append(np.nan)
                if s not in undefined_sources:
                    undefined_sources.append(s)
            else:
                vals.append(float(v))
        ax.bar(x + (i - 2.5) * width, vals, width, label=f"{condition} {condition_label(condition)}",
              color=CONDITION_COLOURS_14[condition], alpha=0.88)
    ax.axhline(1.0, color="black", ls="--", lw=1.5,
              label="relative_cost = 1 (equal to no deferral)")
    ax.set_xticks(x, SOURCES_ORDER, rotation=30, ha="right")
    ax.set_ylabel(f"Relative cost (expected_cost / condition-1 expected_cost, "
                  f"ratio={SELECTED_RATIO}:1)")
    title = "T13: expected cost relative to no deferral, at the deployed operating point"
    if undefined_sources:
        title += f"\n(bars absent for {', '.join(undefined_sources)}: undefined, zero baseline cost)"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "14_F12_expected_cost.png")
    plt.close(fig)


def fig_f13(t14):
    n = len(SOURCES_ORDER)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    for ax, source in zip(axes.flat, SOURCES_ORDER):
        for condition in CONDITIONS_T13T14:
            sub = t14[(t14["source"] == source) & (t14["condition"] == condition)].sort_values("ratio")
            ax.plot(sub["ratio"], sub["expected_cost"], "o-", lw=2, ms=5,
                   color=CONDITION_COLOURS_14[condition], label=f"{condition}")
        ax.set_xscale("log")
        ax.set_xlabel("Cost ratio (log)")
        ax.set_ylabel("Expected cost")
        ax.set_title(source, fontsize=13)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=9, ncol=2)
    fig.suptitle("T14: expected cost vs cost ratio", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / "14_F13_cost_ratio_sweep.png")
    plt.close(fig)


def fig_f14(t15):
    sub = t15[(t15["risk_definition"] == "missed_disease") & (t15["target_risk"] == 0.05)]
    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(SOURCES_ORDER))
    width = 0.15
    for i, condition in enumerate(CONDITIONS_AURC):
        vals, not_reach = [], []
        for s in SOURCES_ORDER:
            row = sub[(sub["source"] == s) & (sub["condition"] == condition)]
            if len(row) == 0:
                vals.append(0.0); not_reach.append(False); continue
            v = row["required_deferral_rate"].values[0]
            if v == "not reachable":
                vals.append(1.0); not_reach.append(True)
            else:
                vals.append(float(v)); not_reach.append(False)
        bars = ax.bar(x + (i - 2) * width, vals, width,
                      label=f"{condition} {condition_label(condition)}",
                      color=CONDITION_COLOURS_14[condition], alpha=0.85)
        for bar, nr in zip(bars, not_reach):
            if nr:
                bar.set_hatch("xxx")
                bar.set_edgecolor("black")
                ax.text(bar.get_x() + bar.get_width() / 2, 1.01, "N/R", ha="center",
                       va="bottom", fontsize=9, rotation=90)
    ax.set_xticks(x, SOURCES_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Required deferral rate (1 - max coverage)")
    ax.set_title("T15: deferral rate required for missed-disease risk <= 5%\n"
                 "(hatched bars = not reachable at any positive coverage)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "14_F14_coverage_at_risk.png")
    plt.close(fig)


def fig_f15(cov_df, pc_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    x = np.arange(len(cov_df))
    colours = [SOURCE_COLOURS_13C[s] for s in cov_df["source"]]
    ax.bar(x, cov_df["measured_coverage"], color=colours, alpha=0.88)
    ax.axhline(0.98, color="black", ls="--", lw=1.5, label="nominal 0.98")
    ax.set_xticks(x, cov_df["source"], rotation=20, ha="right")
    ax.set_ylabel("Measured coverage")
    ax.set_title("Marginal conformal coverage")
    ax.legend(fontsize=10)

    ax2 = axes[1]
    sources = pc_df["source"].unique()
    xw = np.arange(len(CLASSES))
    width = 0.2
    for i, s in enumerate(sources):
        sub = pc_df[pc_df["source"] == s].set_index("class").reindex(CLASSES)
        ax2.bar(xw + (i - 1.5) * width, sub["coverage"], width, label=s,
               color=SOURCE_COLOURS_13C[s], alpha=0.88)
    ax2.axhline(0.98, color="black", ls="--", lw=1.5)
    ax2.set_xticks(xw, CLASSES)
    ax2.set_ylabel("Per-class coverage")
    ax2.set_title("Class-conditional conformal coverage")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "14_F15_conformal_coverage.png")
    plt.close(fig)


def fig_f16(restricted_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, source in zip(axes, ["Rasti", "NEH"]):
        sub = restricted_df[(restricted_df["source"] == source) & (restricted_df["class"] != "OVERALL")]
        x = np.arange(len(CLASSES))
        width = 0.35
        for i, variant in enumerate(["unrestricted", "restricted"]):
            vsub = sub[sub["variant"] == variant].set_index("class").reindex(CLASSES)
            ax.bar(x + (i - 0.5) * width, vsub["accuracy"], width, label=variant,
                  color=SOURCE_COLOURS_13C[source], alpha=0.55 if variant == "unrestricted" else 0.95)
        ax.set_xticks(x, CLASSES)
        ax.set_ylabel("Per-class accuracy")
        ax.set_title(source)
        ax.legend(fontsize=10)
    fig.suptitle("Argmax restricted to classes actually present in the dataset", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGDIR / "14_F16_restricted_argmax.png")
    plt.close(fig)


def fig_f20(t16):
    """Plot matched-coverage error rates, coloured by risk relative to no deferral."""
    import matplotlib.colors as mcolors

    columns = MATCHED_COVERAGE_CONDITIONS + ["Baseline"]
    column_labels = [f"{c} {condition_label(c)}" if c != "Baseline"
                     else "Baseline (100% coverage)" for c in columns]

    all_relative = []
    for coverage in MATCHED_COVERAGE_POINTS:
        sub = t16[t16["coverage"] == coverage]
        for v in sub["relative_to_baseline"]:
            if not isinstance(v, str):
                all_relative.append(float(v))
    all_relative.append(1.0)
    vmin, vmax = min(all_relative), max(all_relative)
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=min(vmin, 0.99), vmax=max(vmax, 1.01))
    cmap = plt.get_cmap("RdBu_r")

    for coverage in MATCHED_COVERAGE_POINTS:
        sub = t16[t16["coverage"] == coverage]
        err_mat = np.full((len(SOURCES_ORDER), len(columns)), np.nan)
        rel_mat = np.full((len(SOURCES_ORDER), len(columns)), np.nan)
        undefined_mask = np.zeros((len(SOURCES_ORDER), len(columns)), dtype=bool)

        for i, source in enumerate(SOURCES_ORDER):
            baseline_row = sub[(sub["source"] == source)]
            baseline_error = float(baseline_row["baseline_error"].iloc[0]) if len(baseline_row) else np.nan
            for j, col in enumerate(columns):
                if col == "Baseline":
                    if baseline_error == 0:
                        undefined_mask[i, j] = True
                    else:
                        err_mat[i, j] = baseline_error
                        rel_mat[i, j] = 1.0
                    continue
                row = sub[(sub["source"] == source) & (sub["condition"] == col)]
                if len(row) == 0:
                    continue
                rel = row["relative_to_baseline"].values[0]
                err_mat[i, j] = row["accepted_error_rate"].values[0]
                if isinstance(rel, str):
                    undefined_mask[i, j] = True
                else:
                    rel_mat[i, j] = float(rel)

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(rel_mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(columns)), column_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(SOURCES_ORDER)), SOURCES_ORDER)
        for i in range(len(SOURCES_ORDER)):
            for j in range(len(columns)):
                if undefined_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               hatch="xxx", edgecolor="black", lw=1.0))
                    continue
                if not np.isnan(err_mat[i, j]):
                    ax.text(j, i, f"{err_mat[i, j]:.4f}", ha="center", va="center", fontsize=10)
                if not np.isnan(rel_mat[i, j]) and rel_mat[i, j] > 1.0:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               hatch="///", edgecolor="black", lw=1.2))
        ax.set_title(f"Matched coverage {coverage:.0%}: accepted error rate "
                     f"(cell text) and relative-to-baseline (colour, 1.0 = no-deferral "
                     f"error rate)\nhatched cells: >1.0 (worse than no deferral) or "
                     f"undefined (zero baseline error)", fontsize=12)
        fig.colorbar(im, ax=ax, label="Accepted error rate / no-deferral error rate")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"14_F20_matched_coverage_{int(coverage*100)}.png")
        plt.close(fig)


# Main
def main():
    t0 = time.time()
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 78)
    report.append("14_cost_and_coverage.py -- operating-point tables, conformal coverage, "
                  "restricted-argmax diagnostic")
    report.append("=" * 78)
    report.append("")
    report.append(f"C_DEFER = {C_DEFER}   COST_RATIOS = {COST_RATIOS}   "
                  f"SELECTED_RATIO (this script) = {SELECTED_RATIO}")

    print("Loading frozen parameters and merged nine-source dataframe...")
    ood_params, conformal_params, policy_params = load_frozen_params()
    merged = load_merged_df()

    case_a, case_lines = check_cost_ratio_case(policy_params)
    report.append("")
    report.append("-" * 78)
    report.append("Cost-ratio case check (Case A vs Case B)")
    report.append("-" * 78)
    report += case_lines

    print("Building T13 (expected cost at deployed ratio)...")
    t13 = build_t13(merged, policy_params)
    t13b = build_t13b_robustness(t13)

    print("Building T14 (cost-ratio sweep)...")
    t14, t14_changes = build_t14(merged, policy_params)

    print("Building T15 (coverage at target risk)...")
    t15 = build_t15(merged)

    print("Building T16 (matched coverage, all nine sources)...")
    t16 = build_t16(merged)

    print("Measuring conformal coverage...")
    cov_df, pc_df, verify_lines = build_conformal_coverage(merged, conformal_params)
    report.append("")
    report.append("-" * 78)
    report.append("Part 5: probs/predicted_class correspondence verification "
                  "(must show 0 mismatches before coverage numbers are trusted)")
    report.append("-" * 78)
    report.append("Note: cache_index is absent from 13b_new_slice_scores.csv (13b's "
                  "build_index_frames() selects a fixed column list that does not "
                  "include it, discovered this session) -- recovered for OCTDL-in-label "
                  "by joining `merged` to cache/octdl_index.csv on `path` (all 2,064 "
                  "OCTDL paths matched exactly). Kermany-test/Rasti/NEH need no such "
                  "join (positional correspondence, verified below).")
    report += verify_lines

    print("Restricted-argmax diagnostic (Rasti, NEH)...")
    restricted_df, cm_lines = restricted_argmax_diagnostic(merged)

    print("Building figures F11-F16, F20...")
    fig_f11()
    fig_f12(t13)
    fig_f13(t14)
    fig_f14(t15)
    fig_f15(cov_df, pc_df)
    fig_f16(restricted_df)
    fig_f20(t16)

    elapsed = time.time() - t0

    report.append("")
    report.append("-" * 78)
    report.append("Tables")
    report.append("-" * 78)
    report.append("14_T13_expected_cost.csv -- expected cost and relative_cost (= expected_cost "
                  "/ that source's own condition-1 cost), nine sources x six conditions, at the "
                  f"deployed ratio {SELECTED_RATIO}:1. relative_cost is the string "
                  "\"undefined (zero baseline cost)\" wherever that source's condition-1 cost "
                  "is exactly 0 (OCTDL-ambiguous, every condition) -- never inf, NaN or blank, "
                  "same treatment as 13c's T08 zero-error degeneracy.")
    report.append("14_T13b_robustness.csv -- one row per condition: worst-case relative_cost "
                  "and which source attains it, count and list of sources with relative_cost "
                  "> 1.0 (arithmetically: that condition's expected cost exceeds never "
                  "deferring, at that source and the deployed ratio -- no evaluation of "
                  "whether that is good or bad), mean/median relative_cost, and which sources "
                  "were excluded from these statistics (undefined relative_cost) and why.")
    report.append("14_T14_cost_ratio_sweep.csv -- expected cost and rank, source x condition "
                  f"x ratio in {COST_RATIOS}.")
    report.append("14_T14_rank_changes.csv -- per (source, condition), whether rank changed "
                  "across the ratio sweep.")
    report.append("14_T15_coverage_at_target_risk.csv -- max reachable coverage (or "
                  "'not reachable') at four target risks, two risk definitions.")
    report.append("14_T16_matched_coverage.csv -- accepted error rate at coverage "
                  f"{MATCHED_COVERAGE_POINTS}, nine sources x conditions "
                  f"{MATCHED_COVERAGE_CONDITIONS}, via the imported risk_coverage_curve "
                  "(); the coverage-grid indexing step itself is six lines "
                  "inline in 08c's own main() (, what writes "
                  "08c_matched_coverage_summary.csv), not a separate function, reproduced "
                  "here with that citation. relative_to_baseline is the string "
                  "\"undefined (zero baseline)\" wherever that source's own no-deferral "
                  "error rate is exactly 0 (OCTDL-ambiguous) -- never inf/NaN/blank, and "
                  "worse_than_baseline is left blank (not False) in the same rows, since "
                  "\"worse than baseline\" cannot be determined when the ratio itself is "
                  "undefined.")
    report.append("14_conformal_coverage.csv -- measured marginal coverage vs nominal 0.98, "
                  "mean set size, four truth-bearing sources.")
    report.append("14_conformal_per_class_coverage.csv -- same, by true class.")
    report.append("14_restricted_argmax.csv -- Rasti/NEH confusion, per-class accuracy, "
                  "overall accuracy, miss rate, unrestricted vs restricted argmax.")

    report.append("")
    report.append("-" * 78)
    report.append("Restricted-argmax confusion matrices")
    report.append("-" * 78)
    report += cm_lines

    report.append("")
    report.append("-" * 78)
    report.append("Figures")
    report.append("-" * 78)
    report.append("14_F11_aurc.png / 14_F11_eaurc.png -- source x condition, from "
                  "13c_T08_aurc.csv (read, not recomputed); undefined cells blank.")
    report.append("14_F12_expected_cost.png -- T13, grouped bar, source x condition.")
    report.append("14_F13_cost_ratio_sweep.png -- T14, one panel per source, expected cost "
                  "vs cost ratio (log), six condition lines.")
    report.append("14_F14_coverage_at_risk.png -- T15 at missed-disease risk target 5%, "
                  "grouped bar, hatched bars mark 'not reachable'.")
    report.append("14_F15_conformal_coverage.png -- measured coverage vs 0.98 reference "
                  "(left), per-class coverage (right).")
    report.append("14_F16_restricted_argmax.png -- Rasti/NEH per-class accuracy, "
                  "unrestricted vs restricted argmax.")
    report.append("14_F20_matched_coverage_70.png / _80.png / _90.png -- one heatmap per "
                  "matched-coverage point, nine sources x [2,3,4,5,B2,Baseline]; cell text "
                  "is the accepted error rate, cell colour is relative_to_baseline on a "
                  "shared diverging scale centred at 1.0 across all three figures; cells "
                  ">1.0 or undefined (zero baseline) are additionally hatched.")

    report.append("")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    (TABDIR / "14_evaluation.txt").write_text("\n".join(str(l) for l in report) + "\n")
    print(f"\nWrote results/tables/14_evaluation.txt")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
