"""Compute AUGRC, the primary full-range ranking metric, for all nine sources.

Generalised risk is accepted errors divided by the total source size.
Tie-averaged curves are integrated from coverage 0 to 1. AURC is retained
for comparison. Mean ranks exclude sources with no baseline errors.
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
FIGDIR = PROJECT / "results" / "figures" / "15"

sys.path.insert(0, str(Path(__file__).resolve().parent))
_aurc09 = importlib.import_module("09_aurc")
_ev13c = importlib.import_module("13c_evaluate_all_sources")

per_sample_curve = _aurc09.per_sample_curve

SOURCES_ORDER = _ev13c.SOURCES_ORDER
CONDITIONS_AURC = _ev13c.CONDITIONS_AURC
load_merged_df = _ev13c.load_merged_df
correct_for = _ev13c.correct_for
risk_definition_for = _ev13c.risk_definition_for
risk_score_for = _ev13c.risk_score_for
SOURCE_COLOURS_13C = _ev13c.SOURCE_COLOURS_13C
CLASSES = _ev13c.CLASSES

CONDITION_COLOURS = {1: "#7f8c8d", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd",
                     5: "#d62728", "B2": "#ff7f0e"}
CONDITION_LABELS = {1: "baseline", 2: "+OOD", 3: "+uncertainty",
                    4: "+conformal", 5: "+all signals", "B2": "softmax threshold"}

SOURCE_MARKERS = {
    "Kermany-test": "o", "RETOUCH-Spectralis": "s", "RETOUCH-Cirrus": "^",
    "RETOUCH-Topcon": "v", "Rasti": "D", "NEH": "P",
    "OCTDL-in-label": "X", "OCTDL-ambiguous": "*", "OCTDL-novel-class": "h",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3,
})


def augrc_trapezoid(coverage, gen_risk):
    """Integrate generalised risk over all coverage nodes, including zero."""
    heights = (gen_risk[:-1] + gen_risk[1:]) * 0.5
    widths = coverage[1:] - coverage[:-1]
    return float(np.sum(heights * widths))


def compute_augrc(merged):
    rows = []
    curves = {}
    for source in SOURCES_ORDER:
        sub = merged[merged["source"] == source]
        correct = correct_for(sub, source)
        risk_def = risk_definition_for(source)
        base_error = 1.0 - correct.mean()
        for condition in CONDITIONS_AURC:
            risk_score = risk_score_for(sub, condition)
            coverage, sel_risk = per_sample_curve(risk_score, correct)
            gen_risk = sel_risk * coverage
            gen_risk[0] = 0.0
            augrc = augrc_trapezoid(coverage, gen_risk)
            curves[(source, condition)] = (coverage, gen_risk)
            if base_error == 0:
                ratio_vs_random = "undefined (0 errors)"
            else:
                ratio_vs_random = augrc / (base_error / 2.0)
            rows.append({"source": source, "condition": condition,
                        "risk_definition": risk_def, "n": len(sub),
                        "base_error": base_error, "augrc": augrc,
                        "augrc_x1000": augrc * 1000.0,
                        "augrc_ratio_vs_random": ratio_vs_random})
    return pd.DataFrame(rows), curves


def _coerce_numeric_or_string(v):
    """Parse numeric AURC values while retaining the undefined marker."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def attach_aurc(df):
    t08 = read_csv(TABDIR / "13c_T08_aurc.csv")
    t08 = t08[["source", "condition", "aurc"]].rename(columns={"aurc": "aurc_x1000"})
    t08["condition"] = t08["condition"].astype(str)
    df = df.copy()
    df["condition"] = df["condition"].astype(str)
    merged = df.merge(t08, on=["source", "condition"], how="left")
    merged["aurc_x1000"] = merged["aurc_x1000"].apply(_coerce_numeric_or_string)
    return merged


def add_ranks(df):
    df = df.copy()
    df["rank_augrc"] = (df.groupby("source")["augrc"]
                        .rank(method="dense", ascending=True).astype(int))

    def rank_aurc_group(g):
        numeric = pd.to_numeric(g["aurc_x1000"], errors="coerce")
        ranked = numeric.rank(method="dense", ascending=True)
        return ranked
    df["rank_aurc"] = df.groupby("source", group_keys=False).apply(rank_aurc_group)
    df["rank_aurc"] = df["rank_aurc"].astype("Int64")
    return df


# Figures
def fig_f21(df):
    conditions = CONDITIONS_AURC
    mat = np.full((len(SOURCES_ORDER), len(conditions)), np.nan)
    for i, s in enumerate(SOURCES_ORDER):
        for j, c in enumerate(conditions):
            row = df[(df["source"] == s) & (df["condition"] == str(c))]
            if len(row):
                mat[i, j] = row["augrc"].values[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(conditions)), [str(c) for c in conditions])
    ax.set_yticks(range(len(SOURCES_ORDER)), SOURCES_ORDER)
    for i in range(len(SOURCES_ORDER)):
        for j in range(len(conditions)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=10)
    ax.set_xlabel("Condition")
    ax.set_title("AUGRC (source x condition), [0, 0.5] scale")
    fig.colorbar(im, ax=ax, label="AUGRC")
    fig.tight_layout()
    fig.savefig(FIGDIR / "15_F21_augrc_heatmap.png")
    plt.close(fig)


def fig_f22(df):
    fig, ax = plt.subplots(figsize=(10, 9))
    plotted_any_excluded = False
    for _, row in df.iterrows():
        if isinstance(row["aurc_x1000"], str) or pd.isna(row["aurc_x1000"]):
            plotted_any_excluded = True
            continue
        mismatch = (pd.notna(row["rank_aurc"]) and row["rank_augrc"] != row["rank_aurc"])
        cond_key = row["condition"] if row["condition"] == "B2" else int(row["condition"])
        ax.scatter(float(row["aurc_x1000"]), row["augrc_x1000"],
                  color=CONDITION_COLOURS[cond_key],
                  marker=SOURCE_MARKERS[row["source"]], s=140,
                  edgecolor="black" if mismatch else "none",
                  linewidth=2.0 if mismatch else 0,
                  zorder=3 if mismatch else 2)
    cond_handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=CONDITION_COLOURS[c],
                               markersize=12, label=f"{c} {CONDITION_LABELS[c]}") for c in CONDITIONS_AURC]
    source_handles = [plt.Line2D([0], [0], marker=SOURCE_MARKERS[s], color="black",
                                 markerfacecolor="lightgray", markersize=11, label=s, linestyle="none")
                      for s in SOURCES_ORDER]
    leg1 = ax.legend(handles=cond_handles, loc="upper left", fontsize=9, title="condition")
    ax.add_artist(leg1)
    ax.legend(handles=source_handles, loc="lower right", fontsize=8.5, title="source", ncol=2)
    ax.set_xlabel("AURC x1000")
    ax.set_ylabel("AUGRC x1000")
    title = "AUGRC vs AURC, per (source, condition); black ring = rank_augrc != rank_aurc"
    if plotted_any_excluded:
        title += "\n(OCTDL-ambiguous excluded: aurc undefined, 0 errors)"
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "15_F22_augrc_vs_aurc.png")
    plt.close(fig)


def fig_f23(curves):
    n = len(SOURCES_ORDER)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    for ax, source in zip(axes.flat, SOURCES_ORDER):
        for condition in CONDITIONS_AURC:
            cov, gen_risk = curves[(source, condition)]
            ax.plot(cov, gen_risk, "-", lw=2, color=CONDITION_COLOURS[condition],
                   label=f"{condition}")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Generalized risk")
        ax.set_title(source, fontsize=13)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=9, ncol=2)
    fig.suptitle("Generalized risk vs coverage", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / "15_F23_generalized_risk_curves.png")
    plt.close(fig)


# Main
def main():
    t0 = time.time()
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 78)
    report.append("15_augrc.py -- AUGRC, nine sources x five conditions")
    report.append("=" * 78)
    report.append("")
    print("Loading merged nine-source dataframe...")
    merged = load_merged_df()

    print("Computing AUGRC...")
    df, curves = compute_augrc(merged)
    df = attach_aurc(df)
    df = add_ranks(df)
    df = df[["source", "condition", "risk_definition", "n", "base_error", "augrc",
            "augrc_x1000", "aurc_x1000", "augrc_ratio_vs_random", "rank_augrc", "rank_aurc"]]
    df.to_csv(TABDIR / "15_augrc.csv", index=False)

    print("Building figures F21-F23...")
    fig_f21(df)
    fig_f22(df)
    fig_f23(curves)

    elapsed = time.time() - t0

    report.append("")
    report.append("-" * 78)
    report.append("AUGRC-rank vs AURC-rank, per source")
    report.append("-" * 78)
    for source in SOURCES_ORDER:
        sub = df[df["source"] == source][["condition", "augrc_x1000", "rank_augrc",
                                          "aurc_x1000", "rank_aurc", "augrc_ratio_vs_random"]]
        report.append(f"  {source}:")
        report.append("    " + sub.to_string(index=False).replace("\n", "\n    "))

    report.append("")
    report.append("-" * 78)
    report.append("augrc_ratio_vs_random = augrc / (base_error / 2.0) -- base_error/2 is "
                  "the AUGRC of a random-ranking baseline on the same source (selective "
                  "risk = base_error at every coverage if risk_score is independent of "
                  "correctness, so generalized_risk = base_error * coverage, integrating "
                  "to base_error/2 over [0,1]). \"undefined (0 errors)\" wherever "
                  "base_error is 0, same convention as 13c's zero-error rows.")
    report.append("Caveat: where n_errors < 10, this ratio is noise-dominated -- not "
                  "removed or blanked on that basis, n_errors listed below so this can "
                  "be judged directly.")
    report.append("-" * 78)
    for source in SOURCES_ORDER:
        row = df[df["source"] == source].iloc[0]
        n_errors = int(round(float(row["base_error"]) * int(row["n"])))
        flag = "  [< 10 errors -- ratio noise-dominated]" if n_errors < 10 else ""
        report.append(f"  {source}: n={int(row['n'])}  n_errors={n_errors}{flag}")

    report.append("")
    report.append("-" * 78)
    report.append("Rank mismatches ((source, condition) where rank_augrc != rank_aurc)")
    report.append("-" * 78)
    mismatches = df[df["rank_aurc"].notna() & (df["rank_augrc"] != df["rank_aurc"])]
    if len(mismatches):
        for _, r in mismatches.iterrows():
            report.append(f"  {r['source']}, condition {r['condition']}: "
                          f"rank_augrc={r['rank_augrc']}  rank_aurc={r['rank_aurc']}")
    else:
        report.append("  none")

    report.append("")
    report.append("-" * 78)
    report.append("Per-condition mean rank -- three sections, not directly comparable to "
                  "each other except (b) internally (see docstring Patch log): the "
                  "original single mean_rank_augrc-over-9-sources vs "
                  "mean_rank_aurc-over-8-sources comparison averaged over two different "
                  "source sets.")
    report.append("-" * 78)

    sources_with_errors = [s for s in SOURCES_ORDER
                           if float(df.loc[df["source"] == s, "base_error"].iloc[0]) > 0]
    excluded_sources = [s for s in SOURCES_ORDER if s not in sources_with_errors]
    conditions_str = [str(x) for x in CONDITIONS_AURC]

    report.append(f"(a) mean rank_augrc over all n_sources={len(SOURCES_ORDER)} sources "
                  f"(AURC not usable on this source set -- undefined for "
                  f"{excluded_sources}):")
    mean_a = {}
    for c in conditions_str:
        m = float(df.loc[df["condition"] == c, "rank_augrc"].mean())
        mean_a[c] = m
        report.append(f"    condition {c}: mean rank_augrc={m:.4f}")

    report.append("")
    report.append(f"(b) like-for-like: mean rank over the n_sources={len(sources_with_errors)} "
                  f"sources with >=1 error (excluded={excluded_sources}), AUGRC and AURC "
                  "both computed on this same source set:")
    df_b = df[df["source"].isin(sources_with_errors)]
    for c in conditions_str:
        ma = float(df_b.loc[df_b["condition"] == c, "rank_augrc"].mean())
        mb = float(df_b.loc[df_b["condition"] == c, "rank_aurc"].mean())
        report.append(f"    condition {c}: mean rank_augrc={ma:.4f}  mean rank_aurc={mb:.4f}")

    report.append("")
    report.append(f"(c) n_sources: (a)={len(SOURCES_ORDER)} sources, excluded=none; "
                  f"(b)={len(sources_with_errors)} sources, excluded={excluded_sources}")

    report.append("")
    report.append("Self-check identity: sum over the 5 conditions of (a)'s mean rank_augrc "
                  "should equal (the grand sum of rank_augrc over all of (b)'s rows, plus "
                  "the excluded sources' own rank_augrc contribution) / len(SOURCES_ORDER) "
                  "-- a decomposition of (a)'s total across the same two source subsets "
                  "(b) already separates, computed via a different code path (per-condition "
                  ".mean() vs a single grand .sum()) so a bug in either would surface here.")
    grand_sum_b_augrc = float(df_b["rank_augrc"].sum())
    excluded_contribution = float(df.loc[df["source"].isin(excluded_sources), "rank_augrc"].sum())
    lhs = sum(mean_a[c] for c in conditions_str)
    rhs = (grand_sum_b_augrc + excluded_contribution) / len(SOURCES_ORDER)
    report.append(f"  excluded sources' rank_augrc contribution (data-derived, expected 5 "
                  f"if every excluded source ties at rank 1 in all 5 conditions): "
                  f"{excluded_contribution}")
    report.append(f"  LHS (sum of (a)'s 5 per-condition means) = {lhs:.6f}")
    report.append(f"  RHS ((b)'s rank_augrc grand sum {grand_sum_b_augrc} + "
                  f"{excluded_contribution}) / {len(SOURCES_ORDER)} = {rhs:.6f}")
    identity_ok = abs(lhs - rhs) < 1e-9
    report.append(f"  Equal: {identity_ok}")
    assert identity_ok, (
        f"15_augrc.py self-check failed: LHS={lhs!r} != RHS={rhs!r} -- the mean-rank "
        "identity between sections (a) and (b) does not hold; stopping before the "
        "report is written.")

    report.append("")
    report.append("-" * 78)
    report.append("Usage limitation (stated exactly, not derived further)")
    report.append("-" * 78)
    report.append("AUGRC scales with base_error the same way AURC does, so only "
                  "within-source condition comparisons are valid; cross-source absolute "
                  "comparisons are not.")

    report.append("")
    report.append("-" * 78)
    report.append("OCTDL-ambiguous")
    report.append("-" * 78)
    report.append("base_error = 0 under the missed-disease definition (0 of 248 rows "
                  "predicted NORMAL). augrc = 0.0 for all five conditions is a defined "
                  "result (generalized risk's integrand is 0/n = 0 throughout, never a "
                  "0/0), not a degenerate or \"perfect\" one -- it reflects the absence "
                  "of any error to detect, not a claim about ranking quality.")

    report.append("")
    report.append("-" * 78)
    report.append("Figures")
    report.append("-" * 78)
    report.append("15_F21_augrc_heatmap.png -- AUGRC, source x condition, [0,0.5] scale.")
    report.append("15_F22_augrc_vs_aurc.png -- scatter, aurc_x1000 vs augrc_x1000, coloured "
                  "by condition, marker shape by source, black ring = rank mismatch; "
                  "OCTDL-ambiguous excluded (aurc undefined).")
    report.append("15_F23_generalized_risk_curves.png -- one panel per source, five "
                  "generalized-risk-vs-coverage curves each.")

    report.append("")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    (TABDIR / "15_augrc.txt").write_text("\n".join(str(l) for l in report) + "\n")
    print(f"\nWrote results/tables/15_augrc.csv, results/tables/15_augrc.txt")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
