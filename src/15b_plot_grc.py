"""Plot generalised risk curves and integrate the 0.70-0.90 coverage window.

Reuse the full-range curves from step 15. The window metric is divided
by the window width and is therefore a mean generalised risk.
"""

from paths import PROJECT, read_csv

import importlib
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
_augrc15 = importlib.import_module("15_augrc")

compute_augrc = _augrc15.compute_augrc
augrc_trapezoid = _augrc15.augrc_trapezoid
load_merged_df = _augrc15.load_merged_df
correct_for = _augrc15.correct_for
risk_definition_for = _augrc15.risk_definition_for
risk_score_for = _augrc15.risk_score_for
SOURCES_ORDER = _augrc15.SOURCES_ORDER
CONDITIONS_AURC = _augrc15.CONDITIONS_AURC
CONDITION_COLOURS = _augrc15.CONDITION_COLOURS
CONDITION_LABELS = _augrc15.CONDITION_LABELS
TABDIR = _augrc15.TABDIR
FIGDIR = _augrc15.FIGDIR
PROJECT = _augrc15.PROJECT

LO = 0.70
HI = 0.90
MATCHED_COVERAGE_POINTS = [0.70, 0.80, 0.90]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 220, "savefig.bbox": "tight",
    "font.size": 14, "axes.titlesize": 13, "axes.labelsize": 14,
    "legend.fontsize": 9.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3,
})


def window_augrc(coverage, gen_risk, lo=LO, hi=HI):
    """Integrate the piecewise-linear curve over [lo, hi] and divide by window width."""
    interior_mask = (coverage > lo) & (coverage < hi)
    nodes_cov = np.unique(np.concatenate([[lo], coverage[interior_mask], [hi]]))
    nodes_risk = np.interp(nodes_cov, coverage, gen_risk)
    heights = (nodes_risk[:-1] + nodes_risk[1:]) * 0.5
    widths = nodes_cov[1:] - nodes_cov[:-1]
    integral = float(np.sum(heights * widths))
    return integral / (hi - lo)


def build_window_table(df_full, curves):
    rows = []
    for _, r in df_full.iterrows():
        source, condition = r["source"], r["condition"]
        coverage, gen_risk = curves[(source, condition)]
        aw = window_augrc(coverage, gen_risk)
        base_error = float(r["base_error"])
        if base_error == 0:
            ratio = "undefined (0 errors)"
        else:
            ratio = aw / (base_error * 0.8)
        rows.append({
            "source": source, "condition": condition,
            "risk_definition": r["risk_definition"], "n": int(r["n"]),
            "base_error": base_error, "augrc_window": aw,
            "augrc_window_x1000": aw * 1000.0,
            "augrc_window_ratio_vs_random": ratio,
            "augrc_x1000": float(r["augrc_x1000"]),
        })
    df = pd.DataFrame(rows)

    # The zero-error ambiguous tier has no informative ranking.
    non_ambiguous = df[df["source"] != "OCTDL-ambiguous"].copy()
    non_ambiguous["rank_augrc_window"] = (
        non_ambiguous.groupby("source")["augrc_window"]
        .transform(lambda s: s.rank(method="dense", ascending=True)))
    df = df.merge(
        non_ambiguous[["source", "condition", "rank_augrc_window"]],
        on=["source", "condition"], how="left")
    df["rank_augrc_window"] = df["rank_augrc_window"].astype("Int64")
    return df


def working_point_coverage(sub, condition):
    if condition == "B2":
        col = "defer_B2"
    else:
        col = f"defer_condition{condition}"
    defer = sub[col].to_numpy()
    assert defer.dtype == bool, f"{col} is {defer.dtype}, expected bool"
    return float((~defer).mean())


def panel_ordering(merged):
    means = merged.groupby("source")["ood_maha_combined"].mean()
    return means.reindex(SOURCES_ORDER).sort_values(ascending=False)


def fig_f24(merged, df_full, curves, maha_means):
    order = [s for s in maha_means.index if s != "OCTDL-ambiguous"]
    n = len(order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.3 * nrows))
    for ax, source in zip(axes.flat, order):
        sub = merged[merged["source"] == source]
        base_error = float(df_full.loc[df_full["source"] == source, "base_error"].iloc[0])
        n_source = len(sub)
        degenerate = base_error == 0

        for condition in CONDITIONS_AURC:
            cov, gen_risk = curves[(source, condition)]
            row = df_full[(df_full["source"] == source) & (df_full["condition"] == condition)]
            augrc_x1000 = float(row["augrc_x1000"].iloc[0])
            label = (f"{condition} {CONDITION_LABELS[condition]} "
                    f"(AUGRC×1000={augrc_x1000:.1f})")
            ax.plot(cov, gen_risk, "-", lw=2, color=CONDITION_COLOURS[condition], label=label)

            wp_cov = working_point_coverage(sub, condition)
            wp_risk = float(np.interp(wp_cov, cov, gen_risk))
            ax.plot(wp_cov, wp_risk, "o", color=CONDITION_COLOURS[condition],
                   markersize=9, markeredgecolor="black", markeredgewidth=0.8, zorder=5)

        ax.plot([0, 1], [0, base_error], "--", color="grey", lw=1.6, zorder=1,
               label=f"random (AUGRC×1000={base_error * 500:.1f})")

        title = f"{source}\nmean ood_maha_combined={maha_means[source]:+.4f}  " \
               f"base_error={base_error:.4f}  n={n_source}"
        if degenerate:
            title += "\n[degenerate: 0 errors, curve flat at 0]"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Generalized risk")
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Generalized risk vs coverage, annotated: random baseline (dashed), "
                "per-condition working point (dot), panels ordered by mean "
                "ood_maha_combined (descending)", fontsize=14)
    fig.text(0.5, 0.005, "OCTDL-ambiguous is omitted: it has zero errors, so its "
            "generalised risk is identically zero at every coverage and carries no "
            "ranking information.", ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(FIGDIR / "15b_F24_grc_annotated.png")
    plt.close(fig)


def fig_f25(window_df, curves, maha_means):
    order = [s for s in maha_means.index if s != "OCTDL-ambiguous"]
    n = len(order)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.3 * nrows))
    for ax, source in zip(axes.flat, order):
        base_error = float(window_df.loc[window_df["source"] == source, "base_error"].iloc[0])
        y_all = []
        for condition in CONDITIONS_AURC:
            cov, gen_risk = curves[(source, condition)]
            mask = (cov >= LO) & (cov <= HI)
            cov_w = cov[mask]
            risk_w = gen_risk[mask]
            row = window_df[(window_df["source"] == source) & (window_df["condition"] == condition)]
            aw_x1000 = float(row["augrc_window_x1000"].iloc[0])
            label = f"{condition} {CONDITION_LABELS[condition]} (window mean×1000={aw_x1000:.1f})"
            ax.plot(cov_w, risk_w, "-", lw=2.2, color=CONDITION_COLOURS[condition], label=label)
            y_all.append(risk_w)

        random_cov = np.array([LO, HI])
        random_risk = base_error * random_cov
        ax.plot(random_cov, random_risk, "--", color="grey", lw=1.6)
        y_all.append(random_risk)

        y_concat = np.concatenate(y_all)
        pad = (y_concat.max() - y_concat.min()) * 0.08 + 1e-9
        ax.set_xlim(LO, HI)
        ax.set_ylim(y_concat.min() - pad, y_concat.max() + pad)
        ax.set_title(source, fontsize=12)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Generalized risk")
        ax.legend(fontsize=7.5, loc="best")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"Generalized risk vs coverage, [{LO:.0%}, {HI:.0%}] window",
                fontsize=14)
    fig.text(0.5, 0.005, "OCTDL-ambiguous is omitted: it has zero errors, so its "
            "generalised risk is identically zero at every coverage and carries no "
            "ranking information.", ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(FIGDIR / "15b_F25_grc_window_70_90.png")
    plt.close(fig)


def main():
    t0 = time.time()
    TABDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 78)
    report.append("15b_plot_grc.py -- windowed AUGRC and annotated generalized-risk curves")
    report.append("=" * 78)
    report.append("")
    print("Loading merged nine-source dataframe and full-range AUGRC (compute_augrc)...")
    merged = load_merged_df()
    df_full, curves = compute_augrc(merged)

    print("Regression check against 15_augrc.csv...")
    ref = read_csv(TABDIR / "15_augrc.csv")
    ref["condition"] = ref["condition"].astype(str)
    mine = df_full[["source", "condition", "augrc_x1000"]].copy()
    mine["condition"] = mine["condition"].astype(str)
    cmp_df = mine.merge(ref[["source", "condition", "augrc_x1000"]], on=["source", "condition"],
                        suffixes=("_mine", "_ref"))
    diff = (cmp_df["augrc_x1000_mine"] - cmp_df["augrc_x1000_ref"]).abs()
    max_diff = float(diff.max())
    report.append("")
    report.append("-" * 78)
    report.append("Regression check: this script's full-range augrc_x1000 (from its own "
                  "compute_augrc call) vs results/tables/15_augrc.csv's augrc_x1000")
    report.append("-" * 78)
    report.append(f"  rows compared: {len(cmp_df)}   max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-9, (
        f"15b_plot_grc.py regression check failed: max abs diff {max_diff!r} against "
        "15_augrc.csv -- curves are not reproducing the same AUGRC as 15_augrc.py's own "
        "run. Stopping before any figure is written.")
    report.append("  PASS -- curves reused from compute_augrc reproduce 15_augrc.csv exactly.")

    print("Building the [0.70, 0.90] window table...")
    window_df = build_window_table(df_full, curves)

    window_df["_condition_str"] = window_df["condition"].astype(str)
    ref_ranks = ref[["source", "condition", "rank_augrc"]].rename(
        columns={"condition": "_condition_str"})
    window_df = window_df.merge(ref_ranks, on=["source", "_condition_str"], how="left")
    missing = window_df[window_df["rank_augrc"].isna()]
    assert len(missing) == 0, (
        "15b_plot_grc.py: rank_augrc merge from 15_augrc.csv failed to match "
        f"(source, condition) pairs: "
        f"{list(zip(missing['source'], missing['_condition_str']))}")
    window_df = window_df.drop(columns=["_condition_str"])
    window_df["rank_augrc"] = window_df["rank_augrc"].astype("Int64")

    window_df = window_df[["source", "condition", "risk_definition", "n", "base_error",
                           "augrc_window", "augrc_window_x1000", "augrc_window_ratio_vs_random",
                           "rank_augrc_window", "augrc_x1000", "rank_augrc"]]
    window_df.to_csv(TABDIR / "15b_augrc_window.csv", index=False)

    print("Computing panel ordering (mean ood_maha_combined, descending)...")
    maha_means = panel_ordering(merged)

    print("Building figures F24, F25...")
    fig_f24(merged, df_full, curves, maha_means)
    fig_f25(window_df, curves, maha_means)

    elapsed = time.time() - t0

    report.append("")
    report.append("-" * 78)
    report.append(f"15b_augrc_window.csv -- window [{LO}, {HI}], full contents")
    report.append("-" * 78)
    report.append(window_df.to_string(index=False))

    report.append("")
    report.append("-" * 78)
    report.append("Full-range vs window mean rank, per condition, like-for-like: both over "
                  "the 8 sources excluding OCTDL-ambiguous")
    report.append("-" * 78)
    non_amb = window_df[window_df["source"] != "OCTDL-ambiguous"]
    for c in CONDITIONS_AURC:
        c_str = str(c)
        sub = non_amb[non_amb["condition"].astype(str) == c_str]
        mean_full = float(sub["rank_augrc"].mean())
        mean_window = float(sub["rank_augrc_window"].mean())
        report.append(f"  condition {c_str}: mean rank_augrc (full range)={mean_full:.4f}  "
                      f"mean rank_augrc_window ([{LO},{HI}])={mean_window:.4f}")
    report.append("  n_sources=8, excluded=['OCTDL-ambiguous']")

    report.append("")
    report.append("-" * 78)
    report.append("mean ood_maha_combined per source, and the resulting panel order "
                  "(descending) used in F24/F25")
    report.append("-" * 78)
    for source, val in maha_means.items():
        report.append(f"  {source}: {val:+.6f}")
    figure_order = [s for s in maha_means.index if s != "OCTDL-ambiguous"]
    report.append("  panel order in F24/F25 (OCTDL-ambiguous excluded from both figures: "
                  "zero errors, generalised risk identically zero, no ranking information): "
                  + " > ".join(figure_order))

    report.append("")
    report.append("-" * 78)
    report.append("Window choice")
    report.append("-" * 78)
    report.append(f"[{LO}, {HI}] uses the same three coverage levels "
                  f"({MATCHED_COVERAGE_POINTS}) as this project's matched-coverage tables "
                  "(see 14_cost_and_coverage.py's T16/matched-coverage section) -- that "
                  "range was already justified there on clinical-feasibility grounds; it "
                  "is not selected post hoc in this script.")

    report.append("")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    (TABDIR / "15b_grc.txt").write_text("\n".join(str(l) for l in report) + "\n")
    print(f"\nWrote results/tables/15b_augrc_window.csv, results/tables/15b_grc.txt")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
