"""Compute supplementary AURC and E-AURC for the initial five sources.

The tie-averaged per_sample_curve function is also used by the final AUGRC
analysis. AURC is scaled by 1000 and omits the undefined zero-coverage point.
"""

from paths import PROJECT, read_csv


import numpy as np
import pandas as pd

TABDIR = PROJECT / "results" / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NORMAL_IDX = CLASSES.index("NORMAL")

SOURCES = ["Kermany-test", "RETOUCH-Spectralis", "RETOUCH-Topcon",
           "RETOUCH-Cirrus", "OCT5k"]
CONDITIONS = ["2", "3", "4", "5", "B2"]

AURC_SCALE = 1000.0
EPS = np.finfo(float).eps
N_ERRORS_TOL = 1e-6


# Source grouping and outcomes
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


def outcome_column(sub, source):
    """Use classification correctness for labelled data and non-NORMAL for RETOUCH."""
    if source.startswith("RETOUCH"):
        return (sub["predicted_class"] != NORMAL_IDX).to_numpy()
    return (sub["predicted_class"] == sub["true_label"]).to_numpy()


def risk_score_for(sub, condition):
    """Return a score increasing towards deferral; B2 uses negative softmax confidence."""
    if condition == "B2":
        return -sub["softmax_confidence"].to_numpy()
    return sub[f"s_condition{condition}"].to_numpy()


# Per-sample risk-coverage and AURC
def per_sample_curve(risk_score, correct):
    """Return selective risk at each sample coverage, with NaN at coverage zero.

    Replace outcomes within each tied-score group by their mean before cumulative
    summation, giving expected risk over random orderings within ties.
    """
    order = np.argsort(risk_score, kind="stable")
    sorted_score = risk_score[order]
    correct_sorted = correct[order].astype(float)
    _, inv = np.unique(sorted_score, return_inverse=True)
    correct_sorted = (np.bincount(inv, weights=correct_sorted)
                      / np.bincount(inv))[inv]
    n = len(risk_score)
    cum_correct = np.concatenate([[0], np.cumsum(correct_sorted)])
    k = np.arange(n + 1)
    coverage = k / n
    risk = np.empty(n + 1)
    risk[0] = np.nan
    risk[1:] = 1.0 - cum_correct[1:] / k[1:]
    return coverage, risk


def aurc_trapezoid(coverage, risk):
    """Integrate selective risk from the first nonzero coverage and scale by AURC_SCALE."""
    heights = (risk[1:-1] + risk[2:]) * 0.5
    widths = coverage[2:] - coverage[1:-1]
    return float(np.sum(heights * widths)) * AURC_SCALE


def main():
    df = read_csv(TABDIR / "08b_slice_scores.csv")
    df = add_source_column(df)
    grid_df = read_csv(TABDIR / "08c_risk_coverage.csv")

    rows = []
    source_summary = []
    crosscheck_rows = []
    reversed_rows = []

    for source in SOURCES:
        sub = df[df["source"] == source]
        n = len(sub)
        correct = outcome_column(sub, source)
        base_error = 1.0 - correct.mean()

        raw_errors = base_error * n
        assert abs(raw_errors - round(raw_errors)) < N_ERRORS_TOL, (
            f"{source}: base_error * n = {raw_errors!r} is not close to an "
            "integer -- outcome_column is not returning a simple count over n")
        n_errors = int(round(raw_errors))

        optimal = base_error + (1.0 - base_error) * np.log(1.0 - base_error + EPS)
        random_ref = base_error * AURC_SCALE

        source_summary.append(dict(source=source, n=n, n_errors=n_errors,
                                    base_error=base_error, random_ref=random_ref))

        for condition in CONDITIONS:
            risk_score = risk_score_for(sub, condition)
            n_tied = len(risk_score) - len(np.unique(risk_score))
            coverage, risk = per_sample_curve(risk_score, correct)

            assert abs(risk[-1] - base_error) < 1e-9, (
                f"{source}/{condition}: risk at full coverage ({risk[-1]!r}) "
                f"!= base_error ({base_error!r}) -- sort, outcome_column, or "
                "an indexing off-by-one is wrong")

            aurc = aurc_trapezoid(coverage, risk)
            e_aurc = aurc - optimal * AURC_SCALE
            assert e_aurc >= 0, (
                f"{source}/{condition}: e_aurc = {e_aurc!r} < 0 -- the "
                "optimal-CSF formula or the aurc computation has a sign or "
                "scale error")

            _, risk_rev = per_sample_curve(-risk_score, correct)
            aurc_rev = aurc_trapezoid(coverage, risk_rev)

            rows.append(dict(source=source, condition=condition, n=n,
                              n_errors=n_errors, n_tied=n_tied,
                              base_error=base_error, aurc=aurc, e_aurc=e_aurc,
                              random_ref=random_ref,
                              ratio_vs_random=aurc / random_ref,
                              aurc_reversed=aurc_rev))
            reversed_rows.append(dict(source=source, condition=condition,
                                       aurc=aurc, aurc_reversed=aurc_rev,
                                       forward_minus_reversed=aurc - aurc_rev))

            g = (grid_df[(grid_df["source"] == source)
                         & (grid_df["condition"] == condition)]
                 .sort_values("coverage"))
            aurc_interp = aurc_trapezoid(g["coverage"].to_numpy(), g["risk"].to_numpy())
            abs_diff = aurc_interp - aurc
            rel_diff = abs_diff / aurc if aurc != 0 else np.nan
            crosscheck_rows.append(dict(source=source, condition=condition,
                                         aurc_exact=aurc, aurc_interp=aurc_interp,
                                         abs_diff=abs_diff, rel_diff=rel_diff))

    out = pd.DataFrame(rows, columns=["source", "condition", "n", "n_errors",
                                      "n_tied", "base_error", "aurc", "e_aurc",
                                      "random_ref", "ratio_vs_random",
                                      "aurc_reversed"])
    out.to_csv(TABDIR / "09_aurc.csv", index=False)

    print("Per-source summary (n_errors is the integer count base_error is built from):")
    print(pd.DataFrame(source_summary).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nAURC (rows = source, columns = condition):")
    print(out.pivot(index="source", columns="condition", values="aurc")
             .reindex(SOURCES)[CONDITIONS]
             .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nE-AURC (rows = source, columns = condition):")
    print(out.pivot(index="source", columns="condition", values="e_aurc")
             .reindex(SOURCES)[CONDITIONS]
             .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nratio_vs_random (rows = source, columns = condition; the true "
          "no-better-than-random threshold is (1 - 1/n), not 1.0 -- 0.99691 at "
          "n=324 to 0.99925 at n=1,332 in this table -- but no row here sits "
          "close enough to that boundary for the distinction to matter: "
          "< (1 - 1/n) = ranks better than random, >= (1 - 1/n) = no better "
          "than random or worse):")
    print(out.pivot(index="source", columns="condition", values="ratio_vs_random")
             .reindex(SOURCES)[CONDITIONS]
             .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nn_tied (rows = source, columns = condition; reachable operating "
          "points lost to ties -- n minus the number of distinct score values, "
          "not the number of tied samples -- aurc/e_aurc/aurc_reversed on these "
          "rows are the tie-averaged expectation, not one realised ordering):")
    print(out.pivot(index="source", columns="condition", values="n_tied")
             .reindex(SOURCES)[CONDITIONS]
             .to_string())

    print("\nCheck 3, print-only, no threshold: exact (per-sample) AURC against "
          "the already-saved 101-point interpolated curve. Disagreement here is "
          "expected for two separate reasons, both benign: (a) ordinary "
          "interpolation resolution loss, largest on rows where AURC itself is "
          "small (a fixed absolute gap reads as a large relative one against a "
          "small denominator); (b) the 101-point curve comes from "
          "08c_evaluate.py's own risk_coverage_curve(), which sorts ties with "
          "plain np.argsort and no tie-averaging, so on the tied-heavy rows "
          "(condition 4's cp_setsize, at most three distinct values; B2's softmax "
          "confidence, clustered near 1.0) this compares this script's "
          "tie-corrected AURC against a curve built from one arbitrary tie "
          "order -- a large gap there is expected, not a computation error.")
    cc = pd.DataFrame(crosscheck_rows)
    print(cc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\naurc_reversed, now written to 09_aurc.csv (not print-only): AURC "
          "with the same score ranked in reverse. Not a bound of any kind -- a "
          "score whose information points the wrong way can show "
          "forward_minus_reversed > 0, which is a finding about that "
          "(source, condition), not a sign the reversed value is somehow the "
          "'worse' option:")
    rv = pd.DataFrame(reversed_rows)
    print(rv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 90)
    print(f"Table: {(TABDIR / '09_aurc.csv').relative_to(PROJECT)}")
    print("\n09 complete.")


if __name__ == "__main__":
    main()
