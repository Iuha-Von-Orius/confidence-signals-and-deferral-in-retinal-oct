"""Fit cost-based deferral policies on the calibration split.

Conditions 2-5 use OOD scores, mutual information, conformal set size, or
their combination. B2 is a cost-optimised softmax threshold. Policies use
a linear score with sigmoid expected cost, L2 regularisation and L-BFGS-B.
Validation supports the cost-ratio comparison; the selected ratio is 20:1.
Evaluation defers when the linear score is positive. No external data are fitted.
"""

from paths import PROJECT

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

CACHE = PROJECT / "cache"
RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "07_policy"
TABDIR = RESULTS / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)

SEED = 42

C_DEFER = 1.0
COST_RATIOS = [5, 10, 20, 50]

L2_LAMBDA = 1e-3
PARAM_BOUND = 50.0

SELECTED_RATIO = 20

COVERAGE_POINTS = [0.70, 0.80, 0.90]
COVERAGE_GRID = np.linspace(0.0, 1.0, 101)

CONDITIONS = {
    2: ["ood_cosine", "ood_maha_combined"],
    3: ["mutual_info"],
    4: ["cp_setsize"],
    5: ["ood_cosine", "ood_maha_combined", "mutual_info", "cp_setsize"],
}
CONDITION_LABELS = {
    1: "baseline", 2: "+OOD", 3: "+uncertainty", 4: "+conformal",
    5: "+all signals", "B2": "softmax threshold",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})

CONDITION_COLOURS = {
    1: "#7f8c8d", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd", 5: "#d62728",
    "B2": "#ff7f0e",
}
FEATURE_COLOURS = {
    "ood_cosine": "#1f77b4", "ood_maha_combined": "#d62728",
    "mutual_info": "#2ca02c", "cp_setsize": "#9467bd",
}
RATIO_COLOURS = ["#08306b", "#2171b5", "#6baed6", "#fdd0a2"]


# Data
def load_split(split):
    """Load cached calibration or validation signals for policy fitting and selection."""
    d = {
        "probs": np.load(CACHE / f"{split}_probs.npy"),
        "labels": np.load(CACHE / f"{split}_labels.npy"),
        "mc_probs": np.load(CACHE / f"{split}_mc_probs.npy"),
        "ood_cosine": np.load(CACHE / f"{split}_ood_cosine.npy"),
        "ood_maha_combined": np.load(CACHE / f"{split}_ood_maha_combined.npy"),
        "cp_setsize": np.load(
            CACHE / f"{split}_cp_setsize_mondrian_a0.02.npy").astype(np.float32),
    }
    d["prediction"] = d["probs"].argmax(axis=1)
    d["correct"] = d["prediction"] == d["labels"]
    d["confidence"] = d["probs"].max(axis=1)
    return d


def mutual_information(mc_probs):
    """Compute H[mean(p)] - mean(H[p]) across stochastic passes, in nats."""
    mc_mean = mc_probs.mean(axis=1)
    eps = 1e-12
    pred_entropy = -(mc_mean * np.log(mc_mean + eps)).sum(axis=1)
    mean_entropy = -(mc_probs * np.log(mc_probs + eps)).sum(axis=2).mean(axis=1)
    return pred_entropy - mean_entropy


def build_state(data, condition):
    """Stack signals in the fixed feature order defined by CONDITIONS."""
    names = CONDITIONS[condition]
    sources = {
        "ood_cosine": data["ood_cosine"], "ood_maha_combined": data["ood_maha_combined"],
        "mutual_info": data["mutual_info"], "cp_setsize": data["cp_setsize"],
    }
    X = np.stack([sources[n] for n in names], axis=1).astype(np.float32)
    return X, names


def standardize(X):
    """Fit column means and standard deviations on calibration data."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


# Policy fitting
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def expected_cost_and_grad(params, X, correct, c_err, c_defer):
    """Return sigmoid expected cost and its gradient, with L2 applied to weights only."""
    w, b = params[:-1], params[-1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s = X @ w + b
    p_defer = sigmoid(s)
    cost_accept = np.where(correct, 0.0, c_err)
    cost = (np.mean(p_defer * c_defer + (1 - p_defer) * cost_accept)
            + 0.5 * L2_LAMBDA * np.sum(w ** 2))

    d_cost_d_s = p_defer * (1 - p_defer) * (c_defer - cost_accept)
    n = len(correct)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        grad_w = X.T @ d_cost_d_s / n + L2_LAMBDA * w
    grad_b = d_cost_d_s.mean()
    return cost, np.concatenate([grad_w, [grad_b]])


def fit_policy(X, correct, c_err, c_defer):
    """Fit the bounded policy parameters with L-BFGS-B from a zero initialisation."""
    d = X.shape[1]
    x0 = np.zeros(d + 1)
    bounds = [(-PARAM_BOUND, PARAM_BOUND)] * (d + 1)
    cost_curve = []

    def callback(params):
        c, _ = expected_cost_and_grad(params, X, correct, c_err, c_defer)
        cost_curve.append(c)

    result = minimize(expected_cost_and_grad, x0, args=(X, correct, c_err, c_defer),
                       jac=True, method="L-BFGS-B", bounds=bounds, callback=callback)
    if not cost_curve:
        cost_curve = [expected_cost_and_grad(result.x, X, correct, c_err, c_defer)[0]]
    w, b = result.x[:-1], result.x[-1]
    return w, b, np.array(cost_curve)


def fit_b2_threshold(confidence, correct, c_err, c_defer):
    """Select the observed softmax cutoff with the lowest recorded prefix cost."""
    order = np.argsort(-confidence)
    correct_sorted = correct[order]
    conf_sorted = confidence[order]
    n = len(confidence)
    k = np.arange(1, n + 1)
    cum_correct = np.cumsum(correct_sorted)
    accepted_wrong = k - cum_correct
    cost = (accepted_wrong * c_err + (n - k) * c_defer) / n
    best_k = int(np.argmin(cost)) + 1
    return float(conf_sorted[best_k - 1])


# Evaluation
def natural_stats(risk_score, correct, threshold, c_err, c_defer):
    """Measure deferral, accepted error and cost when risk_score <= threshold is accepted."""
    defer = risk_score > threshold
    accepted = ~defer
    accepted_error_rate = float((~correct[accepted]).mean()) if accepted.sum() else np.nan
    cost_accept = np.where(correct, 0.0, c_err)
    expected_cost = float(np.mean(np.where(defer, c_defer, cost_accept)))
    return {
        "deferral_rate": float(defer.mean()),
        "accepted_error_rate": accepted_error_rate,
        "expected_cost": expected_cost,
        "n": int(len(correct)),
        "n_accepted": int(accepted.sum()),
    }


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
    risk_grid = np.interp(grid, coverage, risk)
    return grid, risk_grid


def threshold_at_coverage(risk_score, target_coverage):
    """Return the score quantile; tied scores may change the attained coverage."""
    return float(np.quantile(risk_score, target_coverage))


# Figures
def fig_learning_curves(fits):
    """Plot fitting cost by solver iteration for each condition and cost ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, condition in zip(axes.flat, [2, 3, 4, 5]):
        for ratio, colour in zip(COST_RATIOS, RATIO_COLOURS):
            curve = fits[(condition, ratio)]["cost_curve"]
            ax.plot(np.arange(1, len(curve) + 1), curve, "-", lw=1.8, color=colour,
                    label=f"{ratio}:1")
        ax.set_title(f"Condition {condition} ({CONDITION_LABELS[condition]})", fontsize=10)
        ax.set_xlabel("L-BFGS iteration")
        ax.set_ylabel("Expected cost")
        ax.legend(fontsize=7.5, title="cost ratio")
    fig.suptitle("Fitting convergence on calibration — diagnostic, not a result")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_learning_curves.png")
    plt.close(fig)


def fig_cost_ratio_sensitivity(sweep_df):
    """Plot validation deferral, accepted error and cost across cost ratios."""
    metrics = [("deferral_rate", "Deferral rate"),
               ("accepted_error_rate", "Accepted-sample error rate"),
               ("expected_cost", "Expected cost (units of c_defer)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (col, title) in zip(axes, metrics):
        for condition in [2, 3, 4, 5, "B2"]:
            sub = sweep_df[sweep_df.condition == condition].sort_values("ratio")
            ax.plot(sub["ratio"], sub[col], "o-", lw=1.8, ms=5,
                    color=CONDITION_COLOURS[condition],
                    label=f"{condition} {CONDITION_LABELS[condition]}")
        ax.set_xscale("log")
        ax.set_xlabel("Cost ratio  c_err : c_defer")
        ax.set_title(title, fontsize=10)
        if col == "deferral_rate":
            ax.legend(fontsize=7.5)
    fig.suptitle("Cost-ratio sensitivity on val — basis for SELECTED_RATIO")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_cost_ratio_sensitivity.png")
    plt.close(fig)


def fig_coefficient_trajectories(coef_df):
    """Plot fitted coefficients across cost ratios for conditions 2 and 5."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, condition in zip(axes, [2, 5]):
        sub = coef_df[(coef_df.condition == condition) & (coef_df.feature != "intercept")]
        for feature in CONDITIONS[condition]:
            fsub = sub[sub.feature == feature].sort_values("ratio")
            lw = 2.6 if feature == "ood_maha_combined" else 1.6
            ax.plot(fsub["ratio"], fsub["coefficient"], "o-", lw=lw, ms=5,
                    color=FEATURE_COLOURS[feature], label=feature)
        ax.axhline(0, color="black", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("Cost ratio  c_err : c_defer")
        ax.set_title(f"Condition {condition} ({CONDITION_LABELS[condition]})", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Standardised coefficient (defer direction)")
    fig.suptitle("Coefficient trajectories across the cost-ratio sweep")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_coefficient_trajectories.png")
    plt.close(fig)


def fig_mahalanobis_sign_check(coef_df):
    """Plot the fitted Mahalanobis coefficient and validation score-error relationship."""
    sub = coef_df[coef_df.feature == "ood_maha_combined"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.35
    ratios = sorted(sub["ratio"].unique())
    for i, condition in enumerate([2, 5]):
        csub = sub[sub.condition == condition].sort_values("ratio")
        x = np.arange(len(csub)) + i * width
        ax.bar(x, csub["coefficient"], width, label=f"condition {condition}",
               color=CONDITION_COLOURS[condition], alpha=0.85)
    ax.axhline(0, color="black", lw=1.2)
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color="#2ca02c", alpha=0.08)
    ax.set_ylim(ylim)
    ax.text(0.02, 0.95, "predicted region (step 05): positive",
            transform=ax.transAxes, fontsize=9, color="#1a6b1a", va="top")
    ax.set_xticks(np.arange(len(ratios)) + width / 2, [f"{r}:1" for r in ratios])
    ax.set_xlabel("Cost ratio  c_err : c_defer")
    ax.set_ylabel("ood_maha_combined coefficient")
    all_positive = bool((sub["coefficient"] > 0).all())
    verdict = ("PASS — sign matches the step 05 prediction at every ratio" if all_positive
               else "FAIL — sign does not hold at every ratio; see console report")
    ax.set_title(verdict, fontsize=10, color="#1a6b1a" if all_positive else "#8b1a1a")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_mahalanobis_sign_check.png")
    plt.close(fig)
    return all_positive


def fig_risk_coverage(curves, baseline_error):
    """Plot validation risk-coverage curves at the selected cost ratio."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for condition in [2, 3, 4, 5, "B2"]:
        cov, risk = curves[condition]
        ax.plot(cov, risk, "-", lw=2.0, color=CONDITION_COLOURS[condition],
                label=f"{condition} {CONDITION_LABELS[condition]}")
    ax.axhline(baseline_error, color=CONDITION_COLOURS[1], ls="--", lw=1.5,
               label=f"1 baseline (no deferral, error {baseline_error:.4f})")
    ax.set_xlabel("Coverage (fraction accepted)")
    ax.set_ylabel("Error rate among accepted")
    ax.set_title(f"Risk-coverage on val, cost ratio {SELECTED_RATIO}:1", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_risk_coverage.png")
    plt.close(fig)


def fig_risk_coverage_all_ratios(curves_all):
    """Compare validation risk-coverage curves across cost ratios."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, condition in zip(axes.flat, [2, 3, 4, 5]):
        for ratio, colour in zip(COST_RATIOS, RATIO_COLOURS):
            cov, risk = curves_all[(condition, ratio)]
            ax.plot(cov, risk, "-", lw=1.6, color=colour, label=f"{ratio}:1")
        ax.set_title(f"Condition {condition} ({CONDITION_LABELS[condition]})", fontsize=10)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Accepted error rate")
        ax.legend(fontsize=7.5, title="cost ratio")
    fig.suptitle("Risk-coverage curve shape across the cost-ratio sweep (val)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_risk_coverage_all_ratios.png")
    plt.close(fig)


def fig_natural_operating_points(sweep_df):
    """Plot deferral rates at each policy's fitted decision threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    conditions = [2, 3, 4, 5, "B2"]
    x = np.arange(len(COST_RATIOS))
    width = 0.15
    for i, condition in enumerate(conditions):
        sub = sweep_df[sweep_df.condition == condition].sort_values("ratio")
        ax.bar(x + (i - 2) * width, sub["deferral_rate"], width,
               label=f"{condition} {CONDITION_LABELS[condition]}",
               color=CONDITION_COLOURS[condition], alpha=0.85)
    ax.set_xticks(x, [f"{r}:1" for r in COST_RATIOS])
    ax.set_xlabel("Cost ratio  c_err : c_defer")
    ax.set_ylabel("Natural deferral rate (val)")
    ax.set_title("Deferral rate is an output: what each ratio produces unconstrained",
                fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_natural_operating_points.png")
    plt.close(fig)


def fig_per_class_deferral(per_class_df):
    """Plot deferral by true class at the selected cost ratio."""
    sub = per_class_df[per_class_df.ratio == SELECTED_RATIO]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(N_CLASSES)
    width = 0.15
    conditions = [2, 3, 4, 5, "B2"]
    for i, condition in enumerate(conditions):
        csub = sub[sub.condition == condition].set_index("class").loc[CLASSES]
        ax.bar(x + (i - 2) * width, csub["deferral_rate"], width,
               label=f"{condition} {CONDITION_LABELS[condition]}",
               color=CONDITION_COLOURS[condition], alpha=0.85)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Deferral rate")
    ax.set_title(f"Deferral rate by true class — val, cost ratio {SELECTED_RATIO}:1",
                fontsize=10)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_per_class_deferral.png")
    plt.close(fig)


def fig_per_class_accepted_error(matched_df):
    """Plot class-specific accepted error at matched coverage."""
    fig, axes = plt.subplots(1, len(COVERAGE_POINTS), figsize=(15, 4.8), sharey=True)
    for ax, cov in zip(axes, COVERAGE_POINTS):
        sub = matched_df[matched_df.coverage == cov]
        x = np.arange(N_CLASSES)
        width = 0.18
        for i, condition in enumerate([2, 3, 4, 5, "B2"]):
            csub = sub[sub.condition == condition].set_index("class").loc[CLASSES]
            ax.bar(x + (i - 2) * width, csub["accepted_error_rate"], width,
                   color=CONDITION_COLOURS[condition], alpha=0.85,
                   label=f"{condition} {CONDITION_LABELS[condition]}")
        ax.set_xticks(x, CLASSES)
        ax.set_title(f"Coverage {cov:.0%}", fontsize=10)
    axes[0].set_ylabel("Accepted-sample error rate")
    axes[-1].legend(fontsize=7.5)
    fig.suptitle("Per-class accepted error at matched coverage (val)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_per_class_accepted_error.png")
    plt.close(fig)


def fig_condition_heatmap_error(curves, baseline_error):
    """Plot accepted error by condition and target coverage."""
    rows = [1, 2, 3, 4, 5, "B2"]
    mat = np.zeros((len(rows), len(COVERAGE_POINTS)))
    for i, condition in enumerate(rows):
        for j, cov in enumerate(COVERAGE_POINTS):
            idx = int(round(cov * (len(COVERAGE_GRID) - 1)))
            mat[i, j] = baseline_error if condition == 1 else curves[condition][1][idx]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(COVERAGE_POINTS)), [f"{c:.0%}" for c in COVERAGE_POINTS])
    ax.set_yticks(np.arange(len(rows)), [f"{c} {CONDITION_LABELS[c]}" for c in rows])
    for i in range(len(rows)):
        for j in range(len(COVERAGE_POINTS)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("Coverage")
    ax.set_title("Accepted error rate — condition x coverage (val)", fontsize=10)
    fig.colorbar(im, ax=ax, label="Accepted error rate")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_condition_heatmap_error.png")
    plt.close(fig)


def fig_condition_heatmap_deferral(sweep_df):
    """Plot deferral rate by condition and cost ratio."""
    rows = [2, 3, 4, 5, "B2"]
    mat = np.zeros((len(rows), len(COST_RATIOS)))
    for i, condition in enumerate(rows):
        for j, ratio in enumerate(COST_RATIOS):
            sub = sweep_df[(sweep_df.condition == condition) & (sweep_df.ratio == ratio)]
            mat[i, j] = sub["deferral_rate"].values[0]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(COST_RATIOS)), [f"{r}:1" for r in COST_RATIOS])
    ax.set_yticks(np.arange(len(rows)), [f"{c} {CONDITION_LABELS[c]}" for c in rows])
    for i in range(len(rows)):
        for j in range(len(COST_RATIOS)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 0.5 else "black")
    ax.set_xlabel("Cost ratio  c_err : c_defer")
    ax.set_title("Natural deferral rate — condition x ratio (val)", fontsize=10)
    fig.colorbar(im, ax=ax, label="Deferral rate")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_condition_heatmap_deferral.png")
    plt.close(fig)


def fig_gain_over_b2(curves):
    """Plot B2 risk minus policy risk; positive values favour the learned policy."""
    cov_grid, b2_risk = curves["B2"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for condition in [2, 3, 4, 5]:
        cov, risk = curves[condition]
        gain = b2_risk - risk
        ax.plot(cov, gain, "-", lw=2.0, color=CONDITION_COLOURS[condition],
                label=f"{condition} {CONDITION_LABELS[condition]}")
    for cov in COVERAGE_POINTS:
        ax.axvline(cov, color="grey", ls=":", lw=1.0)
    ax.axhline(0, color="black", lw=1.2)
    ylim = ax.get_ylim()
    ax.fill_between(cov_grid, 0, ylim[1], color="#2ca02c", alpha=0.05)
    ax.set_ylim(ylim)
    ax.text(0.02, 0.97, "policy beats B2", transform=ax.transAxes, fontsize=8.5,
            color="#1a6b1a", va="top")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("B2 risk − policy risk  (positive = policy wins)")
    ax.set_title(f"Gain over B2, cost ratio {SELECTED_RATIO}:1 (val)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_gain_over_b2.png")
    plt.close(fig)


def fig_score_distributions(fits, val):
    """Plot fitted scores separately for correct and incorrect predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, condition in zip(axes.flat, [2, 3, 4, 5]):
        s = fits[(condition, SELECTED_RATIO)]["risk_score_val"]
        ax.hist(s[val["correct"]], bins=60, alpha=0.55, density=True,
                label="correct", color="#2ca02c")
        ax.hist(s[~val["correct"]], bins=60, alpha=0.55, density=True,
                label="incorrect", color="#d62728")
        ax.axvline(0, color="black", ls="--", lw=1.2, label="natural threshold")
        ax.set_title(f"Condition {condition} ({CONDITION_LABELS[condition]})", fontsize=10)
        ax.set_xlabel("s(x)  (> 0 defers)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Policy score by outcome — val, cost ratio {SELECTED_RATIO}:1")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_score_distributions.png")
    plt.close(fig)


# Main
def main():
    np.random.seed(SEED)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    print("Loading calibration and val...")
    cal = load_split("calibration")
    val = load_split("val")
    cal["mutual_info"] = mutual_information(cal["mc_probs"])
    val["mutual_info"] = mutual_information(val["mc_probs"])
    print(f"  calibration {len(cal['labels']):>6,}  accuracy {cal['correct'].mean():.4f}")
    print(f"  val         {len(val['labels']):>6,}  accuracy {val['correct'].mean():.4f}")
    baseline_acc = float(val["correct"].mean())
    baseline_error = 1.0 - baseline_acc

    # Fit conditions 2-5 across the cost-ratio sweep, on calibration
    print("\nFitting policies on calibration...")
    fits = {}
    for condition in [2, 3, 4, 5]:
        X_cal_raw, names = build_state(cal, condition)
        X_cal, mean, std = standardize(X_cal_raw)
        X_val_raw, _ = build_state(val, condition)
        X_val = (X_val_raw - mean) / std
        for ratio in COST_RATIOS:
            w, b, curve = fit_policy(X_cal, cal["correct"], ratio, C_DEFER)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                risk_score_val = X_val @ w + b
            fits[(condition, ratio)] = {
                "w": w, "b": b, "mean": mean, "std": std,
                "feature_names": names, "cost_curve": curve,
                "risk_score_val": risk_score_val,
            }
            print(f"  condition {condition} ratio {ratio:>3}:1  "
                  f"{len(curve):>4} iters  final cost {curve[-1]:.4f}")

    # B2's threshold, per ratio, fit on calibration
    b2_fits = {ratio: fit_b2_threshold(cal["confidence"], cal["correct"], ratio, C_DEFER)
               for ratio in COST_RATIOS}

    rows = []
    for ratio in COST_RATIOS:
        rows.append({"condition": 1, "ratio": ratio, "deferral_rate": 0.0,
                     "accepted_error_rate": baseline_error,
                     "expected_cost": baseline_error * ratio,
                     "n": len(val["labels"]), "n_accepted": len(val["labels"])})
    for condition in [2, 3, 4, 5]:
        for ratio in COST_RATIOS:
            f = fits[(condition, ratio)]
            stats = natural_stats(f["risk_score_val"], val["correct"], 0.0, ratio, C_DEFER)
            rows.append({"condition": condition, "ratio": ratio, **stats})
    for ratio in COST_RATIOS:
        t = b2_fits[ratio]
        stats = natural_stats(-val["confidence"], val["correct"], -t, ratio, C_DEFER)
        rows.append({"condition": "B2", "ratio": ratio, **stats})
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(TABDIR / "07_cost_ratio_sweep.csv", index=False)

    # Coefficients (conditions 2 and 5, all ratios)
    coef_rows = []
    for condition in [2, 5]:
        for ratio in COST_RATIOS:
            f = fits[(condition, ratio)]
            for name, coef in zip(f["feature_names"], f["w"]):
                coef_rows.append({"condition": condition, "ratio": ratio,
                                  "feature": name, "coefficient": float(coef)})
            coef_rows.append({"condition": condition, "ratio": ratio,
                              "feature": "intercept", "coefficient": float(f["b"])})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(TABDIR / "07_coefficients.csv", index=False)

    # Risk-coverage curves
    curves = {condition: risk_coverage_curve(fits[(condition, SELECTED_RATIO)]["risk_score_val"],
                                             val["correct"])
              for condition in [2, 3, 4, 5]}
    curves["B2"] = risk_coverage_curve(-val["confidence"], val["correct"])

    curves_all = {(condition, ratio): risk_coverage_curve(
        fits[(condition, ratio)]["risk_score_val"], val["correct"])
        for condition in [2, 3, 4, 5] for ratio in COST_RATIOS}

    rc_rows = [{"condition": c, "coverage": float(cov), "risk": float(r)}
               for c in [2, 3, 4, 5, "B2"] for cov, r in zip(*curves[c])]
    pd.DataFrame(rc_rows).to_csv(TABDIR / "07_risk_coverage.csv", index=False)

    rc_all_rows = [{"condition": c, "ratio": r, "coverage": float(cov), "risk": float(risk)}
                   for (c, r), (cov_arr, risk_arr) in curves_all.items()
                   for cov, risk in zip(cov_arr, risk_arr)]
    pd.DataFrame(rc_all_rows).to_csv(TABDIR / "07_risk_coverage_all_ratios.csv", index=False)

    # Per-class deferral (natural threshold, all ratios)
    per_class_rows = []
    for condition in [2, 3, 4, 5]:
        for ratio in COST_RATIOS:
            rs = fits[(condition, ratio)]["risk_score_val"]
            defer = rs > 0.0
            for k, cls in enumerate(CLASSES):
                m = val["labels"] == k
                acc_m = m & ~defer
                per_class_rows.append({
                    "condition": condition, "ratio": ratio, "class": cls,
                    "deferral_rate": float(defer[m].mean()) if m.sum() else np.nan,
                    "accepted_error_rate": float((~val["correct"][acc_m]).mean())
                        if acc_m.sum() else np.nan,
                    "n_class": int(m.sum()), "n_accepted": int(acc_m.sum()),
                })
    for ratio in COST_RATIOS:
        t = b2_fits[ratio]
        defer = val["confidence"] < t
        for k, cls in enumerate(CLASSES):
            m = val["labels"] == k
            acc_m = m & ~defer
            per_class_rows.append({
                "condition": "B2", "ratio": ratio, "class": cls,
                "deferral_rate": float(defer[m].mean()) if m.sum() else np.nan,
                "accepted_error_rate": float((~val["correct"][acc_m]).mean())
                    if acc_m.sum() else np.nan,
                "n_class": int(m.sum()), "n_accepted": int(acc_m.sum()),
            })
    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(TABDIR / "07_per_class_deferral.csv", index=False)

    # Per-class accepted error at matched coverage (SELECTED_RATIO)
    matched_rows = []
    for condition in [2, 3, 4, 5]:
        rs = fits[(condition, SELECTED_RATIO)]["risk_score_val"]
        for cov in COVERAGE_POINTS:
            tau = threshold_at_coverage(rs, cov)
            accepted = rs <= tau
            for k, cls in enumerate(CLASSES):
                m = val["labels"] == k
                acc_m = m & accepted
                matched_rows.append({
                    "condition": condition, "coverage": cov, "class": cls,
                    "accepted_error_rate": float((~val["correct"][acc_m]).mean())
                        if acc_m.sum() else np.nan,
                    "n_accepted": int(acc_m.sum()), "n_class": int(m.sum()),
                })
    rs_b2 = -val["confidence"]
    for cov in COVERAGE_POINTS:
        tau = threshold_at_coverage(rs_b2, cov)
        accepted = rs_b2 <= tau
        for k, cls in enumerate(CLASSES):
            m = val["labels"] == k
            acc_m = m & accepted
            matched_rows.append({
                "condition": "B2", "coverage": cov, "class": cls,
                "accepted_error_rate": float((~val["correct"][acc_m]).mean())
                    if acc_m.sum() else np.nan,
                "n_accepted": int(acc_m.sum()), "n_class": int(m.sum()),
            })
    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(TABDIR / "07_per_class_accepted_error.csv", index=False)

    # Gain over B2
    gain_rows = []
    cov_grid, b2_risk_grid = curves["B2"]
    for condition in [2, 3, 4, 5]:
        _, risk_grid = curves[condition]
        for cov in COVERAGE_POINTS:
            idx = int(round(cov * (len(COVERAGE_GRID) - 1)))
            gain_rows.append({
                "condition": condition, "coverage": cov,
                "policy_risk": float(risk_grid[idx]), "b2_risk": float(b2_risk_grid[idx]),
                "gain": float(b2_risk_grid[idx] - risk_grid[idx]),
            })
    pd.DataFrame(gain_rows).to_csv(TABDIR / "07_gain_over_b2.csv", index=False)

    # Score distribution summary
    dist_rows = []
    for condition in [2, 3, 4, 5]:
        s = fits[(condition, SELECTED_RATIO)]["risk_score_val"]
        for label, mask in [("correct", val["correct"]), ("incorrect", ~val["correct"])]:
            vals = s[mask]
            dist_rows.append({
                "condition": condition, "outcome": label,
                "mean": float(vals.mean()), "std": float(vals.std()),
                "q25": float(np.percentile(vals, 25)), "median": float(np.median(vals)),
                "q75": float(np.percentile(vals, 75)), "n": int(mask.sum()),
            })
    pd.DataFrame(dist_rows).to_csv(TABDIR / "07_score_distribution_summary.csv", index=False)

    # Consolidated master table
    master_rows = []
    for _, row in sweep_df.iterrows():
        r = dict(row)
        condition, ratio = row["condition"], row["ratio"]
        if condition in (2, 3, 4, 5):
            f = fits[(condition, ratio)]
            r["n_iterations"] = int(len(f["cost_curve"]))
            r["final_training_cost"] = float(f["cost_curve"][-1])
            for name, coef in zip(f["feature_names"], f["w"]):
                r[f"coef_{name}"] = float(coef)
            r["intercept"] = float(f["b"])
        master_rows.append(r)
    pd.DataFrame(master_rows).to_csv(TABDIR / "07_condition_ratio_master.csv", index=False)

    # Figures
    print("\nGenerating figures...")
    fig_learning_curves(fits)
    fig_cost_ratio_sensitivity(sweep_df)
    fig_coefficient_trajectories(coef_df)
    sign_check_pass = fig_mahalanobis_sign_check(coef_df)
    fig_risk_coverage(curves, baseline_error)
    fig_risk_coverage_all_ratios(curves_all)
    fig_natural_operating_points(sweep_df)
    fig_per_class_deferral(per_class_df)
    fig_per_class_accepted_error(matched_df)
    fig_condition_heatmap_error(curves, baseline_error)
    fig_condition_heatmap_deferral(sweep_df)
    fig_gain_over_b2(curves)
    fig_score_distributions(fits, val)

    # cache/policy_params.json — the one artifact step 08 reads
    policy_params = {
        "cost_ratios": COST_RATIOS,
        "c_defer": C_DEFER,
        "selected_cost_ratio": SELECTED_RATIO,
        "conditions": {
            str(condition): {
                "feature_names": fits[(condition, COST_RATIOS[0])]["feature_names"],
                "standardize_mean": fits[(condition, SELECTED_RATIO)]["mean"].tolist(),
                "standardize_std": fits[(condition, SELECTED_RATIO)]["std"].tolist(),
                "weights": {
                    str(ratio): {"w": fits[(condition, ratio)]["w"].tolist(),
                                 "b": float(fits[(condition, ratio)]["b"])}
                    for ratio in COST_RATIOS
                },
            }
            for condition in [2, 3, 4, 5]
        },
        "B2": {
            "thresholds": {str(ratio): b2_fits[ratio] for ratio in COST_RATIOS},
            "selected_threshold": b2_fits[SELECTED_RATIO],
        },
    }
    with open(CACHE / "policy_params.json", "w") as fh:
        json.dump(policy_params, fh, indent=2)

    # Console report
    print("\n" + "=" * 96)
    print("Cost-ratio sensitivity (val):\n")
    show = sweep_df[["condition", "ratio", "deferral_rate", "accepted_error_rate",
                     "expected_cost", "n_accepted", "n"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nMahalanobis sign check (conditions 2 and 5, expected sign: positive):")
    maha = coef_df[coef_df.feature == "ood_maha_combined"]
    for _, r in maha.iterrows():
        flag = "" if r["coefficient"] > 0 else "  <-- UNEXPECTED SIGN"
        print(f"  condition {r['condition']}  ratio {r['ratio']:>3}:1  "
              f"coef {r['coefficient']:+.4f}{flag}")
    print(f"  Positive Mahalanobis coefficients across ratios: {sign_check_pass}")
    print(f"\nSelected cost ratio: {SELECTED_RATIO}:1 (validation comparison).")

    print(f"\nFigures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("07_*.csv")):
        print(f"  {f.name}")
    print(f"\ncache/policy_params.json written. test and cross-device data untouched.")


if __name__ == "__main__":
    main()
