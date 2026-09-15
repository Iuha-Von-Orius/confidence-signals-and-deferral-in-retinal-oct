"""Apply frozen detectors, conformal quantiles and policies to cached features.

Writes per-image scores and decisions for Kermany test, RETOUCH and RASTI.
Internal oct5k names refer to RASTI.
"""

from paths import PROJECT, read_csv

import json

import numpy as np
import pandas as pd

CACHE = PROJECT / "cache"
SPLITS = PROJECT / "splits"
TABDIR = PROJECT / "results" / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)

REGRESSION_REFERENCE = {
    "cosine": (0.7931, 0.1087),
    "maha_combined": (0.189, 0.551),
}
REGRESSION_TOLERANCE = 0.01


# Detector application (from 05)
def cosine_score(features, centroids):
    """Return the maximum cosine similarity to a class centroid."""
    f_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return (f_norm @ c_norm.T).max(axis=1)


def mahalanobis_score(features, centroids, precision, batch=4096):
    """Return the negative minimum squared Mahalanobis distance across classes."""
    n = len(features)
    best = np.full(n, np.inf)
    feats64 = features.astype(np.float64)
    for k in range(N_CLASSES):
        d_k = np.empty(n)
        for start in range(0, n, batch):
            chunk = feats64[start:start + batch] - centroids[k]
            d_k[start:start + batch] = ((chunk @ precision) * chunk).sum(axis=1)
        best = np.minimum(best, d_k)
    return -best


def score_detectors(prefix, ood_params):
    """Apply frozen cosine and six-depth standardised Mahalanobis detectors."""
    feats6 = np.load(CACHE / f"{prefix}_feat_stage6.npy")
    cos = cosine_score(feats6, ood_params["centroids_6"])

    z_all = []
    for stage in range(1, 7):
        feats = np.load(CACHE / f"{prefix}_feat_stage{stage}.npy")
        raw = mahalanobis_score(feats, ood_params[f"centroids_{stage}"],
                                ood_params[f"precision_{stage}"])
        mu = ood_params[f"z_mean_{stage}"][0]
        sd = ood_params[f"z_std_{stage}"][0]
        z_all.append((raw - mu) / (sd + 1e-12))
    combined = np.mean(z_all, axis=0)
    return cos.astype(np.float32), combined.astype(np.float32)


def mutual_information(mc_probs):
    """Compute H[mean(p)] - mean(H[p]) across stochastic passes, in nats."""
    mc_mean = mc_probs.mean(axis=1)
    eps = 1e-12
    pred_entropy = -(mc_mean * np.log(mc_mean + eps)).sum(axis=1)
    mean_entropy = -(mc_probs * np.log(mc_probs + eps)).sum(axis=2).mean(axis=1)
    return pred_entropy - mean_entropy


def compute_cp_setsize(probs, q_mondrian_alpha):
    """Return Mondrian set sizes using the frozen alpha = 0.02 class quantiles."""
    q = {k: q_mondrian_alpha[CLASSES[k]] for k in range(N_CLASSES)}
    sets = np.zeros_like(probs, dtype=bool)
    for k in range(N_CLASSES):
        sets[:, k] = (1.0 - probs[:, k]) <= q[k]
    return sets.sum(axis=1).astype(np.int8)


# Policy application (from 07)
def apply_policy(condition, feature_dict, policy_params, selected_ratio):
    cond = policy_params["conditions"][str(condition)]
    names = cond["feature_names"]
    X = np.stack([feature_dict[n] for n in names], axis=1).astype(np.float32)
    mean = np.array(cond["standardize_mean"], dtype=np.float32)
    std = np.array(cond["standardize_std"], dtype=np.float32)
    Xz = (X - mean) / std
    w = np.array(cond["weights"][str(selected_ratio)]["w"], dtype=np.float32)
    b = float(cond["weights"][str(selected_ratio)]["b"])
    return (Xz @ w + b).astype(np.float32)


def apply_b2(confidence, policy_params, selected_ratio):
    threshold = policy_params["B2"]["thresholds"][str(selected_ratio)]
    return confidence < threshold


# Loading
def load_frozen_params():
    ood_params = dict(np.load(CACHE / "ood_params.npz"))
    conformal_params = json.load(open(CACHE / "conformal_params.json"))
    policy_params = json.load(open(CACHE / "policy_params.json"))
    return ood_params, conformal_params, policy_params


def regression_test(ood_params):
    cos, maha = score_detectors("test", ood_params)
    rows = []
    all_ok = True
    for name, arr, (ref_mean, ref_std) in [
        ("cosine", cos, REGRESSION_REFERENCE["cosine"]),
        ("maha_combined", maha, REGRESSION_REFERENCE["maha_combined"]),
    ]:
        mean, std = float(arr.mean()), float(arr.std())
        ok = (abs(mean - ref_mean) < REGRESSION_TOLERANCE
              and abs(std - ref_std) < REGRESSION_TOLERANCE)
        all_ok &= ok
        rows.append({"signal": name, "computed_mean": mean, "computed_std": std,
                     "reference_mean": ref_mean, "reference_std": ref_std,
                     "pass": ok})
    return pd.DataFrame(rows), all_ok, cos, maha


# Per-dataset slice table
def build_slice_table(prefix, dataset_name, index_df, ood_params,
                      conformal_params, policy_params, selected_ratio,
                      precomputed=None):
    probs = np.load(CACHE / f"{prefix}_probs.npy")
    mc_probs = np.load(CACHE / f"{prefix}_mc_probs.npy")

    if precomputed is not None:
        cos, maha = precomputed
    else:
        cos, maha = score_detectors(prefix, ood_params)

    mi = mutual_information(mc_probs)
    cp_setsize = compute_cp_setsize(probs, conformal_params["q_mondrian"]["0.02"])

    feature_dict = {"ood_cosine": cos, "ood_maha_combined": maha,
                    "mutual_info": mi.astype(np.float32),
                    "cp_setsize": cp_setsize.astype(np.float32)}

    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    df = index_df.copy().reset_index(drop=True)
    df["dataset"] = dataset_name
    df["predicted_class"] = predicted
    df["predicted_class_name"] = [CLASSES[p] for p in predicted]
    df["softmax_confidence"] = confidence
    df["mutual_info"] = mi
    df["ood_cosine"] = cos
    df["ood_maha_combined"] = maha
    df["cp_setsize"] = cp_setsize

    for condition in [2, 3, 4, 5]:
        s = apply_policy(condition, feature_dict, policy_params, selected_ratio)
        df[f"s_condition{condition}"] = s
        df[f"defer_condition{condition}"] = s > 0
    df["defer_B2"] = apply_b2(confidence, policy_params, selected_ratio)
    df["accept_condition1"] = True

    return df


def build_test_index():
    df = read_csv(SPLITS / "test.csv")
    return pd.DataFrame({
        "path": df["filepath"], "volume_id": None, "slice_index": None,
        "vendor": "Kermany", "set": "test",
        "true_label": df["label"],
        "true_label_name": df["class_name"],
        "known_diseased": df["class_name"] != "NORMAL",
    })


def build_retouch_index():
    idx = read_csv(CACHE / "retouch_linear_fixed_index.csv")
    return pd.DataFrame({
        "path": idx["volume_path"], "volume_id": idx["volume_id"],
        "slice_index": idx["slice_index"], "vendor": idx["vendor"],
        "set": idx["set"], "true_label": -1, "true_label_name": None,
        "known_diseased": idx["known_diseased"],
    })


def build_oct5k_index():
    idx = read_csv(CACHE / "oct5k_index.csv")
    return pd.DataFrame({
        "path": idx["volume_path"], "volume_id": idx["volume_id"],
        "slice_index": idx["slice_index"], "vendor": "Spectralis",
        "set": "n/a", "true_label": idx["label"],
        "true_label_name": [CLASSES[l] for l in idx["label"]],
        "known_diseased": idx["known_diseased"],
    })


def main():
    TABDIR.mkdir(parents=True, exist_ok=True)

    print("Loading frozen parameters (ood_params.npz, conformal_params.json, "
          "policy_params.json)...")
    ood_params, conformal_params, policy_params = load_frozen_params()
    selected_ratio = policy_params["selected_cost_ratio"]
    print(f"  selected_cost_ratio = {selected_ratio}")

    print("\n" + "=" * 90)
    print("REGRESSION TEST: re-applying frozen detectors to the test split's "
          "own cached features")
    print("=" * 90)
    reg_df, reg_ok, test_cos, test_maha = regression_test(ood_params)
    reg_df.to_csv(TABDIR / "08b_regression_test.csv", index=False)
    print(reg_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not reg_ok:
        print("\n[STOP] Regression test failed -- recomputed test-split "
              "detector statistics do not match the recorded step-05 "
              "values. This means the frozen parameters were not correctly "
              "applied by this script's detector-application code. Nothing "
              "downstream (RETOUCH, OCT5k, 08c) can be trusted until this is "
              "fixed. Not proceeding.")
        return
    print("\n  PASS -- frozen detectors correctly reproduce the recorded "
          "test-split statistics. Proceeding to cross-device scoring.")

    print("\nBuilding per-slice tables...")
    test_index = build_test_index()
    test_df = build_slice_table("test", "Kermany-test", test_index, ood_params,
                                conformal_params, policy_params, selected_ratio,
                                precomputed=(test_cos, test_maha))
    print(f"  Kermany-test: {len(test_df)} rows")

    retouch_index = build_retouch_index()
    retouch_df = build_slice_table("retouch_linear_fixed", "RETOUCH", retouch_index,
                                   ood_params, conformal_params, policy_params,
                                   selected_ratio)
    print(f"  RETOUCH: {len(retouch_df)} rows")

    oct5k_index = build_oct5k_index()
    oct5k_df = build_slice_table("oct5k", "OCT5k", oct5k_index, ood_params,
                                 conformal_params, policy_params, selected_ratio)
    print(f"  OCT5k: {len(oct5k_df)} rows")

    all_df = pd.concat([test_df, retouch_df, oct5k_df], ignore_index=True)
    all_df.to_csv(TABDIR / "08b_slice_scores.csv", index=False)
    print(f"\nWrote {len(all_df)} total rows to "
          f"{(TABDIR / '08b_slice_scores.csv').relative_to(PROJECT)}")
    print("Columns:", list(all_df.columns))

    print("\n" + "=" * 90)
    print("Deferral rate by dataset/vendor at the natural threshold "
          f"(cost ratio {selected_ratio}:1):")
    for condition in [2, 3, 4, 5]:
        print(f"\n  Condition {condition}:")
        for (dataset, vendor), sub in all_df.groupby(["dataset", "vendor"]):
            rate = sub[f"defer_condition{condition}"].mean()
            print(f"    {dataset:<14} {vendor:<12} n={len(sub):>5}  "
                  f"defer_rate={rate:.4f}")

    print("\n08b complete. Next: 08c_evaluate.py.")


if __name__ == "__main__":
    main()
