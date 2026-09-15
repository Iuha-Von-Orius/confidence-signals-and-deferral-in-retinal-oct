"""Fit cosine and six-depth Mahalanobis detectors on training features.

Both scores increase towards the training distribution. Mahalanobis scores
use a shared covariance with shrinkage, then training-set standardisation
and an equal-weight mean across depths. Parameters are frozen for evaluation.
"""

from paths import PROJECT

import json
import time

import numpy as np
import pandas as pd

CACHE = PROJECT / "cache"
RESULTS = PROJECT / "results"
TABDIR = RESULTS / "tables"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)
N_STAGES = 6

FIT_SPLIT = "train"
SCORE_SPLITS = ["train", "val", "calibration", "test"]

COSINE_STAGE = 6

MAHA_STAGES = list(range(1, N_STAGES + 1))

SHRINKAGE = 1e-3


# Fitting
def fit_class_centroids(features, labels):
    """Mean feature vector per class. Shape (n_classes, dim)."""
    return np.stack([features[labels == k].mean(axis=0) for k in range(N_CLASSES)])


def fit_precision(features, centroids, labels):
    """Estimate shared within-class covariance with trace-scaled shrinkage.

    Return its precision matrix via a Cholesky factor and the condition number.
    """
    # Centre each sample on its own class mean, then pool.
    centred = features - centroids[labels]
    cov = (centred.T @ centred) / len(centred)

    # Shrinkage scaled to the data: lambda * mean(diag) keeps the regularisation
    # proportionate whether the features are 16-d or 1280-d.
    cov += np.eye(cov.shape[0], dtype=cov.dtype) * SHRINKAGE * np.trace(cov) / cov.shape[0]

    # Cholesky gives Sigma = L L^T; the precision follows from inverting L, which
    # is triangular and therefore stable.
    L = np.linalg.cholesky(cov.astype(np.float64))
    L_inv = np.linalg.inv(L)
    precision = L_inv.T @ L_inv

    # Report conditioning so a silently degenerate matrix cannot pass unnoticed.
    cond = np.linalg.cond(cov)
    return precision.astype(np.float64), float(cond)


# Scoring
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
            # einsum computes the quadratic form row-wise without materialising
            # the full (batch, dim) intermediate product.

            d_k[start:start + batch] = ((chunk @ precision) * chunk).sum(axis=1)
        best = np.minimum(best, d_k)

    return -best


# Main
def main():
    TABDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load the fitting split
    fit_labels = np.load(CACHE / f"{FIT_SPLIT}_labels.npy")
    print(f"Fitting on '{FIT_SPLIT}': {len(fit_labels):,} images\n")

    params = {}
    fit_stats = []

    print(f"{'stage':>6} {'dim':>6} {'cond(Sigma)':>14} {'fit s':>7}")
    print("-" * 38)

    for stage in range(1, N_STAGES + 1):
        ts = time.time()
        feats = np.load(CACHE / f"{FIT_SPLIT}_feat_stage{stage}.npy")

        centroids = fit_class_centroids(feats, fit_labels)
        precision, cond = fit_precision(feats, centroids, fit_labels)

        params[f"centroids_{stage}"] = centroids
        params[f"precision_{stage}"] = precision

        print(f"{stage:>6} {feats.shape[1]:>6} {cond:>14.3e} {time.time()-ts:>7.1f}")
        fit_stats.append({"stage": stage, "dim": feats.shape[1],
                          "condition_number": cond})

        # A condition number this large means the inverse is dominated by noise
        # in the smallest eigendirections; shrinkage would need raising.
        if cond > 1e12:
            print(f"       [WARNING] stage {stage} covariance is near-singular")

    print("\nComputing z-score statistics on the fitting split...")
    z_stats = {}
    for stage in MAHA_STAGES:
        feats = np.load(CACHE / f"{FIT_SPLIT}_feat_stage{stage}.npy")
        raw = mahalanobis_score(feats, params[f"centroids_{stage}"],
                                params[f"precision_{stage}"])
        z_stats[stage] = (float(raw.mean()), float(raw.std()))
        params[f"z_mean_{stage}"] = np.array([raw.mean()])
        params[f"z_std_{stage}"] = np.array([raw.std()])
        print(f"  stage {stage}: mean {raw.mean():>12.2f}  std {raw.std():>12.2f}")

    np.savez(CACHE / "ood_params.npz", **params)

    # Score every split
    print("\nScoring splits...")
    summary = []

    for split in SCORE_SPLITS:
        labels = np.load(CACHE / f"{split}_labels.npy")

        # Cosine on stage 6 only.
        feats6 = np.load(CACHE / f"{split}_feat_stage{COSINE_STAGE}.npy")
        cos = cosine_score(feats6, params[f"centroids_{COSINE_STAGE}"])
        np.save(CACHE / f"{split}_ood_cosine.npy", cos.astype(np.float32))

        # Mahalanobis at every depth, raw and z-scored.
        z_all = []
        for stage in MAHA_STAGES:
            feats = np.load(CACHE / f"{split}_feat_stage{stage}.npy")
            raw = mahalanobis_score(feats, params[f"centroids_{stage}"],
                                    params[f"precision_{stage}"])
            mu, sd = z_stats[stage]
            z = (raw - mu) / (sd + 1e-12)

            np.save(CACHE / f"{split}_ood_maha_stage{stage}.npy", raw.astype(np.float32))
            np.save(CACHE / f"{split}_ood_maha_z_stage{stage}.npy", z.astype(np.float32))
            z_all.append(z)

        combined = np.mean(z_all, axis=0)
        np.save(CACHE / f"{split}_ood_maha_combined.npy", combined.astype(np.float32))

        summary.append({
            "split": split,
            "n": len(labels),
            "cosine_mean": cos.mean(),
            "cosine_std": cos.std(),
            "maha_combined_mean": combined.mean(),
            "maha_combined_std": combined.std(),
        })
        print(f"  {split:12s} {len(labels):>7,}  "
              f"cosine {cos.mean():.4f}±{cos.std():.4f}  "
              f"maha_z {combined.mean():+.3f}±{combined.std():.3f}")

    # Tables
    pd.DataFrame(fit_stats).to_csv(TABDIR / "05_covariance_conditioning.csv", index=False)
    df = pd.DataFrame(summary)
    df.to_csv(TABDIR / "05_ood_score_summary.csv", index=False)

    # Report
    print("\n" + "=" * 66)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nMahalanobis z-scores are centred on train by construction.")
    for row in summary:
        if row["split"] != FIT_SPLIT:
            print(f"  {row['split']:12s} mean z = {row['maha_combined_mean']:+.3f}")

    print(f"\nParameters: cache/ood_params.npz")
    print(f"Elapsed: {time.time() - t0:.0f} s")
    print("\nAUROC requires OOD samples and is computed at step 08.")


if __name__ == "__main__":
    main()
