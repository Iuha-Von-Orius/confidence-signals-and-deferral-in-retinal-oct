"""Create stratified Kermany split indices with seed 42.

The supplied splits are fixed; do not regenerate them for the reported experiment.
"""

from paths import PROJECT, DATASET

import numpy as np
import pandas as pd

# Configuration
DATA_ROOT = (DATASET / "Paul_Mooney/OCT2017")
OUT_DIR = PROJECT / "splits"

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

SEED = 42
N_CALIB = 8000
N_VAL = 5000
IMG_EXT = {".jpeg", ".jpg", ".png"}


def collect_files(split_name):
    """Collect all image paths and labels under one split into a DataFrame."""
    rows = []
    for cls in CLASSES:
        cls_dir = DATA_ROOT / split_name / cls
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                rows.append({
                    "filepath": p.relative_to(DATASET).as_posix(),
                    "label": LABEL_MAP[cls],
                    "class_name": cls,
                })
    return pd.DataFrame(rows)


def stratified_take(df, n_total, rng):
    """Draw n_total rows preserving class proportions. Returns (taken, remainder)."""
    taken_idx = []
    for cls in CLASSES:
        cls_idx = df.index[df["class_name"] == cls].to_numpy()
        n_cls = int(round(n_total * len(cls_idx) / len(df)))
        n_cls = min(n_cls, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_cls, replace=False)
        taken_idx.extend(chosen)

    taken_idx = np.array(taken_idx)
    taken = df.loc[taken_idx].copy()
    rest = df.drop(index=taken_idx).copy()
    return taken, rest


def report(name, df):
    """Print the class distribution of one split."""
    print(f"[{name}]  {len(df):,} images")
    for cls in CLASSES:
        n = (df["class_name"] == cls).sum()
        pct = n / len(df) * 100
        print(f"  {cls:<8} {n:>7,}  ({pct:5.1f}%)")
    print()


def main():
    if any((OUT_DIR / f"{name}.csv").exists()
           for name in ("train", "val", "calibration", "test")):
        raise FileExistsError("Existing split CSVs are fixed and will not be overwritten.")
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the full official train set
    pool = collect_files("train")
    n_original = len(pool)
    print(f"Official train set: {n_original:,} images\n")
    report("original distribution", pool)

    # 2. Carve out calibration first — it must never be seen during training
    calib, pool = stratified_take(pool, N_CALIB, rng)

    # 3. Then carve out validation
    val, train = stratified_take(pool, N_VAL, rng)

    # 4. Report
    report("train", train)
    report("validation", val)
    report("calibration", calib)

    # 5. Index the official test set as well
    test = collect_files("test")
    report("test (official)", test)

    # 6. Write CSVs
    for name, df in [("train", train), ("val", val),
                     ("calibration", calib), ("test", test)]:
        out = OUT_DIR / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {out.name}  ({len(df):,} rows)")

    # 7. Integrity checks
    print("\n--- Integrity checks ---")
    sets = {"train": set(train.filepath),
            "val": set(val.filepath),
            "calibration": set(calib.filepath)}
    ok = True
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = sets[a] & sets[b]
            if overlap:
                print(f"  [ERROR] {a} and {b} overlap by {len(overlap)} images")
                ok = False
    if ok:
        print("  No overlap between the three splits")

    total = len(train) + len(val) + len(calib)
    print(f"  Total {total:,} / original {n_original:,} -> "
          f"{'consistent' if total == n_original else 'INCONSISTENT'}")

    print(f"\nSEED = {SEED}")


if __name__ == "__main__":
    main()
