"""Extract NEH and OCTDL features using the preprocessing and model from step 08a.

NEH uses per-image Label values after case-insensitive path deduplication.
OCTDL is divided into in-label, ambiguous and novel-class tiers.
No detector or policy is refitted on these datasets.
"""

from paths import PROJECT, DATASET, read_csv

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TABDIR = PROJECT / "results" / "tables"

NEH_ROOT = (DATASET / "Tehran_Labeled Retinal Optical Coherence Tomography "
                      "Dataset for Classification of Normal, Drusen, and "
                      "CNV Cases")
NEH_CSV = NEH_ROOT / "data_information.csv"
NEH_IMG_ROOT = NEH_ROOT / "NEH_UT_2021RetinalOCTDataset"

OCTDL_ROOT = (DATASET / "OCTDL Optical Coherence Tomography Dataset for "
                        "Image-Based Deep Learning Methods")
OCTDL_CSV = OCTDL_ROOT / "OCTDL_labels.csv"
OCTDL_IMG_ROOT = OCTDL_ROOT / "OCTDL"

# import 08a (digit-prefixed filename, so importlib not `import`)
sys.path.insert(0, str(Path(__file__).resolve().parent))
_ood08a = importlib.import_module("08a_extract_ood_features")

CLASSES = _ood08a.CLASSES
IMG_SIZE = _ood08a.IMG_SIZE
MC_PASSES = _ood08a.MC_PASSES
IMAGENET_MEAN = _ood08a.IMAGENET_MEAN
IMAGENET_STD = _ood08a.IMAGENET_STD
DEVICE = _ood08a.DEVICE
BATCH_SIZE = _ood08a.BATCH_SIZE
CACHE = _ood08a.CACHE
build_backbone_transform = _ood08a.build_backbone_transform
ArrayDataset = _ood08a.ArrayDataset
load_backbone = _ood08a.load_backbone
build_feature_extractor = _ood08a.build_feature_extractor
enable_mc_dropout = _ood08a.enable_mc_dropout
run_full_extraction = _ood08a.run_full_extraction

NEH_PUBLISHED = {"CNV": 5238, "DRUSEN": 5869, "NORMAL": 5696}
NEH_TOTAL_PUBLISHED = 16803
OCTDL_GROUP_PUBLISHED = {"in-label": 1462, "ambiguous": 248, "novel-class": 354}

MANIFEST_COLUMNS = [
    "dataset", "path", "cache_index", "true_label", "true_label_name", "label_group",
    "global_patient_id", "Class", "Label", "Eye", "B-scan", "ext", "Directory",
    "patient_id", "disease", "subcategory", "condition", "eye", "sex", "year",
    "file_name", "format",
]


# Loading
def load_neh():
    """Deduplicate case-insensitive Directory values, keeping the first row in CSV order."""
    df = read_csv(NEH_CSV)
    n_before = len(df)
    df["dir_norm"] = df["Directory"].str.lower()
    df = df.drop_duplicates(subset="dir_norm", keep="first").reset_index(drop=True)
    n_after = len(df)
    df["global_patient_id"] = (df["Class"] + "_"
                                + df["Patient ID"].astype(int).astype(str).str.zfill(3))
    df["ext"] = df["Directory"].str.rsplit(".", n=1).str[-1].str.lower()
    df["path"] = df["Directory"].apply(lambda d: NEH_IMG_ROOT / d)
    return df, n_before, n_after


def load_octdl():
    df = read_csv(OCTDL_CSV)
    df["path"] = df.apply(
        lambda r: OCTDL_IMG_ROOT / r["disease"] / f"{r['file_name']}.jpg", axis=1)
    return df


def octdl_label(disease, condition):
    if disease == "DME":
        return CLASSES.index("DME"), "DME", "in-label"
    if disease == "NO":
        return CLASSES.index("NORMAL"), "NORMAL", "in-label"
    if disease == "AMD" and condition == "drusen":
        return CLASSES.index("DRUSEN"), "DRUSEN", "in-label"
    if disease == "AMD" and condition == "MNV":
        return CLASSES.index("CNV"), "CNV", "in-label"
    if disease == "AMD" and condition == "MNV_suspected":
        return -2, "AMD-MNV_suspected", "ambiguous"
    if disease in ("ERM", "RAO", "RVO", "VID"):
        return -1, disease, "novel-class"
    raise ValueError(f"unmapped OCTDL row: disease={disease!r} condition={condition!r}")


# Two-stage read + record build
def build_neh_records(neh_df):
    """Load NEH greyscale arrays and per-image Label values with aligned index rows."""
    arrays, index_rows = [], []
    n_fail = 0
    for _, row in neh_df.iterrows():
        try:
            with Image.open(row["path"]) as im:
                arr = np.asarray(im.convert("L"))
        except Exception as e:
            n_fail += 1
            print(f"  [READ FAIL] NEH {row['path']}: {e}")
            continue
        true_label = CLASSES.index(row["Label"])
        arrays.append(arr)
        index_rows.append({
            "dataset": "NEH", "path": str(row["path"]), "cache_index": len(arrays) - 1,
            "true_label": true_label, "true_label_name": CLASSES[true_label],
            "label_group": "in-label",
            "global_patient_id": row["global_patient_id"], "Class": row["Class"],
            "Label": row["Label"], "Eye": row["Eye"], "B-scan": row["B-scan"],
            "ext": row["ext"], "Directory": row["Directory"],
        })
    return arrays, index_rows, n_fail


def build_octdl_records(octdl_df):
    """Load OCTDL greyscale arrays, mapped labels, tiers and detected formats."""
    arrays, index_rows = [], []
    n_fail = 0
    for _, row in octdl_df.iterrows():
        try:
            with Image.open(row["path"]) as im:
                fmt = im.format
                arr = np.asarray(im.convert("L"))
        except Exception as e:
            n_fail += 1
            print(f"  [READ FAIL] OCTDL {row['path']}: {e}")
            continue
        true_label, true_label_name, label_group = octdl_label(row["disease"], row["condition"])
        arrays.append(arr)
        index_rows.append({
            "dataset": "OCTDL", "path": str(row["path"]), "cache_index": len(arrays) - 1,
            "true_label": true_label, "true_label_name": true_label_name,
            "label_group": label_group,
            "patient_id": row["patient_id"], "disease": row["disease"],
            "subcategory": row["subcategory"], "condition": row["condition"],
            "eye": row["eye"], "sex": row["sex"], "year": row["year"],
            "file_name": row["file_name"], "format": fmt,
        })
    return arrays, index_rows, n_fail


def main():
    t0 = time.time()
    report = []

    report.append("=" * 78)
    report.append("13a_extract_new_ood_features.py -- NEH + OCTDL feature extraction")
    report.append("=" * 78)
    report.append("")
    print("Loading NEH (Directory column, deduplicated per "
          "12_data_integrity_checks.py's rule)...")
    neh_df, neh_n_before, neh_n_after = load_neh()
    neh_label_unique = sorted(neh_df["Label"].unique().tolist())
    print(f"  Label.unique() = {neh_label_unique}")
    assert set(neh_label_unique) <= set(CLASSES), (
        f"NEH Label contains values outside CLASSES: {neh_label_unique}")
    neh_counts = neh_df["Class"].value_counts().to_dict()

    report.append("")
    report.append("-" * 78)
    report.append("NEH")
    report.append("-" * 78)
    report.append(f"Rows before dedup: {neh_n_before}   after dedup: {neh_n_after}")
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        n = int(neh_counts.get(c, 0))
        pub = NEH_PUBLISHED[c]
        report.append(f"  {c:<8s} n={n:<6d} expected={pub:<6d} match={n == pub}")
    report.append(f"  Total    n={neh_n_after:<6d} expected={NEH_TOTAL_PUBLISHED:<6d} "
                  f"match={neh_n_after == NEH_TOTAL_PUBLISHED}")
    report.append(f"Label.unique() = {neh_label_unique}")
    report.append("Ground truth = Label column (per-B-scan expert read), not Class "
                  "(patient folder). No filtering to Label==Class.")

    print("Loading OCTDL...")
    octdl_df = load_octdl()
    octdl_group_counts = {}
    for disease, condition in zip(octdl_df["disease"], octdl_df["condition"]):
        _, _, group = octdl_label(disease, condition)
        octdl_group_counts[group] = octdl_group_counts.get(group, 0) + 1

    report.append("")
    report.append("-" * 78)
    report.append("OCTDL")
    report.append("-" * 78)
    for g in ["in-label", "ambiguous", "novel-class"]:
        n = octdl_group_counts.get(g, 0)
        pub = OCTDL_GROUP_PUBLISHED[g]
        report.append(f"  {g:<12s} n={n:<6d} expected={pub:<6d} match={n == pub}")
    report.append(f"  Total        n={sum(octdl_group_counts.values()):<6d} "
                  f"expected={len(octdl_df):<6d}")

    print("\nLoading backbone (checkpoints/best.pt)...")
    model = load_backbone()
    feat_model = build_feature_extractor()
    transform = build_backbone_transform()

    print(f"\nBuilding NEH arrays/records ({neh_n_after} images, two-stage "
          "read: Image.open -> convert(\"L\"))...")
    neh_arrays, neh_index_rows, neh_n_fail = build_neh_records(neh_df)
    print(f"  {len(neh_arrays)} read, {neh_n_fail} failed")

    print(f"\nExtracting NEH features ({len(neh_arrays)} images, "
          f"{MC_PASSES} MC passes)...")
    run_full_extraction("neh", neh_arrays, neh_index_rows, model, feat_model,
                        transform, tag_with_method=False)

    print(f"\nBuilding OCTDL arrays/records ({len(octdl_df)} images, "
          "two-stage read)...")
    octdl_arrays, octdl_index_rows, octdl_n_fail = build_octdl_records(octdl_df)
    print(f"  {len(octdl_arrays)} read, {octdl_n_fail} failed")

    print(f"\nExtracting OCTDL features ({len(octdl_arrays)} images, "
          f"{MC_PASSES} MC passes)...")
    run_full_extraction("octdl", octdl_arrays, octdl_index_rows, model, feat_model,
                        transform, tag_with_method=False)

    manifest_df = pd.DataFrame(neh_index_rows + octdl_index_rows, columns=MANIFEST_COLUMNS)
    manifest_path = TABDIR / "13a_new_ood_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path.relative_to(PROJECT)} ({len(manifest_df)} rows)")

    elapsed = time.time() - t0
    report.append("")
    report.append("-" * 78)
    report.append("Read failures / runtime")
    report.append("-" * 78)
    report.append(f"NEH read failures: {neh_n_fail} of {neh_n_after}")
    report.append(f"OCTDL read failures: {octdl_n_fail} of {len(octdl_df)}")
    report.append(f"Wall-clock runtime: {elapsed:.1f} s ({elapsed / 60:.1f} min)")

    report.append("")
    report.append("-" * 78)
    report.append("Pre-registered predictions (recorded here, NOT evaluated by this "
                  "script -- see module docstring for full text)")
    report.append("-" * 78)
    report.append("1. OCTDL novel-class (ERM/RAO/RVO/VID) Mahalanobis distance should "
                  "exceed OCTDL in-label (DME/NO/AMD) Mahalanobis distance.")
    report.append("2. OCTDL's overall Mahalanobis distance should exceed what its "
                  "aspect ratio alone (882x322 vs Kermany 512x496) would predict, "
                  "given its p1 range 7.2-16.3 sits on RETOUCH-Topcon's black-field-"
                  "floor axis.")
    report.append("3. NEH's Mahalanobis distance should sit well below RETOUCH-Topcon "
                  "(mean -23.06) and close to RETOUCH-Spectralis (-0.55) -- both NEH "
                  "and Kermany are Heidelberg Spectralis.")
    report.append("4. NEH DRUSEN accuracy should be materially higher than OCT5k/"
                  "Rasti's DRUSEN accuracy (45.1%, 195/432) -- same hospital, same "
                  "device, same disease category, but NEH's labels are per-B-scan "
                  "expert reads while Rasti's are inherited from a volume-level "
                  "diagnosis.")

    report_path = TABDIR / "13a_new_ood_extraction.txt"
    report_path.write_text("\n".join(report) + "\n")
    print(f"Wrote {report_path.relative_to(PROJECT)}")
    print(f"\nDone in {elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
