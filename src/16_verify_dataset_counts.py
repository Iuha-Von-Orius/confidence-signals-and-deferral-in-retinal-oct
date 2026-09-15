"""Independently verify dataset counts and label mappings against saved indices."""

from paths import PROJECT, read_csv

import importlib
import re
import sys
import time
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

TABDIR = PROJECT / "results" / "tables"
SPLITS = PROJECT / "splits"

sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_00b = importlib.import_module("00b_inspect_ood_data")
mod_11 = importlib.import_module("11_explore_new_data")
mod_02 = importlib.import_module("02_split")

RETOUCH_ROOT = mod_00b.DATASETS["RETOUCH"]
OCT5K_ROOT = mod_00b.DATASETS["OCT5k"]
NEH_CSV = mod_11.NEH_CSV
OCTDL_CSV = mod_11.OCTDL_CSV
KERMANY_ROOT = mod_02.DATA_ROOT
KERMANY_CLASSES = mod_02.CLASSES

VENDORS = ["Cirrus", "Spectralis", "Topcon"]
N_CENTRAL_SLICES = 9

# published figures this script's independent counts are checked against
KERMANY_SPLIT_PUBLISHED = {"train": 83484, "test": 968, "val": 32}
RETOUCH_VENDOR_PUBLISHED = {"Spectralis": 38, "Cirrus": 38, "Topcon": 36}
RETOUCH_BSCANS_PUBLISHED = 11334
OCT5K_CLASS_PUBLISHED = {"AMD": 48, "DME": 50, "Normal": 50}
OCT5K_BSCANS_PUBLISHED = 4142
OCT5K_PER_VOLUME_PUBLISHED = [19, 25, 31, 61]
NEH_CLASS_PUBLISHED = {"CNV": 5238, "DRUSEN": 5869, "NORMAL": 5696}
NEH_LABEL_PUBLISHED = {"CNV": 3234, "DRUSEN": 4992, "NORMAL": 8577}
NEH_PATIENTS_PUBLISHED = {"CNV": 161, "DRUSEN": 160, "NORMAL": 120}
NEH_PATIENTS_TOTAL_PUBLISHED = 441
OCTDL_TIER_PUBLISHED = {"in-label": 1462, "ambiguous": 248, "novel-class": 354}
OCTDL_DISEASE_PUBLISHED = {
    "AMD": 1231, "DME": 147, "ERM": 155, "NO": 332, "RAO": 22, "RVO": 101, "VID": 76,
}
OCTDL_AMD_CONDITION_PUBLISHED = {"drusen": 266, "MNV": 717, "MNV_suspected": 248}

TIFF_RE = re.compile(r"^Image\s*(\d+)\.TIFF$", re.IGNORECASE)
PNG_RE = re.compile(r"^Image\s*(\d+)\.PNG$", re.IGNORECASE)


def octdl_label(disease, condition):
    """Map OCTDL disease and subtype to class and tier; use -2 for ambiguous, -1 for novel."""
    if disease == "DME":
        return KERMANY_CLASSES.index("DME"), "DME", "in-label"
    if disease == "NO":
        return KERMANY_CLASSES.index("NORMAL"), "NORMAL", "in-label"
    if disease == "AMD" and condition == "drusen":
        return KERMANY_CLASSES.index("DRUSEN"), "DRUSEN", "in-label"
    if disease == "AMD" and condition == "MNV":
        return KERMANY_CLASSES.index("CNV"), "CNV", "in-label"
    if disease == "AMD" and condition == "MNV_suspected":
        return -2, "AMD-MNV_suspected", "ambiguous"
    if disease in ("ERM", "RAO", "RVO", "VID"):
        return -1, disease, "novel-class"
    raise ValueError(f"unmapped OCTDL row: disease={disease!r} condition={condition!r}")


# Section 1: independent counts

def count_kermany():
    """Count JPEG images in each Kermany split and class directory."""
    rows = []
    for split in ["train", "val", "test"]:
        for cls in KERMANY_CLASSES:
            n = len(list((KERMANY_ROOT / split / cls).glob("*.jpeg")))
            rows.append({"split": split, "class": cls, "n": n})
    return pd.DataFrame(rows)


def count_retouch_volumes():
    """Count oct.mhd volumes by scanner vendor, excluding segmentation headers."""
    rows = []
    for p in sorted(RETOUCH_ROOT.rglob("oct.mhd")):
        vendor = next((v for v in VENDORS if v in str(p)), None)
        if vendor is None:
            raise ValueError(f"cannot determine RETOUCH vendor for {p}")
        rows.append({"path": p, "vendor": vendor})
    return pd.DataFrame(rows)


def retouch_slice_count(mhd_path):
    """z-extent from the MetaImage header -- GetSize() returns (x, y, z)."""
    return sitk.ReadImage(str(mhd_path)).GetSize()[2]


def count_oct5k_volumes():
    """Volume directories named '{class} (n).E2E' directly under OCT5K_ROOT."""
    rows = []
    for d in sorted(OCT5K_ROOT.rglob("*.E2E")):
        if not d.is_dir():
            continue
        rows.append({"path": d, "class": d.parent.name})
    return pd.DataFrame(rows)


def oct5k_slice_count(vol_dir):
    """Count numbered RASTI images, including Image 0, with TIFF-first PNG fallback."""
    tiffs = [p for p in vol_dir.rglob("*") if TIFF_RE.match(p.name)]
    if tiffs:
        return len(tiffs), False
    pngs = [p for p in vol_dir.rglob("*") if PNG_RE.match(p.name)]
    return len(pngs), True


def load_neh():
    """Deduplicate case-insensitive Directory values, keeping the first row in CSV order."""
    df = read_csv(NEH_CSV)
    n_before = len(df)
    df["dir_norm"] = df["Directory"].str.lower()
    df = df.drop_duplicates(subset="dir_norm", keep="first").reset_index(drop=True)
    n_after = len(df)
    df["global_patient_id"] = (
        df["Class"] + "_" + df["Patient ID"].astype(int).astype(str).str.zfill(3)
    )
    return df, n_before, n_after


def load_octdl():
    df = read_csv(OCTDL_CSV)
    mapped = df.apply(lambda r: octdl_label(r["disease"], r["condition"]), axis=1)
    df["true_label"] = [m[0] for m in mapped]
    df["true_label_name"] = [m[1] for m in mapped]
    df["label_group"] = [m[2] for m in mapped]
    return df


def fmt_dist(counter):
    return ", ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def main():
    t0 = time.time()
    report = []
    mismatches_found = False

    report.append("=" * 78)
    report.append("16_verify_dataset_counts.py -- independent recount of Table 5.1")
    report.append("=" * 78)
    report.append("")
    report.append("Every count below is computed directly from the raw datasets on disk.")
    report.append("Pipeline output tables are read only in Section 5, as the thing being")
    report.append("checked -- never as a source for the counts themselves.")

    # Section 1: independent counts
    report.append("")
    report.append("-" * 78)
    report.append("SECTION 1: independent counts")
    report.append("-" * 78)

    print("Counting Kermany...")
    kermany_df = count_kermany()
    report.append("")
    report.append("Kermany (Dataset/Paul_Mooney/OCT2017)")
    pivot = kermany_df.pivot(index="split", columns="class", values="n")
    for split in ["train", "val", "test"]:
        row = pivot.loc[split]
        total = int(row.sum())
        pub = KERMANY_SPLIT_PUBLISHED[split]
        report.append(f"  {split:<6s} " + " ".join(f"{c}={int(row[c])}" for c in KERMANY_CLASSES)
                      + f"  total={total}  expected={pub}  match={total == pub}")
    kermany_total = int(kermany_df["n"].sum())
    kermany_train = int(pivot.loc["train"].sum())
    kermany_val = int(pivot.loc["val"].sum())
    kermany_test = int(pivot.loc["test"].sum())
    report.append(f"  Grand total: {kermany_train} (train) + {kermany_test} (test) + "
                  f"{kermany_val} (val) = {kermany_train + kermany_test + kermany_val}"
                  f"  (expected 84,484)")

    print("Counting RETOUCH volumes...")
    retouch_vol_df = count_retouch_volumes()
    retouch_vendor_counts = retouch_vol_df["vendor"].value_counts().to_dict()
    report.append("")
    report.append("RETOUCH (Dataset/RETOUCH) -- volumes (oct.mhd)")
    for v in VENDORS:
        n = retouch_vendor_counts.get(v, 0)
        pub = RETOUCH_VENDOR_PUBLISHED[v]
        report.append(f"  {v:<11s} n={n:<4d} expected={pub:<4d} match={n == pub}")
    retouch_total_volumes = len(retouch_vol_df)
    report.append(f"  Total       n={retouch_total_volumes:<4d} expected=112  "
                  f"match={retouch_total_volumes == 112}")

    print("Counting OCT5k/Rasti volumes...")
    oct5k_vol_df = count_oct5k_volumes()
    oct5k_class_counts = oct5k_vol_df["class"].value_counts().to_dict()
    report.append("")
    report.append("OCT5k/Rasti (Dataset/Macular-Dataset-R.Rasti_old) -- volumes (*.E2E)")
    for c in ["AMD", "DME", "Normal"]:
        n = oct5k_class_counts.get(c, 0)
        pub = OCT5K_CLASS_PUBLISHED[c]
        report.append(f"  {c:<8s} n={n:<4d} expected={pub:<4d} match={n == pub}")
    oct5k_total_volumes = len(oct5k_vol_df)
    report.append(f"  Total    n={oct5k_total_volumes:<4d} expected=148  "
                  f"match={oct5k_total_volumes == 148}")

    print("Loading NEH (raw CSV, applying dedup rule)...")
    neh_df, neh_n_before, neh_n_after = load_neh()
    neh_class_counts = neh_df["Class"].value_counts().to_dict()
    neh_label_counts = neh_df["Label"].value_counts().to_dict()
    neh_n_patients = neh_df["global_patient_id"].nunique()
    report.append("")
    report.append("NEH (data_information.csv)")
    report.append(f"  Rows before dedup: {neh_n_before}   after dedup: {neh_n_after}")
    report.append(f"  Unique global_patient_id (post-dedup): {neh_n_patients}")
    report.append("  Class (folder) distribution, post-dedup:")
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        n = neh_class_counts.get(c, 0)
        pub = NEH_CLASS_PUBLISHED[c]
        report.append(f"    {c:<8s} n={n:<6d} expected={pub:<6d} match={n == pub}")
    report.append("  Label (ground truth) distribution, post-dedup -- a DIFFERENT "
                  "grouping of the same rows:")
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        n = neh_label_counts.get(c, 0)
        pub = NEH_LABEL_PUBLISHED[c]
        report.append(f"    {c:<8s} n={n:<6d} expected={pub:<6d} match={n == pub}")

    print("Loading OCTDL (raw CSV, applying independent three-tier mapping)...")
    octdl_df = load_octdl()
    octdl_tier_counts = octdl_df["label_group"].value_counts().to_dict()
    octdl_tier_patients = octdl_df.groupby("label_group")["patient_id"].nunique().to_dict()
    octdl_n_patients_overall = octdl_df["patient_id"].nunique()
    report.append("")
    report.append("OCTDL (OCTDL_labels.csv), independent three-tier mapping")
    for g in ["in-label", "ambiguous", "novel-class"]:
        n = octdl_tier_counts.get(g, 0)
        pub = OCTDL_TIER_PUBLISHED[g]
        np_ = octdl_tier_patients.get(g, 0)
        report.append(f"  {g:<12s} n={n:<6d} expected={pub:<6d} match={n == pub}  "
                      f"unique patient_id={np_}")
    report.append(f"  Total        n={sum(octdl_tier_counts.values()):<6d} expected=2064")
    report.append(f"  Unique patient_id, all tiers combined: {octdl_n_patients_overall}")

    # Section 2: volume slice-count audit
    report.append("")
    report.append("-" * 78)
    report.append("SECTION 2: volume slice-count audit (precondition for the x9 check, "
                  "plus a published-total comparison)")
    report.append("-" * 78)

    print("Reading RETOUCH volume slice counts (SimpleITK header)...")
    retouch_counts = [retouch_slice_count(p) for p in retouch_vol_df["path"]]
    retouch_dist = pd.Series(retouch_counts).value_counts().sort_index().to_dict()
    retouch_min, retouch_max = min(retouch_counts), max(retouch_counts)
    assert retouch_min >= N_CENTRAL_SLICES, (
        f"RETOUCH: minimum readable slice count {retouch_min} < {N_CENTRAL_SLICES} -- "
        f"the central-slice sampling rule this project relies on would not be valid"
    )
    report.append("")
    report.append("RETOUCH -- per-volume slice count distribution")
    report.append(f"  min={retouch_min}  max={retouch_max}  "
                  f"distinct values (value:n_volumes): {fmt_dist(retouch_dist)}")
    report.append(f"  assert min >= {N_CENTRAL_SLICES}: PASS")
    retouch_bscans_total = sum(retouch_counts)
    report.append(f"  Sum of all volumes' slice counts: {retouch_bscans_total}")
    report.append(f"  Published total (Bogunovic et al., IEEE TMI 2019, pp.3, 5): "
                  f"{RETOUCH_BSCANS_PUBLISHED}")
    report.append(f"  Difference: {RETOUCH_BSCANS_PUBLISHED - retouch_bscans_total} "
                  f"-- printed for the record, not interpreted or reconciled.")

    print("Reading OCT5k volume slice counts (TIFF/PNG file count)...")
    oct5k_counts, oct5k_fallback = [], []
    for d in oct5k_vol_df["path"]:
        n, used_png = oct5k_slice_count(d)
        oct5k_counts.append(n)
        if used_png:
            oct5k_fallback.append(d.name)
    oct5k_dist = pd.Series(oct5k_counts).value_counts().sort_index().to_dict()
    oct5k_min, oct5k_max = min(oct5k_counts), max(oct5k_counts)
    assert oct5k_min >= N_CENTRAL_SLICES, (
        f"OCT5k: minimum readable slice count {oct5k_min} < {N_CENTRAL_SLICES} -- "
        f"the central-slice sampling rule this project relies on would not be valid"
    )
    report.append("")
    report.append("OCT5k/Rasti -- per-volume slice count distribution")
    report.append(f"  min={oct5k_min}  max={oct5k_max}  "
                  f"distinct values (value:n_volumes): {fmt_dist(oct5k_dist)}")
    report.append(f"  assert min >= {N_CENTRAL_SLICES}: PASS")
    report.append(f"  Volumes using the PNG fallback (zero TIFF matches): "
                  f"{len(oct5k_fallback)} -- {oct5k_fallback}")
    oct5k_bscans_total = sum(oct5k_counts)
    report.append(f"  Sum of all volumes' slice counts: {oct5k_bscans_total}")
    report.append(f"  Published per-volume counts (Rasti et al., IEEE TMI 2018, "
                  f"37(4):1024-1034, p.1026): {OCT5K_PER_VOLUME_PUBLISHED}, "
                  f"published total: {OCT5K_BSCANS_PUBLISHED}")
    report.append(f"  Difference (total): {OCT5K_BSCANS_PUBLISHED - oct5k_bscans_total} "
                  f"-- printed for the record, not interpreted or reconciled.")

    # The x9 sampling check itself -- compared to the pipeline's actual row
    # count in Section 5, not here.
    retouch_vendor_x9 = {v: retouch_vendor_counts.get(v, 0) * N_CENTRAL_SLICES for v in VENDORS}
    oct5k_volumes_x9 = oct5k_total_volumes * N_CENTRAL_SLICES

    # Section 3: split-integrity check
    report.append("")
    report.append("-" * 78)
    report.append("SECTION 3: splits/ integrity check (the LOCKED train/val/calibration/"
                  "test partition)")
    report.append("-" * 78)

    def load_split(name):
        path = SPLITS / f"{name}.csv"
        df = read_csv(path)
        if "filepath" not in df.columns:
            raise RuntimeError(
                f"{path} has no 'filepath' column; actual columns: {df.columns.tolist()}"
            )
        return df

    print("Reading splits/*.csv...")
    split_dfs = {name: load_split(name) for name in ["train", "val", "calibration", "test"]}
    split_sets = {name: set(df["filepath"]) for name, df in split_dfs.items()}

    pairs = [("train", "val"), ("train", "calibration"), ("val", "calibration")]
    report.append("")
    report.append("Pairwise disjointness")
    all_disjoint = True
    for a, b in pairs:
        overlap = len(split_sets[a] & split_sets[b])
        ok = overlap == 0
        all_disjoint &= ok
        report.append(f"  {a} n {b}: {overlap} shared paths -- {'PASS' if ok else 'FAIL'}")
    test_overlap = len((split_sets["train"] | split_sets["val"] | split_sets["calibration"])
                       & split_sets["test"])
    ok = test_overlap == 0
    all_disjoint &= ok
    report.append(f"  (train u val u calibration) n test: {test_overlap} shared paths -- "
                  f"{'PASS' if ok else 'FAIL'}")
    assert all_disjoint, "splits/*.csv are not pairwise disjoint -- the locked split has changed"

    split_train_sum = len(split_dfs["train"]) + len(split_dfs["val"]) + len(split_dfs["calibration"])
    split_sum_ok = split_train_sum == kermany_train
    report.append("")
    report.append(f"train+val+calibration = {len(split_dfs['train'])} + {len(split_dfs['val'])} "
                  f"+ {len(split_dfs['calibration'])} = {split_train_sum}, vs Section 1's "
                  f"independent Kermany-train count {kermany_train} -- "
                  f"{'PASS' if split_sum_ok else 'FAIL'}")
    split_test_ok = len(split_dfs["test"]) == kermany_test
    report.append(f"test.csv rows = {len(split_dfs['test'])}, vs Section 1's independent "
                  f"Kermany-test count {kermany_test} -- {'PASS' if split_test_ok else 'FAIL'}")
    if not (all_disjoint and split_sum_ok and split_test_ok):
        mismatches_found = True

    # Section 4: comparison against published figures
    report.append("")
    report.append("-" * 78)
    report.append("SECTION 4: comparison against published figures (the one genuinely "
                  "external check -- neither side of this section is pipeline output)")
    report.append("-" * 78)

    octdl_disease_counts = octdl_df["disease"].value_counts().to_dict()
    report.append("")
    report.append("OCTDL disease-level counts vs the OCTDL dataset paper's Table 2 "
                  "(Scientific Data, 2024)")
    published_rows = []
    for disease in sorted(OCTDL_DISEASE_PUBLISHED):
        n = octdl_disease_counts.get(disease, 0)
        pub = OCTDL_DISEASE_PUBLISHED[disease]
        match = n == pub
        published_rows.append(("OCTDL disease " + disease, n, pub, match))
        report.append(f"  {disease:<5s} n={n:<6d} published={pub:<6d} "
                      f"{'MATCH' if match else 'MISMATCH'}")

    amd_rows = octdl_df[octdl_df["disease"] == "AMD"]
    amd_condition_counts = amd_rows["condition"].value_counts().to_dict()
    amd_sum = sum(OCTDL_AMD_CONDITION_PUBLISHED.values())
    amd_sum_actual = sum(amd_condition_counts.get(c, 0) for c in OCTDL_AMD_CONDITION_PUBLISHED)
    report.append("")
    report.append("AMD condition breakdown -- the single most important check of the "
                  "three-tier split's correctness")
    for c in ["drusen", "MNV", "MNV_suspected"]:
        n = amd_condition_counts.get(c, 0)
        pub = OCTDL_AMD_CONDITION_PUBLISHED[c]
        published_rows.append((f"OCTDL AMD condition={c}", n, pub, n == pub))
        report.append(f"  {c:<14s} n={n:<6d} published={pub:<6d} match={n == pub}")
    report.append(f"  Sum: {' + '.join(str(amd_condition_counts.get(c, 0)) for c in ['drusen', 'MNV', 'MNV_suspected'])} "
                  f"= {amd_sum_actual}  (expected {amd_sum})")
    assert amd_sum_actual == amd_sum, "AMD condition counts do not sum to the AMD total"
    amd_condition_set = set(amd_rows["condition"].unique())
    expected_set = {"drusen", "MNV", "MNV_suspected"}
    report.append(f"  AMD condition value set: {sorted(amd_condition_set)}  "
                  f"(expected exactly {sorted(expected_set)})")
    assert amd_condition_set == expected_set, (
        f"AMD rows have condition values outside {expected_set}: "
        f"{amd_condition_set - expected_set} -- the mapping rule does not exhaustively "
        f"cover AMD and some rows are silently falling through"
    )

    neh_patients_by_class = neh_df.groupby("Class")["global_patient_id"].nunique().to_dict()
    report.append("")
    report.append("NEH published patient counts (Sotoudeh-Paima et al., Comput Biol Med "
                  "144 (2022) 105368)")
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        n = neh_patients_by_class.get(c, 0)
        pub = NEH_PATIENTS_PUBLISHED[c]
        published_rows.append((f"NEH patients {c}", n, pub, n == pub))
        report.append(f"  {c:<8s} n={n:<4d} published={pub:<4d} match={n == pub}")
    total_ok = neh_n_patients == NEH_PATIENTS_TOTAL_PUBLISHED
    published_rows.append(("NEH patients total", neh_n_patients, NEH_PATIENTS_TOTAL_PUBLISHED, total_ok))
    report.append(f"  Total    n={neh_n_patients:<4d} published={NEH_PATIENTS_TOTAL_PUBLISHED:<4d} "
                  f"match={total_ok}")
    if any(not row[3] for row in published_rows):
        mismatches_found = True

    # Section 5: comparison against pipeline output
    report.append("")
    report.append("-" * 78)
    report.append("SECTION 5: comparison against pipeline output (13a_new_ood_manifest."
                  "csv, 13b_new_slice_scores.csv, 13c_T01_source_summary.csv -- read ONLY "
                  "here, as the thing being checked)")
    report.append("-" * 78)

    print("Reading pipeline output for the final comparison (Section 5 only)...")
    manifest = read_csv(TABDIR / "13a_new_ood_manifest.csv")
    slice_scores = read_csv(TABDIR / "13b_new_slice_scores.csv", low_memory=False)
    t01 = read_csv(TABDIR / "13c_T01_source_summary.csv").set_index("source")

    manifest_octdl = manifest[manifest["dataset"] == "OCTDL"]
    joined = octdl_df.merge(manifest_octdl, on="file_name", how="inner",
                            suffixes=("_independent", "_pipeline"))
    assert len(joined) == 2064, (
        f"OCTDL join on file_name produced {len(joined)} rows, expected exactly 2064 -- "
        f"either a missing row or a spurious duplicate match"
    )
    mismatch_rows = joined[
        (joined["true_label_independent"] != joined["true_label_pipeline"])
        | (joined["true_label_name_independent"] != joined["true_label_name_pipeline"])
        | (joined["label_group_independent"] != joined["label_group_pipeline"])
    ]
    report.append("")
    report.append("OCTDL row-by-row mapping consistency: this script's independent "
                  "octdl_label() vs 13a_new_ood_manifest.csv's recorded true_label/"
                  "true_label_name/label_group, joined on file_name (2064 rows)")
    assert len(mismatch_rows) == 0, (
        f"{len(mismatch_rows)} OCTDL rows disagree between the independent mapping and "
        f"13a's recorded output -- first few file_names: "
        f"{mismatch_rows['file_name'].head().tolist()}"
    )
    report.append(f"  0 of 2064 rows disagree -- PASS")

    pipeline_rows = []

    def add(quantity, independent, pipeline_value):
        match = None if pipeline_value is None else (independent == pipeline_value)
        pipeline_rows.append((quantity, independent, pipeline_value, match))

    add("Kermany train images", kermany_train, None)
    add("Kermany val images", kermany_val, None)
    add("Kermany test images", kermany_test, int(t01.loc["Kermany-test", "n"]))
    add("Kermany total images (84,484)", kermany_total, None)

    for v, src in zip(VENDORS, ["RETOUCH-Cirrus", "RETOUCH-Spectralis", "RETOUCH-Topcon"]):
        add(f"RETOUCH-{v} volumes", retouch_vendor_counts.get(v, 0), int(t01.loc[src, "n_identifiers"]))
        add(f"RETOUCH-{v} slices (volumes x9)", retouch_vendor_x9[v], int(t01.loc[src, "n"]))

    add("Rasti volumes", oct5k_total_volumes, int(t01.loc["Rasti", "n_identifiers"]))
    add("Rasti slices (volumes x9)", oct5k_volumes_x9, int(t01.loc["Rasti", "n"]))

    add("NEH deduped rows", neh_n_after, int(t01.loc["NEH", "n"]))
    add("NEH unique patients", neh_n_patients, int(t01.loc["NEH", "n_identifiers"]))

    manifest_neh = manifest[manifest["dataset"] == "NEH"]
    neh_class_pipeline = manifest_neh["Class"].value_counts().to_dict()
    slice_scores_neh = slice_scores[slice_scores["dataset"] == "NEH"]
    neh_label_pipeline = slice_scores_neh["Label"].value_counts().to_dict()
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        add(f"NEH Class={c} (folder)", neh_class_counts.get(c, 0), neh_class_pipeline.get(c, 0))
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        add(f"NEH Label={c} (ground truth)", neh_label_counts.get(c, 0), neh_label_pipeline.get(c, 0))

    for g, src in zip(["in-label", "ambiguous", "novel-class"],
                      ["OCTDL-in-label", "OCTDL-ambiguous", "OCTDL-novel-class"]):
        add(f"OCTDL {g} rows", octdl_tier_counts.get(g, 0), int(t01.loc[src, "n"]))
        add(f"OCTDL {g} unique patients", octdl_tier_patients.get(g, 0),
            int(t01.loc[src, "n_identifiers"]))

    slice_scores_octdl = slice_scores[slice_scores["dataset"] == "OCTDL"]
    add("OCTDL overall unique patients (all tiers)", octdl_n_patients_overall,
        int(slice_scores_octdl["patient_id"].nunique()))

    report.append("")
    report.append(f"{'quantity':<45s} {'independent':>12s} {'pipeline':>12s} verdict")
    for quantity, independent, pipeline_value, match in pipeline_rows:
        if pipeline_value is None:
            verdict = "n/a (pipeline never materialises this quantity)"
            pv_str = "n/a"
        else:
            verdict = "MATCH" if match else "MISMATCH"
            pv_str = str(pipeline_value)
            if not match:
                mismatches_found = True
        report.append(f"{quantity:<45s} {independent:>12} {pv_str:>12} {verdict}")

    # closing
    elapsed = time.time() - t0
    report.append("")
    report.append("-" * 78)
    if mismatches_found:
        report.append("MISMATCHES FOUND -- see MISMATCH/FAIL rows above. Not interpreted "
                      "or corrected by this script.")
    else:
        report.append("No mismatches found in any comparison section.")
    report.append("-" * 78)
    report.append(f"Wall-clock runtime: {elapsed:.1f} s")

    report_path = TABDIR / "16_verify_dataset_counts.txt"
    report_path.write_text("\n".join(report) + "\n")
    print(f"\nWrote {report_path.relative_to(PROJECT)}")
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
