"""Audit NEH and OCTDL metadata, formats, intensities and duplicate images.

Compare with Kermany, RETOUCH and RASTI without changing the source data.
NEH per-image labels come from Label, not the patient diagnosis in Class.
"""

from paths import PROJECT, DATASET, read_csv

import hashlib
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

TABDIR = PROJECT / "results" / "tables"
SPLITS = PROJECT / "splits"

SEED = 42

NEH_ROOT = (DATASET / "Tehran_Labeled Retinal Optical Coherence Tomography "
                      "Dataset for Classification of Normal, Drusen, and "
                      "CNV Cases")
NEH_CSV = NEH_ROOT / "data_information.csv"
NEH_IMG_ROOT = NEH_ROOT / "NEH_UT_2021RetinalOCTDataset"

OCTDL_ROOT = (DATASET / "OCTDL Optical Coherence Tomography Dataset for "
                        "Image-Based Deep Learning Methods")
OCTDL_CSV = OCTDL_ROOT / "OCTDL_labels.csv"
OCTDL_IMG_ROOT = OCTDL_ROOT / "OCTDL"

KERMANY_TRAIN_ROOT = DATASET / "Paul_Mooney" / "OCT2017" / "train"
KERMANY_CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]

RETOUCH_ROOT = (DATASET / "RETOUCH/RETOUCH")
OCT5K_ROOT = (DATASET / "Macular-Dataset-R.Rasti_old")

N_CENTRAL_SLICES = 9
VENDORS = ["Cirrus", "Spectralis", "Topcon"]

IMAGE_SLICE_RE = re.compile(r"^Image\s*(\d+)\.TIFF$", re.IGNORECASE)
IMAGE_SLICE_PNG_FALLBACK_RE = re.compile(r"^Image\s*(\d+)\.PNG$", re.IGNORECASE)

OCT5K_LABEL_MAP = {"AMD": "DRUSEN", "DME": "DME", "Normal": "NORMAL"}

CONTAINER_MAX = 65535.0

# 00c_rescale_comparison.csv's Kermany row is read from this exact path
# (read-only reference for STEP 4's factorial addendum; never recomputed).
OOC_CSV = TABDIR / "00c_rescale_comparison.csv"
N_KERMANY_00C = 300

# Published figures this script checks its own counts against. Source: the
NEH_PATIENTS_PUBLISHED = {"CNV": 161, "DRUSEN": 160, "NORMAL": 120}
NEH_TOTAL_16822 = 16822
NEH_TOTAL_12649 = 12649
NEH_WORSTCASE_PUBLISHED = {"CNV": 3240, "DRUSEN": 3742, "NORMAL": 5667}
OCTDL_CLASS_COUNTS_PUBLISHED = {"AMD": 1231, "DME": 147, "ERM": 155,
                                 "NO": 332, "RVO": 101, "VID": 76, "RAO": 22}
OCTDL_TOTAL_PUBLISHED = 2064

# PIL mode -> (bit depth, channel count). Only modes actually needed for
# this data are listed; anything else is reported as-is rather than guessed.
MODE_INFO = {
    "1": (1, 1), "L": (8, 1), "P": (8, 1), "I": (32, 1),
    "I;16": (16, 1), "I;16B": (16, 1), "F": (32, 1),
    "RGB": (8, 3), "RGBA": (8, 4), "CMYK": (8, 4), "YCbCr": (8, 3),
}


def mode_bitdepth(mode):
    return MODE_INFO.get(mode, (f"unknown({mode})", None))[0]


def mode_channels(mode):
    return MODE_INFO.get(mode, (None, f"unknown({mode})"))[1]


# Formatting helpers
def fmt_stats_line(label, values, width=10):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return f"{label:<28s} (no data)"
    mn, p25, med, p75, mx = (np.min(v), np.percentile(v, 25),
                              np.median(v), np.percentile(v, 75), np.max(v))
    return (f"{label:<28s} min={mn:>{width}.1f}  p25={p25:>{width}.1f}  "
            f"median={med:>{width}.1f}  p75={p75:>{width}.1f}  "
            f"max={mx:>{width}.1f}")


def section(title):
    bar = "=" * 78
    return [bar, title, bar]


def subsection(title):
    return ["", f"--- {title} ---"]


def df_table(df, max_col_width=40):
    return df.to_string(index=False)


# Step 1
def step1_locate():
    lines = section("STEP 1: DATASET LOCATION")
    for name, root, csv in [("NEH", NEH_ROOT, NEH_CSV),
                             ("OCTDL", OCTDL_ROOT, OCTDL_CSV)]:
        if not root.exists():
            raise FileNotFoundError(f"{name} root not found: {root}")
        if not csv.exists():
            raise FileNotFoundError(f"{name} label CSV not found: {csv}")
        lines.append(f"{name} root: {root}")
        lines.append(f"{name} label CSV: {csv}")
    if not KERMANY_TRAIN_ROOT.exists():
        raise FileNotFoundError(f"Kermany train root not found: {KERMANY_TRAIN_ROOT}")
    lines.append(f"Kermany train root: {KERMANY_TRAIN_ROOT}")
    if not RETOUCH_ROOT.exists():
        raise FileNotFoundError(f"RETOUCH root not found: {RETOUCH_ROOT}")
    lines.append(f"RETOUCH root: {RETOUCH_ROOT}")
    if not OCT5K_ROOT.exists():
        raise FileNotFoundError(f"OCT5k root not found: {OCT5K_ROOT}")
    lines.append(f"OCT5k root: {OCT5K_ROOT}")
    return lines


# Data loading
def load_neh():
    df = read_csv(NEH_CSV)
    df["global_patient_id"] = (df["Class"] + "_"
                                + df["Patient ID"].astype(int).astype(str).str.zfill(3))
    df["ext"] = df["Directory"].str.rsplit(".", n=1).str[-1].str.lower()
    df["path"] = df["Directory"].apply(lambda d: NEH_IMG_ROOT / d)
    return df


def load_octdl():
    df = read_csv(OCTDL_CSV)
    df["path"] = df.apply(
        lambda r: OCTDL_IMG_ROOT / r["disease"] / f"{r['file_name']}.jpg", axis=1)
    return df


def load_kermany():
    rows = []
    for c in KERMANY_CLASSES:
        d = KERMANY_TRAIN_ROOT / c
        for fn in os.listdir(d):
            p = d / fn
            if p.is_file():
                rows.append({"class": c, "filename": fn, "path": p})
    return pd.DataFrame(rows)


def guess_vendor(path):
    """Identify the scanner vendor from the volume path."""
    s = str(path).lower()
    for v in VENDORS:
        if v.lower() in s:
            return v
    return "unknown"


def find_retouch_volumes():
    """Find RETOUCH OCT headers and read vendor and ElementType metadata."""
    rows = []
    for mhd in sorted(RETOUCH_ROOT.rglob("oct.mhd")):
        fields = {}
        for line in mhd.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip()] = v.strip()
        rows.append({
            "path": mhd, "vendor": guess_vendor(mhd),
            "set": "train" if "TrainingSet" in str(mhd) else "test",
            "element_type": fields.get("ElementType", "?"),
        })
    return pd.DataFrame(rows)


def central_slice_indices(n_slices, k=N_CENTRAL_SLICES):
    """Return the central slice indices, bounded by the volume length."""
    lo = n_slices // 2 - k // 2
    hi = n_slices // 2 + k // 2
    lo, hi = max(0, lo), min(n_slices - 1, hi)
    return list(range(lo, hi + 1))


def decode_linear_fixed(arr_uint16):
    """Map uint16 to uint8 with round(v / 65535 * 255), using a fixed denominator."""
    out = np.round(arr_uint16.astype(np.float32) / CONTAINER_MAX * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def assemble_retouch_slices(volumes_df):
    """Collect nine central B-scans per volume; decode uint16 using fixed scaling."""
    records = []
    for _, row in volumes_df.iterrows():
        img = sitk.ReadImage(str(row["path"]))
        arr = sitk.GetArrayFromImage(img)
        idx = central_slice_indices(arr.shape[0])
        central = arr[idx]
        is_spectralis = row["element_type"] == "MET_USHORT"
        decoded = decode_linear_fixed(central) if is_spectralis else central.astype(np.uint8)
        for pos, slice_idx in enumerate(idx):
            records.append({
                "vendor": row["vendor"], "set": row["set"],
                "volume_id": row["path"].parent.name,
                "slice_index": slice_idx,
                "array": decoded[pos],
                "raw16": central[pos] if is_spectralis else None,
            })
    return records


def find_oct5k_volumes():
    """Find RASTI volumes with Image n.TIFF files; use PNG only if no TIFF matches."""
    rows = []
    for class_dir in sorted(d for d in OCT5K_ROOT.iterdir() if d.is_dir()):
        for vol_dir in sorted(d for d in class_dir.iterdir() if d.is_dir()):
            all_files = [p for p in vol_dir.rglob("*")
                         if p.is_file() and not p.name.startswith(".")]
            slices = [(int(m.group(1)), p) for p in all_files
                      if (m := IMAGE_SLICE_RE.match(p.name))]
            if not slices:
                slices = [(int(m.group(1)), p) for p in all_files
                          if (m := IMAGE_SLICE_PNG_FALLBACK_RE.match(p.name))]
            slices.sort(key=lambda t: t[0])
            if slices:
                rows.append({"class": class_dir.name, "volume_dir": vol_dir,
                             "slices": slices})
    return rows


def assemble_oct5k_slices(volumes):
    """Collect nine central RASTI B-scans with TIFF-first, PNG-fallback loading."""
    records = []
    for v in volumes:
        slices = v["slices"]
        idx = central_slice_indices(len(slices))
        for pos in idx:
            slice_idx, p = slices[pos]
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L"))
            records.append({
                "class": OCT5K_LABEL_MAP[v["class"]],
                "volume_id": v["volume_dir"].name,
                "slice_index": slice_idx,
                "array": arr,
            })
    return records


# Step 2
def step2_file_listing(neh_df, octdl_df):
    lines = section("STEP 2: FILE LISTING AND CLASS COUNTS")

    lines += subsection("NEH: per-class counts, extensions, version check")
    class_counts = neh_df["Class"].value_counts()
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        n = int(class_counts.get(c, 0))
        pub = NEH_PATIENTS_PUBLISHED[c]
        lines.append(f"  Class={c:<8s} n_images={n:<7d}")
    total = len(neh_df)
    match = total == NEH_TOTAL_16822
    lines.append(f"  Total images: {total}  (published 16,822-image version: "
                 f"{NEH_TOTAL_16822}, match={match}; published 12,649 "
                 f"worst-case version: {NEH_TOTAL_12649}, match={total == NEH_TOTAL_12649})")
    lines.append(f"  This copy is the {'16,822' if match else 'NEITHER 16,822 NOR 12,649'} version.")
    ext_counts = neh_df["ext"].value_counts()
    lines.append(f"  Extension distribution: {ext_counts.to_dict()}")

    lines += subsection("OCTDL: per-class counts vs published, extension distribution")
    oc_counts = octdl_df["disease"].value_counts()
    for c, pub in OCTDL_CLASS_COUNTS_PUBLISHED.items():
        n = int(oc_counts.get(c, 0))
        lines.append(f"  disease={c:<6s} n_images={n:<6d} published={pub:<6d} match={n==pub}")
    total_oc = len(octdl_df)
    lines.append(f"  Total images: {total_oc}  published={OCTDL_TOTAL_PUBLISHED}  "
                 f"match={total_oc == OCTDL_TOTAL_PUBLISHED}")
    ext_oc = octdl_df["path"].apply(lambda p: p.suffix.lower().lstrip(".")).value_counts()
    lines.append(f"  Extension distribution: {ext_oc.to_dict()}")

    # Bidirectional CSV/disk reconciliation
    lines += subsection("NEH: CSV -> disk (every Directory path, existence only)")
    exists_mask = neh_df["path"].apply(lambda p: p.exists())
    n_exist, n_missing = int(exists_mask.sum()), int((~exists_mask).sum())
    lines.append(f"  Exists: {n_exist}   Missing: {n_missing}")
    if n_missing:
        missing_paths = neh_df.loc[~exists_mask, "Directory"].tolist()
        shown = missing_paths[:20]
        lines.append(f"  First {len(shown)} of {n_missing} missing paths:")
        for m in shown:
            lines.append(f"    {m}")

    lines += subsection("NEH: disk -> CSV (independent directory walk, extension counts only)")
    disk_files = {}
    for dirpath, _, filenames in os.walk(NEH_IMG_ROOT):
        for fn in filenames:
            if fn.startswith("."):
                continue
            ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
            if ext not in ("tif", "jpg", "tiff", "jpeg"):
                continue
            disk_files[str(Path(dirpath) / fn)] = ext
    disk_total = len(disk_files)
    disk_ext = Counter(disk_files.values())
    lines.append(f"  Disk walk total: {disk_total}   CSV total: {total}   "
                 f"match={disk_total == total}")
    lines.append(f"  Disk walk .tif: {disk_ext.get('tif', 0) + disk_ext.get('tiff', 0)}"
                 f"   CSV .tif: {int(ext_counts.get('tif', 0))}   "
                 f"match={(disk_ext.get('tif', 0) + disk_ext.get('tiff', 0)) == int(ext_counts.get('tif', 0))}")
    lines.append(f"  Disk walk .jpg: {disk_ext.get('jpg', 0) + disk_ext.get('jpeg', 0)}"
                 f"   CSV .jpg: {int(ext_counts.get('jpg', 0))}   "
                 f"match={(disk_ext.get('jpg', 0) + disk_ext.get('jpeg', 0)) == int(ext_counts.get('jpg', 0))}")
    csv_paths_resolved = set(str(p.resolve()) for p in neh_df["path"])
    disk_paths_resolved = set(str(Path(p).resolve()) for p in disk_files)
    on_disk_not_in_csv = disk_paths_resolved - csv_paths_resolved
    lines.append(f"  On disk but not in CSV Directory column: {len(on_disk_not_in_csv)}")
    if on_disk_not_in_csv:
        for p in sorted(on_disk_not_in_csv)[:20]:
            lines.append(f"    {p}")

    n_exact_dup_rows = int(neh_df["Directory"].duplicated().sum())
    n_unique_directory_strings = neh_df["Directory"].nunique()
    in_csv_not_on_disk = csv_paths_resolved - disk_paths_resolved
    lines.append(f"  CSV rows: {total}   exact-duplicate Directory rows "
                 f"(identical string, appears >1 time): {n_exact_dup_rows}   "
                 f"unique Directory strings: {n_unique_directory_strings}")
    lines.append(f"  Unique Directory strings not matched to any disk-walk "
                 f"path by exact string: {len(in_csv_not_on_disk)} "
                 f"(sum of the two effects above: {n_exact_dup_rows} + "
                 f"{len(in_csv_not_on_disk)} = {n_exact_dup_rows + len(in_csv_not_on_disk)}, "
                 f"vs CSV-total-minus-disk-total gap of {total - disk_total})")
    if in_csv_not_on_disk:
        disk_paths_lower = {p.lower() for p in disk_paths_resolved}
        n_case_variant = sum(1 for p in in_csv_not_on_disk if p.lower() in disk_paths_lower)
        lines.append(f"  These {len(in_csv_not_on_disk)} all pass exist()==True per the "
                     f"CSV->disk check above, so none is a missing-file case. Checked "
                     f"directly: {n_case_variant} of {len(in_csv_not_on_disk)} match a "
                     f"disk-walk path when compared case-insensitively (i.e. the CSV's "
                     f"Directory string differs from the real on-disk filename only in "
                     f"letter case); the remaining "
                     f"{len(in_csv_not_on_disk) - n_case_variant} do not and are listed "
                     f"below with no explanation assumed.")
        for p in sorted(in_csv_not_on_disk)[:20]:
            lines.append(f"    {p}")

    lines += subsection("OCTDL: CSV -> disk (every file_name path, existence only)")
    exists_mask_oc = octdl_df["path"].apply(lambda p: p.exists())
    n_exist_oc, n_missing_oc = int(exists_mask_oc.sum()), int((~exists_mask_oc).sum())
    lines.append(f"  Exists: {n_exist_oc}   Missing: {n_missing_oc}")
    if n_missing_oc:
        for m in octdl_df.loc[~exists_mask_oc, "path"].tolist()[:20]:
            lines.append(f"    {m}")
    lines.append("  OCTDL disk -> CSV direction already covered above "
                 "(per-class counts independently matched against the CSV "
                 "and the published figures) -- not repeated here.")

    return lines


# Step 3
def sample_rows(df, n, seed):
    rng = np.random.default_rng(seed)
    n = min(n, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def sample_records(records, n, seed):
    """Sample up to n records without replacement using the supplied generator."""
    rng = np.random.default_rng(seed)
    n = min(n, len(records))
    idx = rng.choice(len(records), size=n, replace=False)
    return [records[i] for i in idx]


def read_header(path):
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode


def read_header_with_format(path):
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode, im.format


def rgb_channels_equal(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            return None
        arr = np.asarray(im)
        return bool((arr[..., 0] == arr[..., 1]).all() and (arr[..., 1] == arr[..., 2]).all())


def step3_image_properties(neh_sample, octdl_sample):
    lines = section("STEP 3: IMAGE PROPERTIES (n=200 per dataset, seed=42)")

    def report_block(label, paths):
        widths, heights, sizes, modes = [], [], [], []
        rgb_eq = []
        for p in paths:
            w, h, mode = read_header(p)
            widths.append(w); heights.append(h); modes.append(mode)
            sizes.append(os.path.getsize(p))
            eq = rgb_channels_equal(p)
            if eq is not None:
                rgb_eq.append(eq)
        lines2 = []
        lines2.append(fmt_stats_line(f"{label} width", widths))
        lines2.append(fmt_stats_line(f"{label} height", heights))
        lines2.append(fmt_stats_line(f"{label} file size (bytes)", sizes))
        lines2.append(f"{label} mode distribution: {Counter(modes)}")
        if rgb_eq:
            n_eq = sum(rgb_eq)
            lines2.append(f"{label} RGB channel equality: {n_eq} of {len(rgb_eq)} "
                          f"RGB images have R==G==B exactly")
        return lines2, modes

    lines += subsection("NEH (pooled, n=200)")
    neh_lines, neh_modes = report_block("NEH", neh_sample["path"].tolist())
    lines += neh_lines

    lines += subsection("NEH by extension")
    ext_counts_sample = neh_sample["ext"].value_counts().to_dict()
    lines.append(f"Of the 200 draws: {ext_counts_sample}")
    for ext in ["tif", "jpg"]:
        sub = neh_sample[neh_sample["ext"] == ext]
        if len(sub) == 0:
            lines.append(f"NEH .{ext}: no samples drawn")
            continue
        sub_lines, _ = report_block(f"NEH .{ext}", sub["path"].tolist())
        lines += sub_lines

    lines += subsection("OCTDL (pooled, n=200)")
    oc_lines, _ = report_block("OCTDL", octdl_sample["path"].tolist())
    lines += oc_lines

    lines += subsection("OCTDL: CSV-recorded vs actual (image_width, image_hight)")
    n_match, n_mismatch, mismatches = 0, 0, []
    for _, row in octdl_sample.iterrows():
        w, h, _ = read_header(row["path"])
        if w == row["image_width"] and h == row["image_hight"]:
            n_match += 1
        else:
            n_mismatch += 1
            mismatches.append((row["file_name"], row["image_width"], row["image_hight"], w, h))
    lines.append(f"Match: {n_match}   Mismatch: {n_mismatch}  (of {len(octdl_sample)} sampled)")
    for fn, cw, ch, aw, ah in mismatches:
        lines.append(f"  {fn}: csv=({cw},{ch}) actual=({aw},{ah})")

    return lines


# Step 4
def raw_pixel_stats(arr):
    a = arr.astype(np.float64)
    pcts = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return {
        "p1": pcts[0], "p5": pcts[1], "p25": pcts[2], "p50": pcts[3],
        "p75": pcts[4], "p95": pcts[5], "p99": pcts[6],
        "mean": a.mean(), "std": a.std(),
    }


def pooled_pixel_stats(arrays):
    """Compute statistics after pooling all sampled pixels, rather than per-image means."""
    pooled = np.concatenate([a.ravel().astype(np.float64) for a in arrays])
    pcts = np.percentile(pooled, [1, 5, 25, 50, 75, 95, 99])
    return {
        "p1": pcts[0], "p5": pcts[1], "p25": pcts[2], "p50": pcts[3],
        "p75": pcts[4], "p95": pcts[5], "p99": pcts[6],
        "mean": pooled.mean(), "std": pooled.std(),
    }


def open_grayscale(path):
    """Read a single-channel image, converting to greyscale when needed."""
    with Image.open(path) as im:
        if im.mode not in ("L", "1", "I", "I;16", "I;16B", "F"):
            im = im.convert("L")
        return np.asarray(im)


def open_grayscale_with_format(path):
    """Read greyscale pixels and the detected image format, independent of the suffix."""
    with Image.open(path) as im:
        fmt = im.format
        if im.mode not in ("L", "1", "I", "I;16", "I;16B", "F"):
            im = im.convert("L")
        return np.asarray(im), fmt


def stats_line(label, stats_list):
    return [f"  mean-of-{len(stats_list)} {key:<6s}: {np.mean([s[key] for s in stats_list]):.4f}"
            for key in ["p1", "p5", "p25", "p50", "p75", "p95", "p99", "mean", "std"]]


def pooled_stats_lines(label, d):
    return [f"  {label} {key:<6s}: {d[key]:.4f}"
            for key in ["p1", "p5", "p25", "p50", "p75", "p95", "p99", "mean", "std"]]


def step4_intensity(neh_df, octdl_df, kermany_df,
                     retouch_volumes_df, retouch_records,
                     oct5k_volumes, oct5k_records, octdl_hdr):
    lines = section("STEP 4: INTENSITY DISTRIBUTIONS (n=500 per dataset, seed=42, raw pixels)")
    lines.append("Computed on raw pixel arrays before any resize or normalisation. "
                 "Independent of results/tables/00c_rescale_comparison.csv "
                 "(different code, not reused, except in the Kermany factorial "
                 "addendum at the end of this section, which reads that file's "
                 "already-saved Kermany row read-only).")
    lines.append("RETOUCH is captured post-SimpleITK-read (and, for Spectralis, "
                 "post-decode) with no PIL step -- MetaImage is not a "
                 "PIL-readable format. NEH, OCTDL, Kermany-train and OCT5k are "
                 "all captured post-PIL-convert(\"L\"). This is the one declared "
                 "path difference in this comparison.")
    lines.append("RETOUCH decode: Spectralis (MET_USHORT, 16-bit) is decoded "
                 "via decode_linear_fixed (round(v/65535*255), clipped to "
                 "[0,255]); Topcon and Cirrus (MET_UCHAR, 8-bit) are cast to "
                 "uint8 directly, no decode. The three RETOUCH rows below are "
                 "therefore not measured under identical processing.")

    per_image_records = []

    samples = {
        "NEH": sample_rows(neh_df, 500, SEED),
        "OCTDL": sample_rows(octdl_df, 500, SEED),
        "Kermany-train": sample_rows(kermany_df, 500, SEED),
    }

    octdl_formats_sample = None

    for dsname, sample in samples.items():
        class_col = {"NEH": "Class", "OCTDL": "disease", "Kermany-train": "class"}[dsname]
        stats_list = []
        formats = []
        for _, row in sample.iterrows():
            if dsname == "OCTDL":
                arr, fmt = open_grayscale_with_format(row["path"])
                formats.append(fmt)
            else:
                arr = open_grayscale(row["path"])
            st = raw_pixel_stats(arr)
            st["dataset"] = dsname
            st["class"] = row[class_col]
            if dsname == "NEH":
                st["ext"] = row["ext"]
            if dsname == "OCTDL":
                st["format"] = formats[-1]
            stats_list.append(st)
        per_image_records.extend(stats_list)

        lines += subsection(f"{dsname} (n={len(stats_list)})")
        lines += stats_line(dsname, stats_list)

        if dsname == "NEH":
            lines += subsection("NEH R/G/B channel stats (same sampled images), pooled and by extension")
            for scope_label, scope_sample in [("NEH pooled", sample),
                                               ("NEH .tif", sample[sample["ext"] == "tif"]),
                                               ("NEH .jpg", sample[sample["ext"] == "jpg"])]:
                if len(scope_sample) == 0:
                    lines.append(f"  {scope_label}: no samples")
                    continue
                r_all, g_all, b_all = [], [], []
                for _, row in scope_sample.iterrows():
                    with Image.open(row["path"]) as im:
                        arr = np.asarray(im).astype(np.float64)
                    r_all.append(arr[..., 0].ravel())
                    g_all.append(arr[..., 1].ravel())
                    b_all.append(arr[..., 2].ravel())
                r = np.concatenate(r_all); g = np.concatenate(g_all); b = np.concatenate(b_all)
                lines.append(f"  {scope_label} (n={len(scope_sample)} images): "
                             f"R mean={r.mean():.4f} sd={r.std():.4f}; "
                             f"G mean={g.mean():.4f} sd={g.std():.4f}; "
                             f"B mean={b.mean():.4f} sd={b.std():.4f}")
                lines.append(f"  {scope_label} pairwise mean |diff|: "
                             f"|R-G|={np.abs(r-g).mean():.4f}  "
                             f"|G-B|={np.abs(g-b).mean():.4f}  "
                             f"|R-B|={np.abs(r-b).mean():.4f}")

        if dsname == "OCTDL":
            octdl_formats_sample = pd.DataFrame(stats_list)
            octdl_formats_sample["path"] = sample["path"].to_numpy()
            octdl_formats_sample["image_width"] = sample["image_width"].to_numpy()
            octdl_formats_sample["image_hight"] = sample["image_hight"].to_numpy()

    # RETOUCH: per-vendor pool -> sample -> stats
    lines += subsection("RETOUCH: native ElementType by vendor, from find_retouch_volumes()")
    et_ct = pd.crosstab(retouch_volumes_df["vendor"], retouch_volumes_df["element_type"])
    lines.append(et_ct.to_string())
    lines.append("MET_USHORT = 16-bit native. MET_UCHAR = 8-bit native. "
                 "(CSV's bitdepth column for these rows reflects the "
                 "post-decode value that enters the pipeline, 8, not this "
                 "native value.)")

    retouch_spectralis_sample = None
    for vendor in ["Spectralis", "Cirrus", "Topcon"]:
        pool = [r for r in retouch_records if r["vendor"] == vendor]
        sample = sample_records(pool, 500, SEED)
        if vendor == "Spectralis":
            retouch_spectralis_sample = sample
        stats_list = []
        for rec in sample:
            st = raw_pixel_stats(rec["array"])
            st["dataset"] = f"RETOUCH-{vendor}"
            st["class"] = "all"
            stats_list.append(st)
        per_image_records.extend(stats_list)
        lines += subsection(f"RETOUCH-{vendor} (pool={len(pool)}, sample n={len(stats_list)})")
        lines += stats_line(f"RETOUCH-{vendor}", stats_list)

    # RETOUCH Spectralis: pre-decode addendum
    lines += subsection("RETOUCH-Spectralis: pre-decode (raw 16-bit) statistics, same sampled slices")
    raw_stats_list = [raw_pixel_stats(rec["raw16"]) for rec in retouch_spectralis_sample]
    lines += stats_line("RETOUCH-Spectralis raw16", raw_stats_list)
    all_raw = np.concatenate([rec["raw16"].ravel() for rec in retouch_spectralis_sample])
    lines.append(f"  RETOUCH-Spectralis raw16 min={int(all_raw.min())} max={int(all_raw.max())} "
                 f"(across all sampled raw pixels, pooled)")

    # OCT5k: pooled sample, split by class post-hoc
    oct5k_sample = sample_records(oct5k_records, 500, SEED)
    stats_list = []
    for rec in oct5k_sample:
        st = raw_pixel_stats(rec["array"])
        st["dataset"] = "OCT5k"
        st["class"] = rec["class"]
        stats_list.append(st)
    per_image_records.extend(stats_list)
    lines += subsection(f"OCT5k (pool={len(oct5k_records)}, sample n={len(stats_list)})")
    lines += stats_line("OCT5k", stats_list)

    # Part 5: OCTDL split by actual container format
    lines += subsection("OCTDL by container format (PIL Image.format, not extension): full population (all 2,064 images)")
    fmt_by_disease = pd.crosstab(octdl_hdr["disease"], octdl_hdr["format"])
    fmt_by_disease_pct = (fmt_by_disease.div(fmt_by_disease.sum(axis=1), axis=0) * 100).round(2)
    lines.append("Counts:")
    lines.append(fmt_by_disease.to_string())
    lines.append("Row-normalised % (share of each disease's images that are each format):")
    lines.append(fmt_by_disease_pct.to_string())
    total_fmt = octdl_hdr["format"].value_counts()
    lines.append(f"Total: {total_fmt.to_dict()}  (of {len(octdl_hdr)} images)")

    lines += subsection("OCTDL by container format: STEP 4's existing 500-sample only")
    for fmt in ["PNG", "JPEG"]:
        sub = octdl_formats_sample[octdl_formats_sample["format"] == fmt]
        lines.append(f"  format={fmt}: n={len(sub)} of {len(octdl_formats_sample)}")
        if len(sub) == 0:
            continue
        sub_stats = sub.to_dict("records")
        lines += stats_line(f"OCTDL {fmt}", sub_stats)
        widths = [read_header(p)[0] for p in sub["path"]]
        heights = [read_header(p)[1] for p in sub["path"]]
        sizes = [os.path.getsize(p) for p in sub["path"]]
        lines.append(fmt_stats_line(f"OCTDL {fmt} width", widths))
        lines.append(fmt_stats_line(f"OCTDL {fmt} height", heights))
        lines.append(fmt_stats_line(f"OCTDL {fmt} file size (bytes)", sizes))

    # Part 6: Kermany 2x2 factorial + 00c reference
    lines += subsection("Kermany p99 factorial design (isolating 3 factors vs 00c_rescale_comparison.csv)")
    lines.append("Factors: (1) statistic definition [per-image percentile "
                 "then mean, vs pooled-pixel percentile]; (2) sampling pool "
                 "[raw Dataset folder, vs this project's locked "
                 "splits/train.csv]; (3) sample size and RNG [n=500 numpy "
                 "default_rng(42), vs n=300 pandas random_state=42]. "
                 "splits/train.csv (70,484 rows) is a subset of the raw "
                 "folder (83,484 files): the carved-out validation and "
                 "calibration images are excluded from it. Stated as a fact "
                 "only, no view taken on which pool is more appropriate.")

    kermany_500_raw_arrays = [open_grayscale(row["path"]) for _, row in samples["Kermany-train"].iterrows()]
    cell1 = [s for s in per_image_records if s["dataset"] == "Kermany-train"]
    cell2 = pooled_pixel_stats(kermany_500_raw_arrays)

    train_csv = read_csv(SPLITS / "train.csv")
    train_csv_sample = sample_rows(train_csv, 500, SEED)
    cell34_arrays = []
    cell3_stats = []
    for _, row in train_csv_sample.iterrows():
        arr = open_grayscale(row["filepath"])
        cell34_arrays.append(arr)
        cell3_stats.append(raw_pixel_stats(arr))
    cell4 = pooled_pixel_stats(cell34_arrays)

    lines.append("Cell 1 (per-image mean, raw folder, n=500, numpy) -- "
                 "reused from the Kermany-train row above, not recomputed:")
    lines += stats_line("cell1", cell1)
    lines.append("Cell 2 (pooled-pixel, raw folder, n=500, numpy, SAME 500 "
                 "files as cell 1 -- isolates statistic definition vs cell 1):")
    lines += pooled_stats_lines("cell2", cell2)
    lines.append("Cell 3 (per-image mean, splits/train.csv, n=500, numpy, "
                 "fresh draw -- isolates sampling pool vs cell 1):")
    lines += stats_line("cell3", cell3_stats)
    lines.append("Cell 4 (pooled-pixel, splits/train.csv, n=500, numpy, SAME "
                 "500 files as cell 3):")
    lines += pooled_stats_lines("cell4", cell4)

    ooc_df = read_csv(OOC_CSV)
    ooc_row = ooc_df[ooc_df["variant"] == "Kermany (8-bit JPEG)"].iloc[0]
    lines.append(f"Reference (00c_rescale_comparison.csv's own recorded Kermany "
                 f"row -- pooled-pixel, splits/train.csv via pandas.sample, "
                 f"n={N_KERMANY_00C}, pandas random_state=42 -- read only, not "
                 f"recomputed. Comparing cell 4 against this isolates sample "
                 f"size/generator, since both are pooled-pixel over "
                 f"splits/train.csv):")
    for key in ["p1", "p5", "p25", "p50", "p75", "p95", "p99", "mean", "std"]:
        lines.append(f"  ref {key:<6s}: {float(ooc_row[key]):.4f}")

    return lines, pd.DataFrame(per_image_records)


# Step 5
def step5_neh_structure(neh_df):
    lines = section("STEP 5a: NEH PATIENT / VOLUME STRUCTURE")
    lines.append(f"data_information.csv columns: {neh_df.columns.tolist()}")

    lines += subsection("Patients per class, B-scans per patient")
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        sub = neh_df[neh_df["Class"] == c]
        n_patients = sub["global_patient_id"].nunique()
        per_patient = sub.groupby("global_patient_id").size()
        lines.append(f"  Class={c:<8s} n_patients={n_patients:<5d} "
                     f"B-scans/patient: min={per_patient.min()} "
                     f"median={per_patient.median():.1f} max={per_patient.max()}")

    lines += subsection("Eye distribution and both-eyes-present check")
    eye_counts = neh_df["Eye"].value_counts()
    lines.append(f"  Eye value counts: {eye_counts.to_dict()}")
    eye_per_patient = neh_df.groupby("global_patient_id")["Eye"].nunique()
    both = int((eye_per_patient > 1).sum())
    lines.append(f"  Patients (class,patient) with both OD and OS present: "
                 f"{both} of {len(eye_per_patient)}")

    lines += subsection("Class x Label crosstab (absolute)")
    ct = pd.crosstab(neh_df["Class"], neh_df["Label"])
    lines.append(df_table(ct.reset_index()))
    lines += subsection("Class x Label crosstab (row-normalised %)")
    ct_pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(2)
    lines.append(df_table(ct_pct.reset_index()))

    lines += subsection("Label == Class: count and share per class")
    same = neh_df[neh_df["Class"] == neh_df["Label"]]
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        n_c = int((neh_df["Class"] == c).sum())
        n_same = int((same["Class"] == c).sum())
        lines.append(f"  Class={c:<8s} Label==Class: {n_same} of {n_c} "
                     f"({100*n_same/n_c:.2f}%)")

    lines += subsection("Filtered to Label==Class only: per-class counts vs published worst-case (12,649)")
    filt_counts = same.groupby("Class").size()
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        computed = int(filt_counts.get(c, 0))
        pub = NEH_WORSTCASE_PUBLISHED[c]
        lines.append(f"  {c:<8s} computed={computed:<6d} published={pub:<6d} match={computed==pub}")
    total_filt = len(same)
    lines.append(f"  Total    computed={total_filt:<6d} published={NEH_TOTAL_12649:<6d} "
                 f"match={total_filt==NEH_TOTAL_12649}")

    return lines


def step5_octdl_structure(octdl_df):
    lines = section("STEP 5b: OCTDL STRUCTURE")

    lines += subsection("Generic per-column unique-value report")
    for col in ["eye", "sex", "year", "disease"]:
        vals = octdl_df[col].unique()
        if len(vals) <= 20:
            lines.append(f"  {col}: {sorted(vals.tolist(), key=str)}")
        else:
            first10 = sorted(vals.tolist(), key=str)[:10]
            lines.append(f"  {col}: {len(vals)} unique values, first 10: {first10}")

    lines += subsection("AMD subcategory breakdown")
    amd = octdl_df[octdl_df["disease"] == "AMD"]
    lines.append(df_table(amd["subcategory"].value_counts().reset_index()))

    lines += subsection("condition: AMD-only condition x subcategory crosstab")
    ct_amd = pd.crosstab(amd["condition"], amd["subcategory"])
    lines.append(df_table(ct_amd.reset_index()))

    lines += subsection("condition: disease x condition crosstab (all seven diseases)")
    ct_all = pd.crosstab(octdl_df["disease"], octdl_df["condition"])
    lines.append(df_table(ct_all.reset_index()))

    lines += subsection("condition values appearing under >=2 diseases")
    cd = octdl_df.groupby("condition")["disease"].unique()
    multi = cd[cd.apply(len) > 1]
    if len(multi) == 0:
        lines.append("  none")
    for cond, diseases in multi.items():
        counts = octdl_df[octdl_df["condition"] == cond].groupby("disease").size().to_dict()
        lines.append(f"  {cond}: diseases={list(diseases)} counts={counts}")

    lines += subsection("condition missingness (three definitions)")
    isna = int(octdl_df["condition"].isna().sum())
    empty = int((octdl_df["condition"].astype(str) == "").sum())
    zero_code = int((octdl_df["condition"].astype(str) == "0").sum())
    lines.append(f"  isna: {isna}   empty string: {empty}   literal '0': {zero_code}")

    lines += subsection("patient_id: global count and reconciliation with published per-disease sum")
    global_n = octdl_df["patient_id"].nunique()
    per_disease_n = octdl_df.groupby("disease")["patient_id"].nunique()
    lines.append(f"  Global unique patient_id: {global_n}")
    lines.append(f"  Per-disease unique patient_id: {per_disease_n.to_dict()}  "
                 f"sum={per_disease_n.sum()}")
    pd_disease = octdl_df.groupby("patient_id")["disease"].unique()
    multi_pd = pd_disease[pd_disease.apply(len) > 1]
    lines.append(f"  patient_id values in >=2 diseases: {len(multi_pd)}")
    for pid, diseases in multi_pd.items():
        lines.append(f"    {pid}: {list(diseases)}")

    lines += subsection("patient_id: images per patient")
    per_patient = octdl_df.groupby("patient_id").size()
    lines.append(f"  min={per_patient.min()} median={per_patient.median():.1f} max={per_patient.max()}")

    lines += subsection("image_width / image_hight distributions (from CSV columns)")
    lines.append(fmt_stats_line("image_width", octdl_df["image_width"]))
    lines.append(fmt_stats_line("image_hight", octdl_df["image_hight"]))

    return lines


# Step 6
def md5_of_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step6_duplicates(neh_df, octdl_df):
    lines = section("STEP 6: DUPLICATE CHECK (full population, MD5 of file bytes)")

    lines += subsection("NEH")
    neh_df = neh_df.copy()
    neh_df["md5"] = neh_df["path"].apply(md5_of_file)
    lines += classify_duplicate_groups(
        neh_df, hash_col="md5", patient_col="global_patient_id", class_col="Class",
        detail_cols=["global_patient_id", "Class", "Label", "Directory"])

    lines += subsection("OCTDL")
    octdl_df = octdl_df.copy()
    octdl_df["md5"] = octdl_df["path"].apply(md5_of_file)
    lines += classify_duplicate_groups(
        octdl_df, hash_col="md5", patient_col="patient_id", class_col="disease",
        detail_cols=["patient_id", "disease", "file_name"])

    return lines


def classify_duplicate_groups(df, hash_col, patient_col, class_col, detail_cols):
    lines = []
    cat1_groups = cat2_groups = cat3_groups = 0
    cat1_files = cat2_files = cat3_files = 0
    cat2_detail, cat3_detail = [], []

    for h, group in df.groupby(hash_col):
        if len(group) < 2:
            continue
        patients = set(group[patient_col])
        classes = set(group[class_col])
        if len(classes) > 1:
            cat3_groups += 1
            cat3_files += len(group)
            cat3_detail.append(group[detail_cols].to_dict("records"))
        elif len(patients) > 1:
            cat2_groups += 1
            cat2_files += len(group)
            cat2_detail.append(group[detail_cols].to_dict("records"))
        else:
            cat1_groups += 1
            cat1_files += len(group)

    lines.append(f"  Category 1 (same patient, same class): "
                 f"{cat1_groups} groups, {cat1_files} files")
    lines.append(f"  Category 2 (same class, different patients): "
                 f"{cat2_groups} groups, {cat2_files} files")
    lines.append(f"  Category 3 (different classes): "
                 f"{cat3_groups} groups, {cat3_files} files")

    for label, detail, total in [("Category 2", cat2_detail, cat2_groups),
                                  ("Category 3", cat3_detail, cat3_groups)]:
        if total == 0:
            continue
        lines.append(f"  {label} groups (showing up to 20 of {total}):")
        for members in detail[:20]:
            lines.append(f"    group: {members}")

    return lines


# CSV construction
def exact_header_scan(df, path_col, class_col):
    """Read dimensions, colour mode and format from every image header."""
    widths, heights, modes, formats = [], [], [], []
    for p in df[path_col]:
        w, h, mode, fmt = read_header_with_format(p)
        widths.append(w); heights.append(h); modes.append(mode); formats.append(fmt)
    out = df[[class_col]].copy()
    out["width"] = widths
    out["height"] = heights
    out["mode"] = modes
    out["format"] = formats
    return out


def collapse_or_join(values):
    distinct = sorted(set(str(v) for v in values))
    if len(distinct) == 1:
        return distinct[0]
    return "|".join(distinct)


def build_csv_rows(neh_df, octdl_df, kermany_df, intensity_df,
                    retouch_volumes_df, retouch_records,
                    oct5k_volumes, oct5k_records, octdl_hdr):
    rows = []

    # NEH
    neh_hdr = exact_header_scan(neh_df, "path", "Class")
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        sub = neh_hdr[neh_hdr["Class"] == c]
        n_images = int((neh_df["Class"] == c).sum())
        n_patients = int(neh_df.loc[neh_df["Class"] == c, "global_patient_id"].nunique())
        bitdepths = sub["mode"].apply(mode_bitdepth)
        nchannels = sub["mode"].apply(mode_channels)
        inten = intensity_df[(intensity_df["dataset"] == "NEH") & (intensity_df["class"] == c)]
        rows.append(dict(
            dataset="NEH", **{"class": c}, n_images=n_images, n_patients=n_patients,
            width_median=float(sub["width"].median()), height_median=float(sub["height"].median()),
            bitdepth=collapse_or_join(bitdepths), n_channels=collapse_or_join(nchannels),
            n_distinct_modes=int(sub["mode"].nunique()),
            p1=float(inten["p1"].mean()), p50=float(inten["p50"].mean()),
            p99=float(inten["p99"].mean()), mean_intensity=float(inten["mean"].mean()),
            n_intensity_sample=int(len(inten)),
        ))

    for c in OCTDL_CLASS_COUNTS_PUBLISHED:
        sub = octdl_hdr[octdl_hdr["disease"] == c]
        n_images = int((octdl_df["disease"] == c).sum())
        n_patients = int(octdl_df.loc[octdl_df["disease"] == c, "patient_id"].nunique())
        bitdepths = sub["mode"].apply(mode_bitdepth)
        nchannels = sub["mode"].apply(mode_channels)
        inten = intensity_df[(intensity_df["dataset"] == "OCTDL") & (intensity_df["class"] == c)]
        rows.append(dict(
            dataset="OCTDL", **{"class": c}, n_images=n_images, n_patients=n_patients,
            width_median=float(sub["width"].median()), height_median=float(sub["height"].median()),
            bitdepth=collapse_or_join(bitdepths), n_channels=collapse_or_join(nchannels),
            n_distinct_modes=int(sub["mode"].nunique()),
            p1=float(inten["p1"].mean()), p50=float(inten["p50"].mean()),
            p99=float(inten["p99"].mean()), mean_intensity=float(inten["mean"].mean()),
            n_intensity_sample=int(len(inten)),
        ))

    # Kermany-train
    kerm_hdr = exact_header_scan(kermany_df, "path", "class")
    for c in KERMANY_CLASSES:
        sub = kerm_hdr[kerm_hdr["class"] == c]
        n_images = int((kermany_df["class"] == c).sum())
        bitdepths = sub["mode"].apply(mode_bitdepth)
        nchannels = sub["mode"].apply(mode_channels)
        inten = intensity_df[(intensity_df["dataset"] == "Kermany-train") & (intensity_df["class"] == c)]
        rows.append(dict(
            dataset="Kermany-train", **{"class": c}, n_images=n_images, n_patients="NA",
            width_median=float(sub["width"].median()), height_median=float(sub["height"].median()),
            bitdepth=collapse_or_join(bitdepths), n_channels=collapse_or_join(nchannels),
            n_distinct_modes=int(sub["mode"].nunique()),
            p1=float(inten["p1"].mean()), p50=float(inten["p50"].mean()),
            p99=float(inten["p99"].mean()), mean_intensity=float(inten["mean"].mean()),
            n_intensity_sample=int(len(inten)),
        ))

    # RETOUCH: three vendor rows, class="all" (no usable label in the pipeline)
    for vendor in ["Spectralis", "Cirrus", "Topcon"]:
        vol_rows = retouch_volumes_df[retouch_volumes_df["vendor"] == vendor]
        n_patients = len(vol_rows)
        vendor_records = [r for r in retouch_records if r["vendor"] == vendor]
        n_images = len(vendor_records)
        widths = [r["array"].shape[1] for r in vendor_records]
        heights = [r["array"].shape[0] for r in vendor_records]
        inten = intensity_df[(intensity_df["dataset"] == f"RETOUCH-{vendor}") & (intensity_df["class"] == "all")]
        rows.append(dict(
            dataset=f"RETOUCH-{vendor}", **{"class": "all"}, n_images=n_images, n_patients=n_patients,
            width_median=float(np.median(widths)), height_median=float(np.median(heights)),
            bitdepth="8", n_channels="1",
            n_distinct_modes=1,
            p1=float(inten["p1"].mean()), p50=float(inten["p50"].mean()),
            p99=float(inten["p99"].mean()), mean_intensity=float(inten["mean"].mean()),
            n_intensity_sample=int(len(inten)),
        ))

    # OCT5k: three class rows (usable label exists, per OCT5K_LABEL_MAP)
    oct5k_class_of = {v["volume_dir"].name: OCT5K_LABEL_MAP[v["class"]] for v in oct5k_volumes}
    for cls in ["DRUSEN", "DME", "NORMAL"]:
        cls_volumes = [v for v in oct5k_volumes if OCT5K_LABEL_MAP[v["class"]] == cls]
        n_patients = len(cls_volumes)
        cls_records = [r for r in oct5k_records if r["class"] == cls]
        n_images = len(cls_records)
        widths = [r["array"].shape[1] for r in cls_records]
        heights = [r["array"].shape[0] for r in cls_records]
        inten = intensity_df[(intensity_df["dataset"] == "OCT5k") & (intensity_df["class"] == cls)]
        rows.append(dict(
            dataset="OCT5k", **{"class": cls}, n_images=n_images, n_patients=n_patients,
            width_median=float(np.median(widths)), height_median=float(np.median(heights)),
            bitdepth="8", n_channels="1",
            n_distinct_modes=1,
            p1=float(inten["p1"].mean()), p50=float(inten["p50"].mean()),
            p99=float(inten["p99"].mean()), mean_intensity=float(inten["mean"].mean()),
            n_intensity_sample=int(len(inten)),
        ))

    return pd.DataFrame(rows, columns=[
        "dataset", "class", "n_images", "n_patients", "width_median", "height_median",
        "bitdepth", "n_channels", "n_distinct_modes", "p1", "p50", "p99",
        "mean_intensity", "n_intensity_sample"])


def main():
    all_lines = []
    all_lines += step1_locate()

    neh_df = load_neh()
    octdl_df = load_octdl()
    kermany_df = load_kermany()

    all_lines += step2_file_listing(neh_df, octdl_df)

    neh_sample200 = sample_rows(neh_df, 200, SEED)
    octdl_sample200 = sample_rows(octdl_df, 200, SEED)
    all_lines += step3_image_properties(neh_sample200, octdl_sample200)

    print("Extracting RETOUCH central slices (SimpleITK, 112 volumes)...")
    retouch_volumes_df = find_retouch_volumes()
    retouch_records = assemble_retouch_slices(retouch_volumes_df)
    print(f"  {len(retouch_records)} slices")

    print("Extracting OCT5k central slices (PIL, volumes found at runtime)...")
    oct5k_volumes = find_oct5k_volumes()
    oct5k_records = assemble_oct5k_slices(oct5k_volumes)
    print(f"  {len(oct5k_records)} slices")

    print("Scanning OCTDL headers (width/height/mode/format, full population)...")
    octdl_hdr = exact_header_scan(octdl_df, "path", "disease")

    step4_lines, intensity_df = step4_intensity(
        neh_df, octdl_df, kermany_df,
        retouch_volumes_df, retouch_records, oct5k_volumes, oct5k_records, octdl_hdr)
    all_lines += step4_lines

    all_lines += step5_neh_structure(neh_df)
    all_lines += step5_octdl_structure(octdl_df)

    all_lines += step6_duplicates(neh_df, octdl_df)

    txt_path = TABDIR / "11_explore_new_data.txt"
    txt_path.write_text("\n".join(all_lines) + "\n")

    csv_df = build_csv_rows(neh_df, octdl_df, kermany_df, intensity_df,
                             retouch_volumes_df, retouch_records,
                             oct5k_volumes, oct5k_records, octdl_hdr)
    csv_path = TABDIR / "11_explore_new_data.csv"
    csv_df.to_csv(csv_path, index=False)

    print(f"Wrote {txt_path.relative_to(PROJECT)}")
    print(f"Wrote {csv_path.relative_to(PROJECT)}")

    p99s = {}
    for ds in ["NEH", "OCTDL", "Kermany-train", "RETOUCH-Spectralis",
               "RETOUCH-Cirrus", "RETOUCH-Topcon", "OCT5k"]:
        p99s[ds] = intensity_df.loc[intensity_df["dataset"] == ds, "p99"].mean()
    print("\np99: " + "  ".join(f"{k}={v:.4f}" for k, v in p99s.items()))


if __name__ == "__main__":
    main()
