"""Check OCTDL alpha channels and NEH duplicate metadata entries.

Source images and metadata are read-only. Reports are written to results/tables.
"""

from paths import PROJECT, DATASET, read_csv


import numpy as np
import pandas as pd
from PIL import Image

TABDIR = PROJECT / "results" / "tables"

NEH_ROOT = (DATASET / "Tehran_Labeled Retinal Optical Coherence Tomography "
                      "Dataset for Classification of Normal, Drusen, and "
                      "CNV Cases")
NEH_CSV = NEH_ROOT / "data_information.csv"

OCTDL_ROOT = (DATASET / "OCTDL Optical Coherence Tomography Dataset for "
                        "Image-Based Deep Learning Methods")
OCTDL_CSV = OCTDL_ROOT / "OCTDL_labels.csv"
OCTDL_IMG_ROOT = OCTDL_ROOT / "OCTDL"

# Published NEH worst-case-version figures (read_data.py's "Option 2"),
# the same reference 11_explore_new_data.py checked against.
NEH_WORSTCASE_PUBLISHED = {"CNV": 3240, "DRUSEN": 3742, "NORMAL": 5667}
NEH_TOTAL_12649 = 12649
NEH_DISK_TOTAL = 16803


def section(title):
    bar = "=" * 78
    return [bar, title, bar]


def subsection(title):
    return ["", f"--- {title} ---"]


# Part 1: OCTDL format audit
def part1_octdl_format_audit():
    lines = section("PART 1: OCTDL IMAGE FORMAT AUDIT (all 2,064 images, no sampling)")

    df = read_csv(OCTDL_CSV)
    modes, formats = [], []
    for _, row in df.iterrows():
        p = OCTDL_IMG_ROOT / row["disease"] / f"{row['file_name']}.jpg"
        with Image.open(p) as im:
            modes.append(im.mode)
            formats.append(im.format)
    df["mode"] = modes
    df["format"] = formats

    lines += subsection("1. PIL Image.mode distribution by disease")
    mode_ct = pd.crosstab(df["disease"], df["mode"])
    lines.append(mode_ct.to_string())

    lines += subsection("2. PIL Image.format distribution by disease (actual container format)")
    fmt_ct = pd.crosstab(df["disease"], df["format"])
    lines.append(fmt_ct.to_string())

    lines += subsection("3. Files whose format != \"JPEG\" despite the .jpg extension")
    non_jpeg = df[df["format"] != "JPEG"][["file_name", "disease", "format", "mode"]].copy()
    lines.append(f"Total: {len(non_jpeg)} of {len(df)}")
    shown = non_jpeg.head(50)
    lines.append(f"Showing {len(shown)} of {len(non_jpeg)}:")
    lines.append(shown.to_string(index=False))

    lines += subsection("4. Alpha channel stats for all RGBA images")
    rgba_rows = df[df["mode"] == "RGBA"]
    amins, amaxs, ameans = [], [], []
    non_opaque = []
    for _, row in rgba_rows.iterrows():
        p = OCTDL_IMG_ROOT / row["disease"] / f"{row['file_name']}.jpg"
        with Image.open(p) as im:
            arr = np.asarray(im)
        alpha = arr[..., 3]
        amin, amax, amean = int(alpha.min()), int(alpha.max()), float(alpha.mean())
        amins.append(amin); amaxs.append(amax); ameans.append(amean)
        if amin < 255:
            non_opaque.append((row["file_name"], row["disease"], amin, amax, amean))
    lines.append(f"n RGBA images: {len(rgba_rows)}")
    lines.append(f"alpha min  -- across-image min={min(amins)}, max={max(amins)}, mean={np.mean(amins):.4f}")
    lines.append(f"alpha max  -- across-image min={min(amaxs)}, max={max(amaxs)}, mean={np.mean(amaxs):.4f}")
    lines.append(f"alpha mean -- across-image min={min(ameans):.4f}, max={max(ameans):.4f}, mean={np.mean(ameans):.4f}")
    lines.append(f"Images with alpha min < 255 (not fully opaque): {len(non_opaque)}")

    lines += subsection("5. Three-way grayscale comparison for non-opaque RGBA images")
    if not non_opaque:
        lines.append("Not applicable: 0 images have alpha min < 255, so there is no "
                     "non-opaque RGBA image to compare compositing methods on.")
    else:
        diffs_wb, diffs_wk, diffs_bk = [], [], []
        for file_name, disease, *_ in non_opaque:
            p = OCTDL_IMG_ROOT / disease / f"{file_name}.jpg"
            with Image.open(p) as im:
                direct_l = np.asarray(im.convert("L")).astype(np.float64)
                white_bg = Image.new("RGB", im.size, (255, 255, 255))
                white_bg.paste(im, mask=im.split()[3])
                white_l = np.asarray(white_bg.convert("L")).astype(np.float64)
                black_bg = Image.new("RGB", im.size, (0, 0, 0))
                black_bg.paste(im, mask=im.split()[3])
                black_l = np.asarray(black_bg.convert("L")).astype(np.float64)
            diffs_wb.append(np.abs(direct_l - white_l).mean())
            diffs_wk.append(np.abs(direct_l - black_l).mean())
            diffs_bk.append(np.abs(white_l - black_l).mean())
        lines.append(f"mean |direct convert(L) - white-composite then L| : {np.mean(diffs_wb):.4f}")
        lines.append(f"mean |direct convert(L) - black-composite then L| : {np.mean(diffs_wk):.4f}")
        lines.append(f"mean |white-composite then L - black-composite then L| : {np.mean(diffs_bk):.4f}")

    return lines, df, non_jpeg


# Part 2: NEH duplicate-row audit
def part2_neh_duplicate_audit():
    lines = section("PART 2: NEH INDEX-FILE DUPLICATE-ROW AUDIT")

    df = read_csv(NEH_CSV)
    cols = ["Patient ID", "Class", "Eye", "B-scan", "Label", "Directory"]
    df["dir_norm"] = df["Directory"].str.lower()

    lines += subsection("1. Exact-duplicate rows (identical Directory string)")
    exact_dup_rows = df[df["Directory"].duplicated(keep=False)].sort_values("Directory")
    lines.append(f"{len(exact_dup_rows)} rows, in "
                 f"{exact_dup_rows['Directory'].nunique()} groups of 2")
    all_six_identical = True
    for d, g in exact_dup_rows.groupby("Directory"):
        if not (g[cols].nunique() == 1).all():
            all_six_identical = False
    lines.append(f"All six columns identical within every group: {all_six_identical}")
    lines.append(exact_dup_rows[cols].to_string())

    lines += subsection("2. Case-variant groups (Directory differs only in letter case)")
    case_variant_rows = df.groupby("dir_norm").filter(
        lambda g: g["Directory"].nunique() > 1)
    n_case_groups = case_variant_rows["dir_norm"].nunique()
    lines.append(f"{len(case_variant_rows)} rows, in {n_case_groups} groups of 2")
    for dl, g in case_variant_rows.groupby("dir_norm"):
        label_consistent = g["Label"].nunique() == 1
        class_consistent = g["Class"].nunique() == 1
        lines.append(f"  group (normalised={dl}): Label consistent={label_consistent}  "
                     f"Class consistent={class_consistent}")
        lines.append("    " + g[cols].to_string(index=False).replace("\n", "\n    "))

    lines += subsection("3. (written to CSV, tagged by category -- see 12_data_integrity_checks.csv)")
    lines.append("neh_exact_duplicate: 12 groups / 24 rows.  "
                 "neh_case_variant: 7 groups / 14 rows.")

    lines += subsection("4. Deduplication rule (stated precisely, for reproduction)")
    lines.append('Normalise: dir_norm = Directory.str.lower() (case-fold only; no '
                 'other change to the string). Deduplicate: pandas '
                 'drop_duplicates(subset="dir_norm", keep="first"), i.e. for every '
                 'group of rows sharing a normalised Directory, keep the row that '
                 'appears first in data_information.csv\'s own top-to-bottom row '
                 'order (the CSV\'s original index), discard the rest. No other '
                 'ordering (e.g. by Patient ID, B-scan) is applied.')

    deduped = df.drop_duplicates(subset="dir_norm", keep="first")

    lines += subsection("5. Per-class counts after deduplication, vs disk total")
    dedup_counts = deduped["Class"].value_counts()
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        lines.append(f"  Class={c:<8s} n={int(dedup_counts.get(c, 0))}")
    total_deduped = len(deduped)
    lines.append(f"  Total: {total_deduped}   disk total (from 11_explore_new_data.py's "
                 f"independent os.walk): {NEH_DISK_TOTAL}   match={total_deduped == NEH_DISK_TOTAL}")

    lines += subsection("6. Label==Class filter re-run on deduplicated data, vs published worst-case")
    same_deduped = deduped[deduped["Class"] == deduped["Label"]]
    filt_counts = same_deduped.groupby("Class").size()
    for c in ["CNV", "DRUSEN", "NORMAL"]:
        computed = int(filt_counts.get(c, 0))
        pub = NEH_WORSTCASE_PUBLISHED[c]
        lines.append(f"  {c:<8s} computed={computed:<6d} published={pub:<6d} "
                     f"match={computed==pub}  diff={computed - pub}")
    total_filt_deduped = len(same_deduped)
    lines.append(f"  Total    computed={total_filt_deduped:<6d} published={NEH_TOTAL_12649:<6d} "
                 f"match={total_filt_deduped==NEH_TOTAL_12649}  diff={total_filt_deduped - NEH_TOTAL_12649}")

    return lines, exact_dup_rows, case_variant_rows


def main():
    all_lines = []

    part1_lines, octdl_df, non_jpeg_df = part1_octdl_format_audit()
    all_lines += part1_lines

    part2_lines, exact_dup_rows, case_variant_rows = part2_neh_duplicate_audit()
    all_lines += part2_lines

    txt_path = TABDIR / "12_data_integrity_checks.txt"
    txt_path.write_text("\n".join(all_lines) + "\n")

    # CSV: union schema, "check" column distinguishes the three record kinds
    csv_rows = []
    for _, row in non_jpeg_df.iterrows():
        csv_rows.append(dict(
            check="octdl_non_jpeg", file_name=row["file_name"], disease=row["disease"],
            format=row["format"], mode=row["mode"],
            patient_id=None, cls=None, eye=None, bscan=None, label=None,
            directory=None, group_id=None,
        ))

    for gi, (d, g) in enumerate(exact_dup_rows.groupby("Directory"), start=1):
        for _, row in g.iterrows():
            csv_rows.append(dict(
                check="neh_exact_duplicate", file_name=None, disease=None,
                format=None, mode=None,
                patient_id=row["Patient ID"], cls=row["Class"], eye=row["Eye"],
                bscan=row["B-scan"], label=row["Label"], directory=row["Directory"],
                group_id=f"exact_{gi}",
            ))

    for gi, (dl, g) in enumerate(case_variant_rows.groupby("dir_norm"), start=1):
        for _, row in g.iterrows():
            csv_rows.append(dict(
                check="neh_case_variant", file_name=None, disease=None,
                format=None, mode=None,
                patient_id=row["Patient ID"], cls=row["Class"], eye=row["Eye"],
                bscan=row["B-scan"], label=row["Label"], directory=row["Directory"],
                group_id=f"case_{gi}",
            ))

    csv_df = pd.DataFrame(csv_rows, columns=[
        "check", "file_name", "disease", "format", "mode",
        "patient_id", "cls", "eye", "bscan", "label", "directory", "group_id"])
    csv_df = csv_df.rename(columns={"cls": "class"})
    csv_path = TABDIR / "12_data_integrity_checks.csv"
    csv_df.to_csv(csv_path, index=False)

    print(f"Wrote {txt_path.relative_to(PROJECT)}")
    print(f"Wrote {csv_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
