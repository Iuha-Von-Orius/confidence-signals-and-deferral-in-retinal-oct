"""Audit RETOUCH and RASTI files, volume geometry and image formats."""

from paths import PROJECT, DATASET


import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError:
    sys.exit("SimpleITK not installed.  pip install SimpleITK")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed.  pip install Pillow")

DATASETS = {
    "RETOUCH": (DATASET / "RETOUCH"),
    "OCT5k": (DATASET / "Macular-Dataset-R.Rasti_old"),
}
TABDIR = PROJECT / "results" / "tables"

MAX_TREE_DEPTH = 4
MAX_ENTRIES_PER_DIR = 8
N_VOLUMES_TO_OPEN = 6

IMAGE_SLICE_RE = re.compile(r"^Image\s*(\d+)\.TIFF$", re.IGNORECASE)
IMAGE_SLICE_PNG_FALLBACK_RE = re.compile(r"^Image\s*(\d+)\.PNG$", re.IGNORECASE)

# Vendor names appear somewhere in the RETOUCH path. Matched case-insensitively
# because the challenge organisers were not consistent about capitalisation.
VENDORS = ["Cirrus", "Spectralis", "Topcon"]

IMAGE_EXT = {".mhd", ".mha", ".nii", ".gz", ".dcm", ".png", ".jpg", ".jpeg",
             ".tif", ".tiff", ".bmp", ".npy", ".raw", ".zraw"}


# Tree
def print_tree(root, prefix="", depth=0):
    """Directory listing, truncated in both depth and breadth."""
    if depth > MAX_TREE_DEPTH:
        print(f"{prefix}...")
        return
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        print(f"{prefix}[permission denied]")
        return

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file() and not e.name.startswith(".")]

    for i, d in enumerate(dirs[:MAX_ENTRIES_PER_DIR]):
        last = (i == len(dirs[:MAX_ENTRIES_PER_DIR]) - 1) and not files
        print(f"{prefix}{'└── ' if last else '├── '}{d.name}/")
        print_tree(d, prefix + ("    " if last else "│   "), depth + 1)
    if len(dirs) > MAX_ENTRIES_PER_DIR:
        print(f"{prefix}│   ... and {len(dirs) - MAX_ENTRIES_PER_DIR} more directories")

    for i, f in enumerate(files[:MAX_ENTRIES_PER_DIR]):
        size = f.stat().st_size
        unit = f"{size/1024**2:.1f} MB" if size > 1024**2 else f"{size/1024:.0f} KB"
        print(f"{prefix}{'└── ' if i == len(files[:MAX_ENTRIES_PER_DIR])-1 else '├── '}"
              f"{f.name}  ({unit})")
    if len(files) > MAX_ENTRIES_PER_DIR:
        print(f"{prefix}    ... and {len(files) - MAX_ENTRIES_PER_DIR} more files")


# Inventory
def inventory(root):
    """Count and size every file, grouped by extension."""
    by_ext = defaultdict(lambda: {"count": 0, "bytes": 0})
    suspicious, total_bytes, total_files = [], 0, 0

    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        size = p.stat().st_size
        ext = p.suffix.lower() or "(no extension)"
        by_ext[ext]["count"] += 1
        by_ext[ext]["bytes"] += size
        total_bytes += size
        total_files += 1

        # A zero-byte file, or a name carrying an error marker, is evidence of an
        # interrupted transfer rather than a quirk of the format.
        if size == 0 or "error" in p.name.lower() or "partial" in p.name.lower():
            suspicious.append((p, size))

    return by_ext, total_bytes, total_files, suspicious


def check_metaimage_pairs(root):
    """Check each MetaImage header against the presence and size of its binary file."""
    results = []
    for mhd in sorted(root.rglob("*.mhd")):
        entry = {"path": mhd, "header_ok": False, "data_file": None,
                 "data_exists": False, "data_bytes": 0, "expected_bytes": None,
                 "dims": None, "dtype": None}
        try:
            text = mhd.read_text(errors="replace")
            entry["header_ok"] = True

            fields = {}
            for line in text.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    fields[k.strip()] = v.strip()

            entry["dtype"] = fields.get("ElementType")
            if "DimSize" in fields:
                entry["dims"] = [int(x) for x in fields["DimSize"].split()]

            data_name = fields.get("ElementDataFile")
            if data_name and data_name != "LOCAL":
                data_path = mhd.parent / data_name
                entry["data_file"] = data_name
                entry["data_exists"] = data_path.exists()
                if entry["data_exists"]:
                    entry["data_bytes"] = data_path.stat().st_size

            # Only meaningful for uncompressed data; .zraw is compressed and will
            # legitimately be smaller than the product of the dimensions.
            bytes_per = {"MET_UCHAR": 1, "MET_CHAR": 1, "MET_USHORT": 2,
                         "MET_SHORT": 2, "MET_UINT": 4, "MET_INT": 4,
                         "MET_FLOAT": 4, "MET_DOUBLE": 8}.get(entry["dtype"])
            if entry["dims"] and bytes_per:
                entry["expected_bytes"] = int(np.prod(entry["dims"])) * bytes_per
        except Exception as e:
            entry["error"] = str(e)[:120]
        results.append(entry)
    return results


def open_volumes(paths, n):
    """Read a sample of volumes and report their geometry and intensity range."""
    if not paths:
        return []
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)
    out = []
    for i in sorted(set(idx)):
        p = paths[i]
        rec = {"path": p, "ok": False}
        try:
            img = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(img)
            rec.update({
                "ok": True,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "spacing": tuple(round(s, 4) for s in img.GetSpacing()),
                "n_slices": arr.shape[0] if arr.ndim == 3 else 1,
            })
        except Exception as e:
            rec["error"] = str(e)[:160]
        out.append(rec)
    return out


def inspect_rasti_volumes(root):
    """Report slice counts, formats and numbering for each RASTI volume."""
    rows = []
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for vol_dir in sorted(d for d in class_dir.iterdir() if d.is_dir()):
            all_files = [p for p in vol_dir.rglob("*")
                         if p.is_file() and not p.name.startswith(".")]

            slices = []
            for p in all_files:
                m = IMAGE_SLICE_RE.match(p.name)
                if m:
                    slices.append((int(m.group(1)), p))
            slices.sort(key=lambda t: t[0])

            n_png_total = sum(1 for p in all_files if p.suffix.lower() == ".png")

            used_png_fallback = False
            if not slices:
                for p in all_files:
                    m = IMAGE_SLICE_PNG_FALLBACK_RE.match(p.name)
                    if m:
                        slices.append((int(m.group(1)), p))
                slices.sort(key=lambda t: t[0])
                used_png_fallback = bool(slices)

            n_excluded_png = 0 if used_png_fallback else n_png_total
            n_other = len(all_files) - len(slices) - n_excluded_png

            dims, open_ok = {}, True
            for idx, p in slices:
                try:
                    with Image.open(p) as im:
                        dims[idx] = im.size
                except Exception:
                    open_ok = False

            row = {
                "dataset": "OCT5k", "class": class_dir.name,
                "volume_id": vol_dir.name, "path": str(vol_dir.relative_to(root)),
                "n_slices": len(slices),
                "slice_format": "PNG" if used_png_fallback else "TIFF",
                "used_png_fallback": used_png_fallback,
                "n_excluded_png": n_excluded_png, "n_excluded_other": n_other,
                "open_ok": open_ok,
            }
            if dims:
                indices = sorted(dims)
                row["slice_index_min"] = indices[0]
                row["slice_index_max"] = indices[-1]
                row["slice_index_gap"] = (indices[-1] - indices[0] + 1) != len(indices)

                size_counts = Counter(dims.values())
                mode_w, mode_h = size_counts.most_common(1)[0][0]
                row["width_mode"], row["height_mode"] = mode_w, mode_h
                row["dims_consistent"] = len(size_counts) == 1

                if 0 in dims:
                    row["image0_present"] = True
                    row["image0_width"], row["image0_height"] = dims[0]
                else:
                    row["image0_present"] = False
            rows.append(row)
    return rows


def guess_vendor(path):
    s = str(path).lower()
    for v in VENDORS:
        if v.lower() in s:
            return v
    return "unknown"


# Report
def audit(name, root):
    print("\n" + "=" * 78)
    print(f"  {name}   {root}")
    print("=" * 78)

    if not root.exists():
        print("  [ERROR] path does not exist")
        return []

    print("\n--- Directory structure ---")
    print(f"{root.name}/")
    print_tree(root)

    print("\n--- File inventory ---")
    by_ext, total_bytes, total_files, suspicious = inventory(root)
    print(f"{'extension':>16} {'count':>8} {'total size':>14}")
    print("-" * 42)
    for ext, v in sorted(by_ext.items(), key=lambda kv: -kv[1]["bytes"]):
        gb = v["bytes"] / 1024**3
        size = f"{gb:.2f} GB" if gb >= 1 else f"{v['bytes']/1024**2:.1f} MB"
        print(f"{ext:>16} {v['count']:>8,} {size:>14}")
    print("-" * 42)
    print(f"{'TOTAL':>16} {total_files:>8,} {total_bytes/1024**3:>11.2f} GB")

    if suspicious:
        print(f"\n  [WARNING] {len(suspicious)} suspicious files "
              "(zero bytes, or 'error'/'partial' in the name):")
        for p, s in suspicious[:12]:
            print(f"    {s:>10,} bytes  {p.relative_to(root)}")
        if len(suspicious) > 12:
            print(f"    ... and {len(suspicious) - 12} more")
    else:
        print("\n  No zero-byte or error-named files.")

    # MetaImage integrity
    pairs = check_metaimage_pairs(root)
    rows = []
    if pairs:
        print(f"\n--- MetaImage headers: {len(pairs)} .mhd files ---")
        missing = [p for p in pairs if p["data_file"] and not p["data_exists"]]
        short = [p for p in pairs
                 if p["data_exists"] and p["expected_bytes"]
                 and p["data_bytes"] < p["expected_bytes"] * 0.95
                 and not str(p["data_file"]).endswith(".zraw")]

        print(f"  binary present:  {sum(p['data_exists'] for p in pairs)}/{len(pairs)}")
        if missing:
            print(f"  [ERROR] {len(missing)} headers whose binary is missing:")
            for p in missing[:8]:
                print(f"    {p['path'].relative_to(root)}  ->  {p['data_file']}")
        if short:
            print(f"  [ERROR] {len(short)} binaries shorter than the header implies:")
            for p in short[:8]:
                print(f"    {p['path'].relative_to(root)}  "
                      f"{p['data_bytes']:,} of {p['expected_bytes']:,} bytes "
                      f"({p['data_bytes']/p['expected_bytes']*100:.0f}%)")
        if not missing and not short:
            print("  All binaries present and of the expected size.")

        dim_counter = Counter(tuple(p["dims"]) for p in pairs if p["dims"])
        if dim_counter:
            print("\n  Volume dimensions from headers (DimSize = width height slices):")
            for d, c in dim_counter.most_common(10):
                print(f"    {str(d):>24}  ×{c}")

        type_counter = Counter(p["dtype"] for p in pairs if p["dtype"])
        if type_counter:
            print("\n  Element types:")
            for t, c in type_counter.most_common():
                print(f"    {t:>14}  ×{c}")

        vendor_counter = Counter(guess_vendor(p["path"]) for p in pairs)
        print("\n  Grouped by vendor keyword in the path:")
        for v, c in vendor_counter.most_common():
            print(f"    {v:>12}  {c} volumes")

        for p in pairs:
            rows.append({
                "dataset": name,
                "path": str(p["path"].relative_to(root)),
                "vendor": guess_vendor(p["path"]),
                "dims": str(p["dims"]),
                "dtype": p["dtype"],
                "data_exists": p["data_exists"],
                "data_mb": round(p["data_bytes"] / 1024**2, 2),
            })

        # The decisive check
        print(f"\n--- Opening {N_VOLUMES_TO_OPEN} volumes (the real integrity test) ---")
        for rec in open_volumes([p["path"] for p in pairs], N_VOLUMES_TO_OPEN):
            rel = rec["path"].relative_to(root)
            if rec["ok"]:
                print(f"  OK   {rel}")
                print(f"       shape {rec['shape']}  dtype {rec['dtype']}  "
                      f"range [{rec['min']:.0f}, {rec['max']:.0f}]  "
                      f"mean {rec['mean']:.1f}")
                print(f"       {rec['n_slices']} slices,  spacing {rec['spacing']}")
            else:
                print(f"  FAIL {rel}")
                print(f"       {rec.get('error')}")

    # OCT5k (Rasti): structured, one-row-per-volume audit
    if name == "OCT5k":
        rasti_rows = inspect_rasti_volumes(root)
        rows += rasti_rows
        rdf = pd.DataFrame(rasti_rows)

        print(f"\n--- OCT5k: {len(rdf)} volumes "
              f"({rdf['n_slices'].sum():,} slices, TIFF or PNG-fallback) ---")
        print("\n  Volumes and slices per class:")
        for cls, sub in rdf.groupby("class"):
            print(f"    {cls:>8}  {len(sub):>3} volumes  "
                  f"{sub['n_slices'].sum():>5,} slices  "
                  f"(min {sub['n_slices'].min()}, max {sub['n_slices'].max()}, "
                  f"mean {sub['n_slices'].mean():.1f})")

        fallback = rdf[rdf["used_png_fallback"] == True]
        print(f"\n  PNG fallback used (volume had zero TIFF slices): "
              f"{len(fallback)} volumes")
        for _, r in fallback.iterrows():
            print(f"    {r['path']}  {r['n_slices']} slices recovered from PNG")

        excluded_png = rdf["n_excluded_png"].sum()
        excluded_other = rdf["n_excluded_other"].sum()
        print(f"\n  Excluded (do not match the slice pattern, and not a "
              f"PNG-fallback volume): {excluded_png} PNG files, "
              f"{excluded_other} other files.")

        gaps = rdf[rdf["slice_index_gap"] == True]
        print(f"\n  Slice-index gaps (max-min+1 != n_slices): {len(gaps)} volumes")
        for _, r in gaps.head(10).iterrows():
            print(f"    {r['path']}  indices {r['slice_index_min']}-"
                  f"{r['slice_index_max']}, {r['n_slices']} files")

        not_ok = rdf[~rdf["open_ok"]]
        if len(not_ok):
            print(f"\n  [ERROR] {len(not_ok)} volumes had a slice that failed to "
                  "open:")
            for _, r in not_ok.iterrows():
                print(f"    {r['path']}")
        else:
            print("\n  Every matched slice opened cleanly with PIL.")

        inconsistent = rdf[rdf["dims_consistent"] == False]
        print(f"\n  Volumes where not every matched slice shares one size: "
              f"{len(inconsistent)} of {len(rdf)}")

        with_image0 = rdf[rdf["image0_present"] == True]
        if len(with_image0):
            image0_dims = Counter(zip(with_image0["image0_width"],
                                       with_image0["image0_height"]))
            mode_dims = Counter(zip(rdf["width_mode"], rdf["height_mode"]))
            print(f"\n  Image 0 present in {len(with_image0)}/{len(rdf)} volumes. "
                  f"Its (width, height): {dict(image0_dims.most_common(5))}")
            print(f"  Modal (width, height) across all matched slices per volume: "
                  f"{dict(mode_dims.most_common(5))}")

    # Plain image files, generic fallback for any other dataset
    else:
        plain = [p for p in root.rglob("*")
                 if p.is_file() and p.suffix.lower() in
                 {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]
        if plain:
            print(f"\n--- {len(plain):,} ordinary image files ---")
            print("  Examples:")
            for p in plain[:6]:
                print(f"    {p.relative_to(root)}  ({p.stat().st_size/1024:.0f} KB)")
            print("  Sizes and modes from a sample:")
            for p in plain[:5]:
                with Image.open(p) as im:
                    print(f"    {im.size}  mode {im.mode}   {p.name}")

    return rows


def main():
    TABDIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, root in DATASETS.items():
        all_rows += audit(name, root)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(TABDIR / "00b_ood_data_inventory.csv", index=False)
        print(f"\nInventory written to "
              f"{(TABDIR / '00b_ood_data_inventory.csv').relative_to(PROJECT)}")

    print("\n" + "=" * 78)
    print("What to read off this:")
    print("  - Any FAIL when opening volumes means the download is incomplete.")
    print("    Re-download before doing anything else.")
    print("  - Slice count per volume determines how many central slices to keep.")
    print("  - Intensity range decides whether these can share Kermany's")
    print("    preprocessing or need rescaling first (Kermany is 8-bit, 0-255).")
    print("  - Volume counts determine the per-vendor evaluation sample size.")
    print("  - OCT5k: this audit reports facts only (slice counts, Image 0's")
    print("    dimensions against the rest, gaps, open failures) -- whether to")
    print("    use Image 0 as a slice is a decision for 08a's reading strategy,")
    print("    not made here.")


if __name__ == "__main__":
    main()
