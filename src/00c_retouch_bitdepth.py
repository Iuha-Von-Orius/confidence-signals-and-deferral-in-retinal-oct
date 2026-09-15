"""Inspect RETOUCH bit depth and compare candidate intensity conversions.

The final pipeline uses fixed 16-bit to 8-bit scaling in step 08a.
"""

from paths import PROJECT, DATASET, read_csv

import random
from collections import Counter
from functools import reduce
from math import gcd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image

RETOUCH_ROOT = (DATASET / "RETOUCH/RETOUCH")
SPLITS = PROJECT / "splits"
RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "00c_retouch_bitdepth"
TABDIR = RESULTS / "tables"

SEED = 42
N_KERMANY_SAMPLE = 300
N_SPECTRALIS_FOR_HIST = 8
N_UCHAR_REFERENCE = 6

VENDORS = ["Cirrus", "Spectralis", "Topcon"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})

STRATEGY_COLOURS = {
    "Kermany (8-bit JPEG)": "#2c3e50",
    "naive /257": "#d62728",
    "per-volume min-max": "#ff7f0e",
    "global 1st-99th pct": "#2ca02c",
}


# Discovery
def guess_vendor(path):
    s = str(path).lower()
    for v in VENDORS:
        if v.lower() in s:
            return v
    return "unknown"


def find_oct_volumes():
    """Find OCT MetaImage headers and read vendor and ElementType metadata."""
    rows = []
    for mhd in sorted(RETOUCH_ROOT.rglob("oct.mhd")):
        fields = {}
        for line in mhd.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k.strip()] = v.strip()
        rows.append({
            "path": mhd,
            "vendor": guess_vendor(mhd),
            "set": "train" if "TrainingSet" in str(mhd) else "test",
            "element_type": fields.get("ElementType", "?"),
        })
    return rows


# Per-volume stats
def volume_stats(mhd_path):
    """Measure intensity range, distinct values and their greatest common divisor."""
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    uniq = np.unique(arr)
    nonzero = uniq[uniq != 0]
    g = int(reduce(gcd, [int(x) for x in nonzero])) if len(nonzero) else 0

    theoretical_max = 65535 if arr.dtype == np.uint16 else 255
    frac_at_max = float((arr == arr.max()).mean())
    frac_top_1pct = float((arr >= theoretical_max * 0.99).mean())

    pcts = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n_unique": int(len(uniq)),
        "gcd_nonzero": g,
        "dtype": str(arr.dtype),
        "min": int(arr.min()), "max": int(arr.max()), "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p1": pcts[0], "p5": pcts[1], "p25": pcts[2], "p50": pcts[3],
        "p75": pcts[4], "p95": pcts[5], "p99": pcts[6],
        "frac_at_max": frac_at_max, "frac_top_1pct": frac_top_1pct,
    }


def central_slice(mhd_path):
    """Return the middle B-scan as a two-dimensional array."""
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    return arr[arr.shape[0] // 2]


# Kermany reference
def load_kermany_sample(n, seed=SEED):
    """Sample Kermany training images for reference intensity statistics."""
    df = read_csv(SPLITS / "train.csv")
    sample = df.sample(n=n, random_state=seed)
    pixels = []
    for fp in sample["filepath"]:
        with Image.open(fp) as im:
            pixels.append(np.asarray(im.convert("L")).ravel())
    return np.concatenate(pixels)


# Rescaling candidates
def rescale_naive(arr):
    """Map the theoretical uint16 range to uint8 using fixed linear scaling."""
    return (arr.astype(np.float64) / 65535 * 255).clip(0, 255).astype(np.uint8)


def rescale_minmax(arr):
    """Scale a volume's observed minimum and maximum to 0 and 255."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float64) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def rescale_percentile(arr, lo, hi):
    """Clip to a shared intensity window and scale to 0-255."""
    clipped = np.clip(arr.astype(np.float64), lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


# Figures
def fig_elementtype_by_vendor(df):
    counts = df.groupby(["vendor", "element_type"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    counts.plot(kind="bar", ax=ax, color=["#1f77b4", "#d62728"], alpha=0.85)
    ax.set_ylabel("Volumes")
    ax.set_title("oct.mhd ElementType by vendor")
    ax.set_xticklabels(counts.index, rotation=0)
    ax.legend(title="ElementType")
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_elementtype_by_vendor.png")
    plt.close(fig)


def fig_unique_values_distribution(stats_df):
    """Plot distinct-value counts against the 256-value limit of uint8."""
    ushort = stats_df[stats_df.element_type == "MET_USHORT"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(ushort["n_unique"], bins=20, color="#2c3e50", alpha=0.85)
    ax.axvline(256, color="#d62728", ls="--", lw=2,
               label="256 -- expected if 8-bit data in a 16-bit container")
    ax.set_xlabel("Distinct pixel values in the volume")
    ax.set_ylabel("Number of volumes")
    ax.set_title("Distinct-value count, all 38 MET_USHORT volumes")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_unique_values_distribution.png")
    plt.close(fig)


def fig_raw_intensity_histograms(kermany_px, spectralis_slices, uchar_slices):
    """Plot raw intensity distributions on separate axes for each data type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(kermany_px, bins=80, color=STRATEGY_COLOURS["Kermany (8-bit JPEG)"], alpha=0.85)
    axes[0].set_title(f"Kermany (n={N_KERMANY_SAMPLE} images)\n8-bit, native scale")
    axes[0].set_xlabel("Pixel value (0-255)")

    spec_px = np.concatenate([s.ravel() for s in spectralis_slices])
    axes[1].hist(spec_px, bins=80, color="#d62728", alpha=0.85)
    axes[1].set_title(f"Spectralis (n={len(spectralis_slices)} volumes)\n16-bit, native scale")
    axes[1].set_xlabel("Pixel value (0-65535)")

    uchar_px = np.concatenate([s.ravel() for s in uchar_slices])
    axes[2].hist(uchar_px, bins=80, color="#1f77b4", alpha=0.85)
    axes[2].set_title(f"Topcon + Cirrus (n={len(uchar_slices)} volumes)\n8-bit, native scale")
    axes[2].set_xlabel("Pixel value (0-255)")

    for ax in axes:
        ax.set_ylabel("Pixel count")
    fig.suptitle("Raw intensity distributions, each on its own declared scale")
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_raw_intensity_histograms.png")
    plt.close(fig)


def fig_saturation_check(stats_df):
    ushort = stats_df[stats_df.element_type == "MET_USHORT"].sort_values("frac_top_1pct")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ushort))
    ax.bar(x, ushort["frac_top_1pct"] * 100, color="#8b1a1a", alpha=0.8)
    ax.set_xticks([])
    ax.set_xlabel("Spectralis volumes (sorted)")
    ax.set_ylabel("% of pixels in top 1% of the 16-bit range")
    ax.set_title("Sensor saturation check -- a spike here would mean the maximum\n"
                 "reflects clipping, not a genuine bright feature")
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_saturation_check.png")
    plt.close(fig)


def fig_percentile_profile(stats_df, kermany_px):
    ushort = stats_df[stats_df.element_type == "MET_USHORT"]
    pct_cols = ["p1", "p5", "p25", "p50", "p75", "p95", "p99"]
    pct_labels = [1, 5, 25, 50, 75, 95, 99]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    for _, row in ushort.iterrows():
        ax.plot(pct_labels, [row[c] for c in pct_cols], "-", lw=1.0,
                color="#d62728", alpha=0.35)
    ax.plot(pct_labels, ushort[pct_cols].mean(), "o-", lw=2.5, ms=6,
            color="#2c3e50", label="mean across 38 volumes")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Pixel value (16-bit)")
    ax.set_title("Spectralis percentile profile, one line per volume")
    ax.legend(fontsize=8.5)

    kermany_pcts = np.percentile(kermany_px, pct_labels)
    ax2.plot(pct_labels, kermany_pcts, "o-", lw=2.5, ms=6,
             color=STRATEGY_COLOURS["Kermany (8-bit JPEG)"], label="Kermany (8-bit)")
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("Pixel value (0-255)")
    ax2.set_title("Kermany percentile profile, for reference")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_percentile_profile.png")
    plt.close(fig)


def fig_rescale_comparison_histograms(kermany_px, spectralis_slices, window):
    lo, hi = window
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(kermany_px, bins=80, density=True, alpha=0.45,
            color=STRATEGY_COLOURS["Kermany (8-bit JPEG)"],
            label="Kermany (8-bit JPEG)")

    all_raw = np.concatenate([s.ravel() for s in spectralis_slices])
    naive = rescale_naive(all_raw)
    ax.hist(naive, bins=80, density=True, alpha=0.45,
            color=STRATEGY_COLOURS["naive /257"], label="naive /257")

    minmax_each = np.concatenate([rescale_minmax(s).ravel() for s in spectralis_slices])
    ax.hist(minmax_each, bins=80, density=True, alpha=0.45,
            color=STRATEGY_COLOURS["per-volume min-max"], label="per-volume min-max")

    pct_scaled = rescale_percentile(all_raw, lo, hi)
    ax.hist(pct_scaled, bins=80, density=True, alpha=0.45,
            color=STRATEGY_COLOURS["global 1st-99th pct"],
            label=f"global {lo:.0f}-{hi:.0f} pct clip")

    ax.set_xlabel("Rescaled pixel value (0-255)")
    ax.set_ylabel("Density")
    ax.set_title("Candidate rescalings of the Spectralis sample against Kermany")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_rescale_comparison_histograms.png")
    plt.close(fig)


def fig_rescale_comparison_images(example_slice, window):
    lo, hi = window
    variants = {
        "raw (16-bit, matplotlib auto-scaled)": example_slice,
        "naive /257": rescale_naive(example_slice),
        "per-volume min-max": rescale_minmax(example_slice),
        f"global {lo:.0f}-{hi:.0f} pct clip": rescale_percentile(example_slice, lo, hi),
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, im) in zip(axes, variants.items()):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")
    fig.suptitle("One Spectralis B-scan under each rescaling")
    fig.tight_layout()
    fig.savefig(FIGDIR / "00c_rescale_comparison_images.png")
    plt.close(fig)


# Main
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    print("Discovering oct.mhd volumes and their declared ElementType...")
    volumes = find_oct_volumes()
    df = pd.DataFrame(volumes)
    print(df.groupby(["vendor", "element_type"]).size().to_string())

    elementtype_summary = (df.groupby(["vendor", "set", "element_type"])
                            .size().reset_index(name="n_volumes"))
    elementtype_summary.to_csv(TABDIR / "00c_elementtype_summary.csv", index=False)
    fig_elementtype_by_vendor(df)

    print("\nComputing per-volume stats for all 112 oct.mhd volumes "
          "(unique values, GCD, percentiles, saturation)...")
    stats_rows = []
    for _, row in df.iterrows():
        s = volume_stats(row["path"])
        stats_rows.append({"path": str(row["path"]), "vendor": row["vendor"],
                            "set": row["set"], "element_type": row["element_type"],
                            **s})
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(TABDIR / "00c_volume_stats.csv", index=False)

    ushort = stats_df[stats_df.element_type == "MET_USHORT"]
    print(f"\nMET_USHORT volumes: {len(ushort)}")
    print(f"  unique values: min {ushort.n_unique.min()}, max {ushort.n_unique.max()}, "
          f"mean {ushort.n_unique.mean():.0f}")
    print(f"  GCD of nonzero values: {sorted(ushort.gcd_nonzero.unique())}")
    if ushort.n_unique.min() > 1000 and set(ushort.gcd_nonzero.unique()) == {1}:
        print("  -> genuine high-dynamic-range data, NOT 8-bit repackaged in a "
              "wider container (256 distinct values, GCD>=256 would indicate that).")
    fig_unique_values_distribution(stats_df)
    fig_saturation_check(stats_df)

    print("\nLoading Kermany reference sample...")
    kermany_px = load_kermany_sample(N_KERMANY_SAMPLE)
    print(f"  {N_KERMANY_SAMPLE} images, {len(kermany_px):,} pixels, "
          f"mean {kermany_px.mean():.1f}, range [{kermany_px.min()},{kermany_px.max()}]")

    fig_percentile_profile(stats_df, kermany_px)

    print(f"\nLoading central slices: {N_SPECTRALIS_FOR_HIST} Spectralis + "
          f"{N_UCHAR_REFERENCE} Topcon/Cirrus...")
    spectralis_paths = df[df.vendor == "Spectralis"]["path"].tolist()
    uchar_paths = df[df.vendor != "Spectralis"]["path"].tolist()
    rng = random.Random(SEED)
    spec_sample_paths = rng.sample(spectralis_paths, N_SPECTRALIS_FOR_HIST)
    uchar_sample_paths = rng.sample(uchar_paths, N_UCHAR_REFERENCE)
    spectralis_slices = [central_slice(p) for p in spec_sample_paths]
    uchar_slices = [central_slice(p) for p in uchar_sample_paths]

    fig_raw_intensity_histograms(kermany_px, spectralis_slices, uchar_slices)

    # Window for the percentile-clip candidate: pooled 1st/99th percentile
    # across the Spectralis sample, not any single volume's own extremes.
    pooled = np.concatenate([s.ravel() for s in spectralis_slices])
    window = tuple(np.percentile(pooled, [1, 99]))
    print(f"\nPooled 1st-99th percentile window across the Spectralis sample: "
          f"{window[0]:.0f}-{window[1]:.0f}")

    fig_rescale_comparison_histograms(kermany_px, spectralis_slices, window)
    fig_rescale_comparison_images(spectralis_slices[0], window)

    # Rescale comparison table
    rows = [{"variant": "Kermany (8-bit JPEG)", "mean": kermany_px.mean(),
             "std": kermany_px.std(),
             **{f"p{p}": v for p, v in zip([1, 5, 25, 50, 75, 95, 99],
                                            np.percentile(kermany_px, [1, 5, 25, 50, 75, 95, 99]))}}]
    for name, arr in [
        ("naive /257", rescale_naive(pooled)),
        ("per-volume min-max", np.concatenate([rescale_minmax(s).ravel() for s in spectralis_slices])),
        (f"global {window[0]:.0f}-{window[1]:.0f} pct clip", rescale_percentile(pooled, *window)),
    ]:
        rows.append({"variant": name, "mean": arr.mean(), "std": arr.std(),
                     **{f"p{p}": v for p, v in zip([1, 5, 25, 50, 75, 95, 99],
                                                    np.percentile(arr, [1, 5, 25, 50, 75, 95, 99]))}})
    rescale_df = pd.DataFrame(rows)
    rescale_df.to_csv(TABDIR / "00c_rescale_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print("Rescale comparison (0-255 scale, all rows):\n")
    print(rescale_df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print(f"\nFigures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("00c_*.csv")):
        print(f"  {f.name}")
    print("\nNo rescaling strategy is adopted here -- this is a comparison for "
          "08a's design to be read off, following the project's convention of "
          "measuring before deciding (see e.g. step 06's alpha sweep).")


if __name__ == "__main__":
    main()
