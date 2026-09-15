"""Extract frozen-model features for RETOUCH and RASTI.

Use nine central B-scans per volume. Convert uint16 intensities with
round(v / 65535 * 255); uint8 images retain their scale. RASTI accepts TIFF
slices, with PNG fallback only when a volume has no matching TIFFs.
Internal oct5k names refer to RASTI. Writes indices, arrays and diagnostics.
"""

from paths import get_device, PROJECT, DATASET, read_csv

import random
import re

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

SPLITS = PROJECT / "splits"
CKPT_DIR = PROJECT / "checkpoints"
CACHE = PROJECT / "cache"
RESULTS = PROJECT / "results"
FIGDIR = RESULTS / "figures" / "08a_verify"
TABDIR = RESULTS / "tables"

RETOUCH_ROOT = (DATASET / "RETOUCH/RETOUCH")
OCT5K_ROOT = (DATASET / "Macular-Dataset-R.Rasti_old")

SEED = 42
N_CENTRAL_SLICES = 9
N_KERMANY_SAMPLE = 300
DECODE_SELF_CHECK_TOLERANCE = 0.5

VENDORS = ["Cirrus", "Spectralis", "Topcon"]
BYTES_PER = {"MET_UCHAR": 1, "MET_CHAR": 1, "MET_USHORT": 2, "MET_SHORT": 2,
             "MET_UINT": 4, "MET_INT": 4, "MET_FLOAT": 4, "MET_DOUBLE": 8}

IMAGE_SLICE_RE = re.compile(r"^Image\s*(\d+)\.TIFF$", re.IGNORECASE)
IMAGE_SLICE_PNG_FALLBACK_RE = re.compile(r"^Image\s*(\d+)\.PNG$", re.IGNORECASE)

RESCALE_METHOD = "linear_fixed"

MODEL_NAME = "tf_efficientnetv2_b0"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)
IMG_SIZE = 260
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2
BATCH_SIZE = 64
MC_PASSES = 50
DEVICE = get_device()
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OCT5K_LABEL_MAP = {"AMD": CLASSES.index("DRUSEN"), "DME": CLASSES.index("DME"),
                   "Normal": CLASSES.index("NORMAL")}


def cache_filename(dataset_tag, stage):
    """Include the rescaling method in RETOUCH cache filenames."""
    return f"{dataset_tag}_{RESCALE_METHOD}_{stage}.npy"


plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
})
CURVE_COLOURS = {
    "Kermany": "#2c3e50",
    "OCT5k-Spectralis": "#9467bd",
    "RETOUCH-Spectralis (decoded)": "#d62728",
    "RETOUCH-Topcon": "#1f77b4",
    "RETOUCH-Cirrus": "#2ca02c",
}


# Decode
def decode_linear_fixed(arr_uint16):
    """Map uint16 to uint8 with round(v / 65535 * 255), using a fixed denominator."""
    CONTAINER_MAX = 65535.0
    out = np.round(arr_uint16.astype(np.float32) / CONTAINER_MAX * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


RESCALERS = {"linear_fixed": decode_linear_fixed}


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
        dims = [int(x) for x in fields["DimSize"].split()]
        rows.append({
            "path": mhd, "vendor": guess_vendor(mhd),
            "set": "train" if "TrainingSet" in str(mhd) else "test",
            "element_type": fields.get("ElementType", "?"),
            "dims": dims,
        })
    return pd.DataFrame(rows)


def verify_header_consistency(df):
    """Check that volume dimensions and declared byte counts agree."""
    rows = []
    for _, row in df.iterrows():
        raw = row["path"].parent / "oct.raw"
        expected = 1
        for d in row["dims"]:
            expected *= d
        expected *= BYTES_PER[row["element_type"]]
        actual = raw.stat().st_size
        rows.append({
            "path": str(row["path"]), "vendor": row["vendor"],
            "element_type": row["element_type"], "dims": str(row["dims"]),
            "expected_bytes": expected, "actual_bytes": actual,
            "consistent": expected == actual,
        })
    return pd.DataFrame(rows)


def central_slice_indices(n_slices, k=N_CENTRAL_SLICES):
    lo = n_slices // 2 - k // 2
    hi = n_slices // 2 + k // 2
    lo, hi = max(0, lo), min(n_slices - 1, hi)
    return list(range(lo, hi + 1))


def load_retouch_central_slices(mhd_path, decode=False):
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)
    idx = central_slice_indices(arr.shape[0])
    central = arr[idx]
    if decode:
        central = RESCALERS[RESCALE_METHOD](central)
    return central


def find_oct5k_volumes():
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


def load_oct5k_central_slices(volume_row):
    slices = volume_row["slices"]
    idx = central_slice_indices(len(slices))
    out = []
    for i in idx:
        _, p = slices[i]
        with Image.open(p) as im:
            out.append(np.asarray(im.convert("L")))
    return np.stack(out)


def load_kermany_sample(n, seed=SEED):
    df = read_csv(SPLITS / "train.csv")
    sample = df.sample(n=n, random_state=seed)
    out = []
    for fp in sample["filepath"]:
        with Image.open(fp) as im:
            out.append(np.asarray(im.convert("L")))
    return out


# Stats / classifier
def image_stats(arr):
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return [float(arr.mean()), float(arr.std()), p5, p25, p50, p75, p95]


def confusion_accuracy(feats_a, feats_b, seed=SEED):
    """Measure five-fold cross-validated group classification from seven intensity statistics."""
    X = np.array(feats_a + feats_b)
    y = np.array([0] * len(feats_a) + [1] * len(feats_b))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


# Figures
def fig_five_way_histogram(samples_255):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for name, px in samples_255.items():
        ax.hist(px, bins=80, range=(0, 255), density=True, alpha=0.4,
                color=CURVE_COLOURS[name], label=name)
    ax.set_xlabel("Pixel value (0-255)")
    ax.set_ylabel("Density")
    ax.set_title("Five-way intensity comparison, 9-central-slice samples,\n"
                 "RETOUCH-Spectralis after fixed linear decode")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "08a_five_way_histogram.png")
    plt.close(fig)


def fig_raw16bit_histogram(raw_px):
    frac_sat = float((raw_px == 65535).mean())
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(raw_px, bins=100, color="#8b1a1a", alpha=0.85)
    ax.axvline(65535, color="black", ls="--", lw=1.5,
               label=f"65535 (container max) -- {frac_sat*100:.3f}% of pixels")
    ax.set_xlabel("Raw pixel value (16-bit)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title("RETOUCH-Spectralis, pre-decode, 9-central-slice sample\n"
                 "Saturation check at the container ceiling")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "08a_raw16bit_histogram.png")
    plt.close(fig)


def fig_decode_mean_per_volume(decode_df):
    """Plot decoded volume means and their differences from raw means divided by 257."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax.bar(range(len(decode_df)), decode_df["post_decode_mean"].sort_values(),
           color="#2c3e50", alpha=0.85)
    ax.set_xlabel("Spectralis volumes (sorted by post-decode mean)")
    ax.set_ylabel("Post-decode mean (0-255)")
    ax.set_title("Post-decode mean per volume (descriptive)")

    disc = decode_df["self_check_diff"].sort_values()
    ax2.bar(range(len(disc)), disc, color="#8b1a1a", alpha=0.85)
    ax2.axhspan(-DECODE_SELF_CHECK_TOLERANCE, DECODE_SELF_CHECK_TOLERANCE,
               color="#2ca02c", alpha=0.15,
               label=f"tolerance +/-{DECODE_SELF_CHECK_TOLERANCE}")
    ax2.set_xlabel("Spectralis volumes (sorted)")
    ax2.set_ylabel("decoded.mean() - raw.mean()/257")
    ax2.set_title("Check (a): self-consistency (pass/fail signal)")
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "08a_decode_mean_per_volume.png")
    plt.close(fig)


def load_backbone():
    """Load the frozen classifier for deterministic and stochastic inference."""
    model = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=N_CLASSES,
        drop_rate=DROP_RATE, drop_path_rate=DROP_PATH_RATE,
    )
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    print(f"  Loaded best.pt (epoch {ckpt['epoch'] + 1}, "
          f"val macro-F1 {ckpt['val_macro_f1']:.4f})")
    return model


def build_feature_extractor():
    """Load the intermediate feature model; classifier and head keys are unused."""
    model = timm.create_model(
        MODEL_NAME, pretrained=False, features_only=True,
        drop_path_rate=DROP_PATH_RATE,
    )
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"  features_only graph: {len(unexpected)} unexpected keys "
          f"(expected: conv_head, bn2, classifier), {len(missing)} missing")
    if missing:
        print(f"  [ERROR] missing weights: {missing[:5]}")
    model.to(DEVICE).eval()
    return model


def build_backbone_transform():
    """Resize RGB inputs and apply ImageNet normalisation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def enable_mc_dropout(model):
    """Enable stochastic modules while keeping BatchNorm in evaluation mode.

    In this backbone, MC randomness comes from DropPath; functional classifier
    dropout remains inactive because the model-level training flag is False.
    """
    n = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            module.train()
            n += 1
        elif module.__class__.__name__ == "DropPath":
            module.train()
            n += 1
    return n


@torch.no_grad()
def extract_mc_dropout(model, loader, n_passes):
    """Return probabilities from repeated stochastic forward passes."""
    enabled = enable_mc_dropout(model)
    print(f"    {enabled} stochastic modules active")
    per_pass = []
    for t in tqdm(range(n_passes), desc="    MC passes", leave=False):
        torch.manual_seed(SEED + t)
        batch_probs = []
        for x, _ in loader:
            probs = torch.softmax(model(x.to(DEVICE)), dim=1)
            batch_probs.append(probs.cpu().numpy())
        per_pass.append(np.concatenate(batch_probs))
    model.eval()
    return np.stack(per_pass).transpose(1, 0, 2).astype(np.float32)


@torch.no_grad()
def extract_stages(feat_model, loader):
    """Return pooled features from the five intermediate resolution levels."""
    stages = {i: [] for i in range(5)}
    for x, _ in tqdm(loader, desc="    stages 1-5", leave=False):
        maps = feat_model(x.to(DEVICE))
        for i, fmap in enumerate(maps):
            stages[i].append(fmap.mean(dim=(2, 3)).cpu().numpy())
    return [np.concatenate(stages[i]).astype(np.float32) for i in range(5)]


@torch.no_grad()
def extract_penultimate_and_logits(model, loader, want_logits):
    """Identical to 04_extract_features.py:extract_penultimate_and_logits -- stage 6 is forward_features + forward_head(pre_logits=True) on the full (non-features_only) backbone, not the 192-d stage-5 output."""
    pen_all, logits_all = [], []
    for x, _ in tqdm(loader, desc="    stage 6 + logits", leave=False):
        x = x.to(DEVICE)
        feats = model.forward_features(x)
        pen = model.forward_head(feats, pre_logits=True)
        pen_all.append(pen.cpu().numpy())
        if want_logits:
            logits_all.append(model.get_classifier()(pen).cpu().numpy())
    out = {"penultimate": np.concatenate(pen_all).astype(np.float32)}
    if want_logits:
        out["logits"] = np.concatenate(logits_all).astype(np.float32)
    return out


class ArrayDataset(Dataset):
    """Load greyscale arrays with their index records and deterministic transform."""
    def __init__(self, arrays, transform):
        self.arrays = arrays
        self.transform = transform

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        img = Image.fromarray(self.arrays[idx], mode="L").convert("RGB")
        return self.transform(img), 0


# Phase 2: assembly
def assemble_retouch_slices(df):
    """Collect nine central B-scans per volume; decode uint16 using fixed scaling."""
    arrays, index_rows = [], []
    for _, row in df.iterrows():
        img = sitk.ReadImage(str(row["path"]))
        arr = sitk.GetArrayFromImage(img)
        idx = central_slice_indices(arr.shape[0])
        central = arr[idx]
        if row["element_type"] == "MET_USHORT":
            central = RESCALERS[RESCALE_METHOD](central)
        else:
            central = central.astype(np.uint8)
        for pos, slice_idx in enumerate(idx):
            arrays.append(central[pos])
            index_rows.append({
                "dataset": "RETOUCH", "vendor": row["vendor"], "set": row["set"],
                "volume_path": str(row["path"]),
                "volume_id": row["path"].parent.name,
                "slice_index": slice_idx, "label": -1,
                "known_diseased": True,
            })
    return arrays, index_rows


def assemble_oct5k_slices(volumes):
    """Collect nine central RASTI B-scans with TIFF-first, PNG-fallback loading."""
    arrays, index_rows = [], []
    for v in volumes:
        slices = v["slices"]
        idx = central_slice_indices(len(slices))
        for pos in idx:
            slice_idx, p = slices[pos]
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L"))
            arrays.append(arr)
            index_rows.append({
                "dataset": "OCT5k", "vendor": "Spectralis", "set": "n/a",
                "volume_path": str(v["volume_dir"]),
                "volume_id": v["volume_dir"].name,
                "slice_index": slice_idx, "label": OCT5K_LABEL_MAP[v["class"]],
                "known_diseased": v["class"] != "Normal",
            })
    return arrays, index_rows


def run_full_extraction(dataset_tag, arrays, index_rows, model, feat_model,
                        transform, tag_with_method):
    """Cache six feature vectors, logits and deterministic and stochastic probabilities."""
    prefix = f"{dataset_tag}_{RESCALE_METHOD}" if tag_with_method else dataset_tag
    print(f"\n{dataset_tag}: {len(arrays)} slices")

    ds = ArrayDataset(arrays, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    stages = extract_stages(feat_model, loader)
    pen = extract_penultimate_and_logits(model, loader, want_logits=True)
    all_feats = stages + [pen["penultimate"]]
    for i, f in enumerate(all_feats, start=1):
        np.save(CACHE / f"{prefix}_feat_stage{i}.npy", f)

    logits = pen["logits"]
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy().astype(np.float32)
    np.save(CACHE / f"{prefix}_logits.npy", logits)
    np.save(CACHE / f"{prefix}_probs.npy", probs)

    mc_probs = extract_mc_dropout(model, loader, MC_PASSES)
    np.save(CACHE / f"{prefix}_mc_probs.npy", mc_probs)

    spread = float(mc_probs.std(axis=1).mean())
    print(f"  mean per-class std across MC passes: {spread:.5f}"
          + ("  [WARNING] near-zero spread" if spread < 1e-4 else ""))

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(CACHE / f"{prefix}_index.csv", index=False)
    print(f"  saved: {prefix}_feat_stage1..6.npy, {prefix}_probs.npy, "
          f"{prefix}_mc_probs.npy, {prefix}_index.csv")
    return prefix


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    print("Discovering RETOUCH oct.mhd volumes...")
    df = find_oct_volumes()
    print(df.groupby(["vendor", "element_type"]).size().to_string())

    print("\nHeader-consistency cross-check, all 112 oct.mhd "
          "(width x height x slices x bytes_per_element == actual .raw size)...")
    consistency = verify_header_consistency(df)
    consistency.to_csv(TABDIR / "08a_header_consistency.csv", index=False)
    n_bad = (~consistency["consistent"]).sum()
    if n_bad:
        print(f"  [ERROR] {n_bad} volumes inconsistent -- stopping.")
        for _, r in consistency[~consistency.consistent].iterrows():
            print(f"    {r['path']}  expected {r['expected_bytes']}, "
                  f"actual {r['actual_bytes']}")
        return
    print(f"  All {len(consistency)} consistent -- declared ElementType "
          "matches the file size for every volume.")

    spectralis = df[df.vendor == "Spectralis"].reset_index(drop=True)
    topcon = df[df.vendor == "Topcon"].reset_index(drop=True)
    cirrus = df[df.vendor == "Cirrus"].reset_index(drop=True)
    print(f"\n{len(spectralis)} Spectralis, {len(topcon)} Topcon, "
          f"{len(cirrus)} Cirrus volumes.")

    print(f"\nCheck (a): decoding {N_CENTRAL_SLICES} central slices per "
          f"Spectralis volume ({RESCALE_METHOD})...")
    decode_rows = []
    decoded_slices_by_volume = {}
    raw_slices_by_volume = {}
    for _, row in spectralis.iterrows():
        img = sitk.ReadImage(str(row["path"]))
        arr = sitk.GetArrayFromImage(img)
        idx = central_slice_indices(arr.shape[0])
        raw = arr[idx]
        decoded = RESCALERS[RESCALE_METHOD](raw)
        assert decoded.dtype == np.uint8
        assert decoded.min() >= 0 and decoded.max() <= 255
        raw_slices_by_volume[row["path"]] = raw
        decoded_slices_by_volume[row["path"]] = decoded
        raw_mean = float(raw.mean())
        decoded_mean = float(decoded.mean())
        decode_rows.append({
            "path": str(row["path"]), "raw_central9_mean": raw_mean,
            "post_decode_mean": decoded_mean,
            "self_check_diff": decoded_mean - raw_mean / 257.0,
        })
    decode_df = pd.DataFrame(decode_rows)

    whole_vol_stats = read_csv(TABDIR / "00c_volume_stats.csv")[["path", "mean"]]
    whole_vol_stats = whole_vol_stats.rename(columns={"mean": "raw_whole_volume_mean"})
    decode_df = decode_df.merge(whole_vol_stats, on="path", how="left")
    decode_df["central9_vs_whole_pct"] = (
        (decode_df["raw_central9_mean"] - decode_df["raw_whole_volume_mean"])
        / decode_df["raw_whole_volume_mean"] * 100.0
    )
    decode_df.to_csv(TABDIR / "08a_decode_check.csv", index=False)

    max_diff = decode_df["self_check_diff"].abs().max()
    print(f"  Post-decode mean: min {decode_df.post_decode_mean.min():.1f}, "
          f"max {decode_df.post_decode_mean.max():.1f}, "
          f"mean {decode_df.post_decode_mean.mean():.1f}  (descriptive only)")
    print(f"  Self-consistency: max |decoded.mean() - raw.mean()/257| = "
          f"{max_diff:.4f}  (tolerance {DECODE_SELF_CHECK_TOLERANCE})")
    fig_decode_mean_per_volume(decode_df)

    failing = decode_df[decode_df["self_check_diff"].abs() > DECODE_SELF_CHECK_TOLERANCE]
    if len(failing):
        print(f"\n  [STOP] {len(failing)} volumes fail self-consistency -- "
              "not proceeding to (b)/(c). This means the decoder does not "
              "correctly implement its own declared formula.")
        for _, r in failing.iterrows():
            print(f"    {r['path']}  diff={r['self_check_diff']:.4f}")
        return
    print(f"  All {len(decode_df)} volumes self-consistent -- decoder "
          "verified correct. Proceeding to (b).")

    # Check (b): five-way histogram + raw saturation check
    print("\nCheck (b): building the five-way histogram sample...")
    kermany_imgs = load_kermany_sample(N_KERMANY_SAMPLE)
    kermany_px = np.concatenate([im.ravel() for im in kermany_imgs])

    oct5k_volumes = find_oct5k_volumes()
    print(f"  OCT5k: {len(oct5k_volumes)} volumes, sampling "
          f"{N_CENTRAL_SLICES} central slices each...")
    oct5k_slices = [load_oct5k_central_slices(v) for v in oct5k_volumes]
    oct5k_px = np.concatenate([s.ravel() for s in oct5k_slices])

    retouch_spectralis_px = np.concatenate(
        [s.ravel() for s in decoded_slices_by_volume.values()])
    retouch_spectralis_raw_px = np.concatenate(
        [s.ravel() for s in raw_slices_by_volume.values()])

    print(f"  RETOUCH-Topcon: {len(topcon)} volumes...")
    topcon_px = np.concatenate([
        load_retouch_central_slices(p).ravel() for p in topcon["path"]
    ])
    print(f"  RETOUCH-Cirrus: {len(cirrus)} volumes...")
    cirrus_px = np.concatenate([
        load_retouch_central_slices(p).ravel() for p in cirrus["path"]
    ])

    fig_five_way_histogram({
        "Kermany": kermany_px,
        "OCT5k-Spectralis": oct5k_px,
        "RETOUCH-Spectralis (decoded)": retouch_spectralis_px,
        "RETOUCH-Topcon": topcon_px,
        "RETOUCH-Cirrus": cirrus_px,
    })
    fig_raw16bit_histogram(retouch_spectralis_raw_px)

    frac_sat = float((retouch_spectralis_raw_px == 65535).mean())
    print(f"  Saturation at 65535: {frac_sat*100:.4f}% of pixels in the "
          "9-central-slice sample.")

    summary_rows = []
    for name, px in [("Kermany", kermany_px), ("OCT5k-Spectralis", oct5k_px),
                     ("RETOUCH-Spectralis (decoded)", retouch_spectralis_px),
                     ("RETOUCH-Topcon", topcon_px),
                     ("RETOUCH-Cirrus", cirrus_px)]:
        p5, p25, p50, p75, p95, p99 = np.percentile(px, [5, 25, 50, 75, 95, 99])
        summary_rows.append({"dataset": name, "mean": px.mean(), "std": px.std(),
                             "p5": p5, "p25": p25, "p50": p50, "p75": p75,
                             "p95": p95, "p99": p99})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABDIR / "08a_five_way_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # Check (c): confusion classifier
    print("\nCheck (c): confusion classifier, Kermany vs RETOUCH-Spectralis...")
    kermany_feats = [image_stats(im) for im in kermany_imgs]
    spec_raw_feats = [image_stats(s) for arr in raw_slices_by_volume.values()
                      for s in arr]
    spec_decoded_feats = [image_stats(s) for arr in decoded_slices_by_volume.values()
                          for s in arr]

    acc_before, std_before = confusion_accuracy(kermany_feats, spec_raw_feats)
    acc_after, std_after = confusion_accuracy(kermany_feats, spec_decoded_feats)
    print(f"  Before decode (raw 16-bit vs Kermany 8-bit): "
          f"{acc_before:.3f} +/- {std_before:.3f}  "
          "(sanity check only -- trivially separable by scale alone)")
    print(f"  After decode  (decoded 8-bit vs Kermany 8-bit): "
          f"{acc_after:.3f} +/- {std_after:.3f}  "
          "(the real test; 0.5 = indistinguishable on these statistics)")

    oct5k_feats = [image_stats(s) for arr in oct5k_slices for s in arr]
    acc_oct5k, std_oct5k = confusion_accuracy(kermany_feats, oct5k_feats)
    print(f"  Reference (Kermany vs OCT5k-Spectralis, never decoded): "
          f"{acc_oct5k:.3f} +/- {std_oct5k:.3f}  "
          "(same device as RETOUCH-Spectralis, native 8-bit throughout)")

    pd.DataFrame([
        {"condition": "before_decode", "comparison": "Kermany vs RETOUCH-Spectralis (raw 16-bit)",
         "accuracy": acc_before, "std": std_before,
         "n_kermany": len(kermany_feats), "n_other": len(spec_raw_feats)},
        {"condition": "after_decode", "comparison": "Kermany vs RETOUCH-Spectralis (decoded)",
         "accuracy": acc_after, "std": std_after,
         "n_kermany": len(kermany_feats), "n_other": len(spec_decoded_feats)},
        {"condition": "reference_never_decoded", "comparison": "Kermany vs OCT5k-Spectralis (native 8-bit)",
         "accuracy": acc_oct5k, "std": std_oct5k,
         "n_kermany": len(kermany_feats), "n_other": len(oct5k_feats)},
    ]).to_csv(TABDIR / "08a_confusion_classifier.csv", index=False)

    print("\n" + "=" * 90)
    print(f"Figures: {FIGDIR.relative_to(PROJECT)}/")
    for f in sorted(FIGDIR.glob("*.png")):
        print(f"  {f.name}")
    print(f"Tables:  {TABDIR.relative_to(PROJECT)}/")
    for f in sorted(TABDIR.glob("08a_*.csv")):
        print(f"  {f.name}")
    print("\nDecision rule (Phase 1, reference only): three post-decode "
          "curves broadly overlapping / no saturation spike / Kermany and "
          "OCT5k themselves agreeing were the three checks. Confirmed by the "
          "user on 2026-08-26 after review: proceeding to Phase 2. The one "
          "remaining stop condition (Spectralis scoring more OOD than "
          "Topcon/Cirrus) is evaluated in 08c, once real OOD scores exist --")
    print("it cannot be checked from pixel statistics alone.")

    # Phase 2
    print("\n" + "=" * 90)
    print("PHASE 2: full cross-device feature extraction "
          "(backbone + MC Dropout, all 112 RETOUCH + 148 OCT5k volumes)")
    print("=" * 90)

    transform = build_backbone_transform()
    print("\nLoading backbone...")
    model = load_backbone()
    feat_model = build_feature_extractor()

    print("\nAssembling RETOUCH slices (decoding Spectralis where needed)...")
    retouch_arrays, retouch_index = assemble_retouch_slices(df)
    print(f"  {len(retouch_arrays)} slices from {len(df)} volumes "
          f"({len(spectralis)} Spectralis decoded, "
          f"{len(topcon) + len(cirrus)} Topcon/Cirrus passed through)")
    run_full_extraction("retouch", retouch_arrays, retouch_index, model,
                        feat_model, transform, tag_with_method=True)

    print("\nAssembling OCT5k slices...")
    oct5k_arrays, oct5k_index = assemble_oct5k_slices(oct5k_volumes)
    print(f"  {len(oct5k_arrays)} slices from {len(oct5k_volumes)} volumes")
    run_full_extraction("oct5k", oct5k_arrays, oct5k_index, model,
                        feat_model, transform, tag_with_method=False)

    print("\n" + "=" * 90)
    print("PHASE 2 COMPLETE. Cached to cache/:")
    print("  retouch_linear_fixed_feat_stage1..6.npy, _probs.npy, _mc_probs.npy, _index.csv")
    print("  oct5k_feat_stage1..6.npy, _probs.npy, _mc_probs.npy, _index.csv")
    print("\nNext: 08b_score_ood.py applies the frozen OOD detectors, "
          "conformal thresholds and policy to these cached features.")


if __name__ == "__main__":
    main()
