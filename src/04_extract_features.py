"""Cache six feature vectors per image from the frozen classifier.

Validation, calibration and test also receive logits, probabilities and 50
stochastic passes. MC inference enables DropPath while BatchNorm and the
classifier dropout remain in evaluation mode. Cache rows follow the split CSVs.
"""

from paths import get_device, PROJECT, read_csv

import json
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Configuration
SPLITS = PROJECT / "splits"
CKPT_DIR = PROJECT / "checkpoints"
CACHE = PROJECT / "cache"

MODEL_NAME = "tf_efficientnetv2_b0"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)

# Must match training exactly. A mismatch produces features that do not
# correspond to what the model learned, without raising an error.
IMG_SIZE = 260
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2

# Larger than training: inference has no backward pass, so memory is free.
BATCH_SIZE = 64

MC_PASSES = 50

DECISION_SPLITS = ["val", "calibration", "test"]
FEATURE_ONLY_SPLITS = ["train"]

DEVICE = get_device()
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_STAGES = 6


# Dataset
class OCTDataset(Dataset):
    """Load images and labels in split-CSV order, converting images to RGB."""

    def __init__(self, csv_path, transform):
        self.df = read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), int(row["label"])


def build_transform():
    """Identical to training. No augmentation, no randomness."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Model loading
def load_backbone():
    """Rebuild the architecture and load the trained weights."""
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=N_CLASSES,
        drop_rate=DROP_RATE,
        drop_path_rate=DROP_PATH_RATE,
    )
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    print(f"Loaded best.pt  (epoch {ckpt['epoch'] + 1}, "
          f"val macro-F1 {ckpt['val_macro_f1']:.4f})")
    return model


def build_feature_extractor():
    """Load the intermediate feature model; classifier and head keys are unused."""
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        features_only=True,
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


# Extraction
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
    """Stage 6: the 1280-dim vector immediately before the classifier, plus logits."""
    pen_all, logits_all, labels_all = [], [], []
    for x, y in tqdm(loader, desc="    stage 6 + logits", leave=False):
        x = x.to(DEVICE)
        feats = model.forward_features(x)
        pen = model.forward_head(feats, pre_logits=True)
        pen_all.append(pen.cpu().numpy())
        if want_logits:
            # Reuse the same forward_features output rather than a second full
            # pass: the classifier is one linear layer applied to `pen`.
            logits_all.append(model.get_classifier()(pen).cpu().numpy())
        labels_all.append(y.numpy())

    out = {
        "penultimate": np.concatenate(pen_all).astype(np.float32),
        "labels": np.concatenate(labels_all).astype(np.int64),
    }
    if want_logits:
        out["logits"] = np.concatenate(logits_all).astype(np.float32)
    return out


@torch.no_grad()
def extract_mc_dropout(model, loader, n_passes):
    """Return probabilities from repeated stochastic forward passes."""
    enabled = enable_mc_dropout(model)
    print(f"    {enabled} stochastic modules active")

    per_pass = []
    for t in tqdm(range(n_passes), desc="    MC passes", leave=False):
        # Per-pass seeding keeps the whole extraction reproducible while still
        # giving a different dropout mask on each pass.
        torch.manual_seed(SEED + t)
        batch_probs = []
        for x, _ in loader:
            probs = torch.softmax(model(x.to(DEVICE)), dim=1)
            batch_probs.append(probs.cpu().numpy())
        per_pass.append(np.concatenate(batch_probs))

    model.eval()
    # stack gives (T, N, 4); transpose to (N, T, 4) so indexing by image is natural
    return np.stack(per_pass).transpose(1, 0, 2).astype(np.float32)


def process_split(split, model, feat_model, transform, with_mc):
    """Run one split end to end and write its arrays to cache/."""
    ds = OCTDataset(SPLITS / f"{split}.csv", transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[{split}]  {len(ds):,} images"
          f"{'  (features + softmax + MC)' if with_mc else '  (features only)'}")
    t0 = time.time()

    stages = extract_stages(feat_model, loader)
    pen = extract_penultimate_and_logits(model, loader, want_logits=with_mc)

    all_feats = stages + [pen["penultimate"]]
    for i, f in enumerate(all_feats, start=1):
        np.save(CACHE / f"{split}_feat_stage{i}.npy", f)
    np.save(CACHE / f"{split}_labels.npy", pen["labels"])

    meta = {
        "split": split,
        "n_images": len(ds),
        "feature_dims": [f.shape[1] for f in all_feats],
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "model": MODEL_NAME,
        "drop_rate": DROP_RATE,
        "drop_path_rate": DROP_PATH_RATE,
        "seed": SEED,
    }

    if with_mc:
        logits = pen["logits"]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        np.save(CACHE / f"{split}_logits.npy", logits)
        np.save(CACHE / f"{split}_probs.npy", probs.astype(np.float32))

        mc = extract_mc_dropout(model, loader, MC_PASSES)
        np.save(CACHE / f"{split}_mc_probs.npy", mc)
        meta["mc_passes"] = MC_PASSES

        spread = float(mc.std(axis=1).mean())
        meta["mc_mean_std"] = round(spread, 6)
        print(f"    mean per-class std across passes: {spread:.5f}")
        if spread < 1e-4:
            print("    [WARNING] near-zero spread — MC Dropout gives no signal.")

    meta["seconds"] = round(time.time() - t0, 1)
    with open(CACHE / f"{split}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"    done in {meta['seconds'] / 60:.1f} min")
    return meta


# Main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE} | image {IMG_SIZE} | batch {BATCH_SIZE} | "
          f"MC passes {MC_PASSES}\n")

    model = load_backbone()
    feat_model = build_feature_extractor()
    transform = build_transform()

    metas = []
    for split in FEATURE_ONLY_SPLITS:
        metas.append(process_split(split, model, feat_model, transform, with_mc=False))
    for split in DECISION_SPLITS:
        metas.append(process_split(split, model, feat_model, transform, with_mc=True))

    # Summary
    print("\n" + "=" * 62)
    dims = metas[0]["feature_dims"]
    print(f"Feature dimensions, stages 1-6: {dims}")
    if dims[-1] != 1280:
        print(f"  [ERROR] stage 6 should be 1280-dim, got {dims[-1]}")
    else:
        print("  stage 6 = 1280 as expected (penultimate, post conv_head)")

    total_mb = sum(f.stat().st_size for f in CACHE.glob("*.npy")) / 1024**2
    print(f"Cache: {total_mb:.0f} MB across {len(list(CACHE.glob('*.npy')))} arrays")
    print(f"Total time: {sum(m['seconds'] for m in metas) / 60:.1f} min")
    print("\nSteps 05-08 read these arrays and require no GPU.")


if __name__ == "__main__":
    main()
