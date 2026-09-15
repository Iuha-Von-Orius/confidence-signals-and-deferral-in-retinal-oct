"""Fine-tune EfficientNetV2-B0 and select the best validation macro-F1.

Writes best.pt, last.pt and train_log.csv. Resumes from last.pt when present.
"""

from paths import get_device, PROJECT, read_csv

import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Configuration
SPLITS = PROJECT / "splits"
CKPT_DIR = PROJECT / "checkpoints"
RESULTS = PROJECT / "results"

MODEL_NAME = "tf_efficientnetv2_b0"
CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
N_CLASSES = len(CLASSES)

IMG_SIZE = 260

# Throughput was flat across batch 16/32/64 in benchmarking (204/219/217 img/s),
# so the GPU saturates early. 32 gives that throughput at half the memory of 64.
BATCH_SIZE = 32

EPOCHS = 15
PATIENCE = 3
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

DROP_RATE = 0.3
DROP_PATH_RATE = 0.2


DEVICE = get_device()

# ImageNet statistics — the pretrained weights expect inputs normalised this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    """Deterministic preprocessing: resize, to tensor, normalise. No augmentation."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# Evaluation
@torch.no_grad()
def evaluate(model, loader):
    """Run the model over a loader and return metrics."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for x, y in tqdm(loader, desc="  eval", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        all_preds.append(logits.argmax(1).cpu())
        all_labels.append(y.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return {
        "loss": total_loss / len(labels),
        "acc": (preds == labels).mean(),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "per_class_f1": f1_score(labels, preds, average=None),
        "confusion": confusion_matrix(labels, preds),
    }


# Main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Data
    tf = build_transform()
    train_ds = OCTDataset(SPLITS / "train.csv", tf)
    val_ds = OCTDataset(SPLITS / "val.csv", tf)

    # num_workers=4 decodes JPEGs in parallel so the GPU is not left waiting.
    # persistent_workers keeps them alive between epochs, avoiding respawn cost.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, persistent_workers=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"Device: {DEVICE} | Image size: {IMG_SIZE} | Batch: {BATCH_SIZE}\n")

    # Model
    model = timm.create_model(
        MODEL_NAME,
        pretrained=not (CKPT_DIR / "last.pt").exists(),
        num_classes=N_CLASSES,
        drop_rate=DROP_RATE,
        drop_path_rate=DROP_PATH_RATE,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Cosine schedule decays the LR smoothly to ~0 by the final epoch, which
    # stabilises the last few epochs relative to a constant LR.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # Resume if a previous run was interrupted
    start_epoch = 0
    best_f1 = 0.0
    epochs_without_improvement = 0
    log_rows = []

    last_path = CKPT_DIR / "last.pt"
    if last_path.exists():
        # Full training state includes NumPy values as well as tensors.
        ckpt = torch.load(last_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        log_rows = ckpt["log_rows"]
        print(f"Resuming from epoch {start_epoch} (best macro-F1 so far: {best_f1:.4f})\n")

    if start_epoch >= EPOCHS or epochs_without_improvement >= PATIENCE:
        print("The saved training run is complete. Existing checkpoints are unchanged.")
        return

    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(y)
            n_seen += len(y)
            pbar.set_postfix(loss=f"{running_loss/n_seen:.4f}")

        sched.step()
        train_loss = running_loss / n_seen
        val = evaluate(model, val_loader)
        mins = (time.time() - t0) / 60

        print(f"Epoch {epoch+1:2d} | {mins:5.1f} min | "
              f"train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"val acc {val['acc']:.4f} | val macro-F1 {val['macro_f1']:.4f}")
        print("   per-class F1: " + "  ".join(
            f"{c} {f:.3f}" for c, f in zip(CLASSES, val["per_class_f1"])))

        log_rows.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            **{f"f1_{c}": f for c, f in zip(CLASSES, val["per_class_f1"])},
            "lr": sched.get_last_lr()[0],
            "minutes": mins,
        })
        pd.DataFrame(log_rows).to_csv(RESULTS / "train_log.csv", index=False)

        # Checkpointing
        # best.pt holds weights only: everything downstream just needs the model.
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            epochs_without_improvement = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "config": {
                    "model_name": MODEL_NAME, "img_size": IMG_SIZE,
                    "drop_rate": DROP_RATE, "drop_path_rate": DROP_PATH_RATE,
                    "classes": CLASSES,
                },
            }, CKPT_DIR / "best.pt")
            print(f"   -> new best, saved best.pt")
        else:
            epochs_without_improvement += 1
            print(f"   no improvement ({epochs_without_improvement}/{PATIENCE})")

        # last.pt holds full state so an interrupted run can resume exactly.
        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "log_rows": log_rows,
        }, last_path)

        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping: no improvement for {PATIENCE} epochs.")
            break

        print()

    # Summary
    print(f"\nBest validation macro-F1: {best_f1:.4f}")
    print(f"Weights: {CKPT_DIR / 'best.pt'}")
    print(f"Log:     {RESULTS / 'train_log.csv'}")
    print("\nFinal confusion matrix (rows = true, cols = predicted):")
    print("            " + "  ".join(f"{c:>7}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:<9} " + "  ".join(f"{v:>7,}" for v in val["confusion"][i]))


if __name__ == "__main__":
    main()
