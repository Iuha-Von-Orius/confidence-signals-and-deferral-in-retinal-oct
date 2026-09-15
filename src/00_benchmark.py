"""Benchmark EfficientNetV2-B0 training throughput using synthetic images."""

from paths import get_device, synchronise

import time
import torch
import torch.nn.functional as F
import timm

DEVICE = get_device()
MODEL_NAME = "tf_efficientnetv2_b0"
IMG_SIZE = 224
N_TRAIN_IMAGES = 70484
BATCH_SIZES = [16, 32, 64]
N_WARMUP = 5
N_TIMED = 20


def benchmark(batch_size):
    """Run a few training steps at this batch size, return images/sec."""
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=4).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    y = torch.randint(0, 4, (batch_size,), device=DEVICE)

    # Warm-up: first few steps include one-off setup cost
    for _ in range(N_WARMUP):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    synchronise(DEVICE)

    # Timed run
    t0 = time.time()
    for _ in range(N_TIMED):
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()
    synchronise(DEVICE)
    elapsed = time.time() - t0

    return N_TIMED * batch_size / elapsed


def main():
    print(f"Device: {DEVICE}")
    print(f"Model:  {MODEL_NAME}  @ {IMG_SIZE}x{IMG_SIZE}")
    print(f"Training split: {N_TRAIN_IMAGES:,} images\n")

    print(f"{'batch':>6}  {'img/sec':>9}  {'min/epoch':>10}  {'15 epochs':>12}")
    print("-" * 44)

    for bs in BATCH_SIZES:
        try:
            ips = benchmark(bs)
            mins = N_TRAIN_IMAGES / ips / 60
            print(f"{bs:>6}  {ips:>9.0f}  {mins:>10.1f}  {mins*15/60:>10.1f} h")
        except Exception as e:
            print(f"{bs:>6}  failed: {type(e).__name__}: {e}")

    print("\nNote: real training will be somewhat slower — this excludes")
    print("data loading and JPEG decoding.")


if __name__ == "__main__":
    main()
