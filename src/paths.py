"""Project paths and portable dataset references in CSV files."""

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("OCT_DATA_ROOT", PROJECT.parent / "Dataset")).expanduser().resolve()


def get_device():
    """Select CUDA, MPS or CPU; OCT_DEVICE overrides the automatic choice."""
    import torch

    requested = os.environ.get("OCT_DEVICE")
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronise(device):
    """Wait for GPU work before taking benchmark timings."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_data_path(value):
    """Resolve a dataset-relative path or a recorded path containing /Dataset/."""
    if not isinstance(value, str) or not value:
        return value
    normalised = value.replace("\\", "/")
    if "/Dataset/" in normalised:
        return str(DATASET / normalised.split("/Dataset/", 1)[1])
    if not Path(value).is_absolute():
        return str(DATASET / value)
    return value


def read_csv(path, **kwargs):
    """Read a CSV and resolve its image-path columns without changing row order."""
    import pandas as pd

    frame = pd.read_csv(path, **kwargs)
    for column in ("filepath", "path"):
        if column in frame:
            frame[column] = frame[column].map(resolve_data_path)
    return frame
