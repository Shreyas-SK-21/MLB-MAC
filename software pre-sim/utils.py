"""
utils.py
========
Utility functions shared across the MLB-quantization research framework.

Responsibilities
----------------
- Seed setting for reproducibility
- Dataset loading (Fashion-MNIST, 80/20 split)
- Directory scaffolding
- Logging helpers
- Checkpoint save / load
- Configuration persistence
"""

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure stdout uses UTF-8 on Windows (prevents charmap codec errors when
# printing non-ASCII characters to the terminal or redirected streams).
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from config import Config


# ===========================================================================
# Reproducibility
# ===========================================================================

def set_seed(seed: int):
    """
    Fix all random seeds for full reproducibility.

    Covers: Python built-in, NumPy, PyTorch (CPU + CUDA), and cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ===========================================================================
# Directory scaffolding
# ===========================================================================

def create_output_dirs(cfg: Config):
    """
    Create the results directory tree:
        results/
        ├── checkpoints/
        ├── logs/
        ├── plots/
        └── metrics/
    """
    for d in [
        cfg.results_dir,
        cfg.checkpoints_dir,
        cfg.logs_dir,
        cfg.plots_dir,
        cfg.metrics_dir,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Logging
# ===========================================================================

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a logger that writes to stdout (and optionally to *log_file*).

    Parameters
    ----------
    name     : logger name (typically __name__ or the experiment name)
    log_file : absolute path for the file sink (optional)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (always UTF-8 to handle any Unicode in messages)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ===========================================================================
# Fashion-MNIST dataset
# ===========================================================================

# Fashion-MNIST statistics (channel mean and std computed over training set)
_FM_MEAN = (0.2860,)
_FM_STD  = (0.3530,)


def get_data_loaders(
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return (train_loader, val_loader, test_loader) for Fashion-MNIST.

    Split strategy
    --------------
    The official 60,000-sample training split is divided into:
        train : (1 − val_split) × 60,000  samples  (e.g. 48,000 for 0.2)
        val   :  val_split      × 60,000  samples  (e.g. 12,000 for 0.2)
    The official 10,000-sample test split is used as-is for final evaluation.

    Transforms
    ----------
    Train : RandomHorizontalFlip → ToTensor → Normalize
    Val/Test : ToTensor → Normalize
    """
    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_FM_MEAN, _FM_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_FM_MEAN, _FM_STD),
    ])

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    full_train_dataset = datasets.FashionMNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # We need the val set to use eval_transform.
    # Strategy: create the same dataset twice with different transforms,
    # then split indices deterministically using a seeded generator.
    full_train_eval_dataset = datasets.FashionMNIST(
        root=cfg.data_dir,
        train=True,
        download=False,  # already downloaded above
        transform=eval_transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    # ------------------------------------------------------------------
    # Train / validation split (reproducible)
    # ------------------------------------------------------------------
    n_total = len(full_train_dataset)
    n_val   = int(n_total * cfg.val_split)
    n_train = n_total - n_val

    # Use a seeded generator so split is the same across runs
    split_gen = torch.Generator()
    split_gen.manual_seed(cfg.seed)

    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=split_gen
    )

    # Subset objects – train uses augmented transform, val uses eval transform
    train_subset = Subset(full_train_dataset,      train_indices.indices)
    val_subset   = Subset(full_train_eval_dataset, val_indices.indices)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    # Persistent workers (keep worker processes alive between batches)
    persistent = cfg.num_workers > 0

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
        drop_last=True,          # keeps batch size uniform for BN stability
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(
    state: Dict,
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
):
    """
    Save a training checkpoint to *filepath*.

    If *is_best* is True, also copy to *best_filepath* (or
    `<dir>/best_model.pth` by default).

    The *state* dict should contain at minimum:
        epoch          : int
        model_state    : model.state_dict()
        optimizer_state: optimizer.state_dict()
        scheduler_state: scheduler.state_dict()  (optional)
        val_acc        : float
        val_loss       : float
        config         : cfg.to_dict()
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)

    if is_best:
        import shutil
        if best_filepath is None:
            best_filepath = str(Path(filepath).parent / "best_model.pth")
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath: str, device: str) -> Dict:
    """
    Load a checkpoint from *filepath* onto *device*.

    Returns the raw state dict (caller is responsible for restoring state).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    return torch.load(filepath, map_location=device)


# ===========================================================================
# CSV logging
# ===========================================================================

class CSVLogger:
    """
    Append-mode CSV writer that writes one row per call to `write_row`.

    Creates the file and header on first instantiation.
    """

    def __init__(self, filepath: str, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Write header if file is new / empty
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
        self._file = open(filepath, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: Dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ===========================================================================
# Config persistence
# ===========================================================================

def save_run_config(cfg: Config, fp32_accuracy: Optional[float] = None):
    """
    Save the full run configuration (hyperparameters, quantization config,
    seed) as a JSON file in the logs directory.

    Parameters
    ----------
    cfg           : populated Config object
    fp32_accuracy : optional FP32 baseline accuracy to record
    """
    d = cfg.to_dict()
    if fp32_accuracy is not None:
        d["fp32_baseline_accuracy"] = fp32_accuracy
    d["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_path = os.path.join(cfg.logs_dir, f"{cfg.experiment_name}_config.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    return out_path


# ===========================================================================
# Timing helpers
# ===========================================================================

class Timer:
    """Context-manager wall-clock timer."""

    def __init__(self):
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        assert self._start is not None
        self.elapsed = time.perf_counter() - self._start


# ===========================================================================
# Progress bar helper (no external dependencies)
# ===========================================================================

def progress_bar(current: int, total: int, msg: str = "", width: int = 30) -> str:
    """Return a simple ASCII progress bar string."""
    filled = int(width * current / total)
    bar    = "#" * filled + "-" * (width - filled)
    pct    = 100.0 * current / total
    return f"[{bar}] {pct:5.1f}%  {msg}"
