"""
evaluate.py
===========
Evaluation utilities for MLB-quantization experiments.

Functions
---------
evaluate_model        – single-pass loss + accuracy on any DataLoader
inference_time        – per-image latency benchmark
evaluate_and_report   – full evaluation pipeline with formatted output
"""

import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ===========================================================================
# Core evaluation function
# ===========================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate *model* on every batch in *loader*.

    Parameters
    ----------
    model     : the model to evaluate (any nn.Module)
    loader    : DataLoader (val or test)
    criterion : loss function (CrossEntropyLoss recommended)
    device    : torch.device

    Returns
    -------
    avg_loss : float  – mean cross-entropy loss
    accuracy : float  – fraction of correct predictions [0, 1]
    """
    model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ===========================================================================
# Per-class accuracy
# ===========================================================================

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def evaluate_per_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, float]:
    """
    Compute per-class accuracy.

    Returns
    -------
    dict  {class_name: accuracy}
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds  = logits.argmax(dim=1)

            for cls in range(num_classes):
                mask             = labels == cls
                class_correct[cls] += (preds[mask] == labels[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    per_class: Dict[str, float] = {}
    for cls in range(num_classes):
        name = FASHION_MNIST_CLASSES[cls] if cls < len(FASHION_MNIST_CLASSES) else str(cls)
        acc  = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        per_class[name] = acc

    return per_class


# ===========================================================================
# Inference latency benchmark
# ===========================================================================

def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    n_warmup: int = 50,
    n_measure: int = 200,
    batch_size: int = 1,
    input_shape: Tuple[int, ...] = (1, 1, 28, 28),
) -> float:
    """
    Benchmark per-image inference latency.

    Parameters
    ----------
    model        : model to benchmark
    device       : torch.device
    n_warmup     : warm-up forward passes (not timed)
    n_measure    : timed forward passes
    batch_size   : images per forward pass (default 1 for latency)
    input_shape  : shape of a single batch (B, C, H, W)

    Returns
    -------
    ms_per_image : float  – mean milliseconds per image
    """
    model.eval()
    dummy = torch.randn(input_shape, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # CUDA synchronization for accurate timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / n_measure) * 1000.0  # convert to ms
    return ms_per_image


# ===========================================================================
# Full evaluation report
# ===========================================================================

def evaluate_and_report(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger=None,
) -> Dict:
    """
    Run a comprehensive evaluation and return a report dict.

    Computes:
    - Validation loss & accuracy
    - Test loss & accuracy
    - Per-class accuracy (on validation set)
    - Inference latency (per image, ms)

    Parameters
    ----------
    model       : trained (and possibly quantized) model
    val_loader  : validation DataLoader
    test_loader : test DataLoader
    criterion   : loss function
    device      : torch.device
    logger      : optional logger

    Returns
    -------
    report : dict with all evaluation metrics
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    _log("--- Evaluation Report ---")

    # Validation
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
    _log(f"  Val  loss={val_loss:.4f}  acc={val_acc * 100:.2f}%")

    # Test
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    _log(f"  Test loss={test_loss:.4f}  acc={test_acc * 100:.2f}%")

    # Per-class (on val)
    per_class = evaluate_per_class(model, val_loader, device)
    _log("  Per-class accuracy (val):")
    for cls_name, cls_acc in per_class.items():
        _log(f"    {cls_name:<20s}: {cls_acc * 100:.2f}%")

    # Inference latency
    ms_per_image = measure_inference_time(model, device)
    _log(f"  Inference latency: {ms_per_image:.3f} ms/image")

    report = {
        "val_loss":      val_loss,
        "val_accuracy":  val_acc,
        "test_loss":     test_loss,
        "test_accuracy": test_acc,
        "per_class_acc": per_class,
        "ms_per_image":  ms_per_image,
    }
    return report
