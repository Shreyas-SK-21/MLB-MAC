"""
metrics.py
==========
Experiment metrics computation, CSV reporting, and plot generation.

Metrics computed (all 13 required + layer report)
--------------------------------------------------
 1.  Final validation accuracy
 2.  Final validation loss
 3.  Training time (seconds)
 4.  Inference time per image (ms)
 5.  Total parameter count
 6.  Model size in MB  (FP32)
 7.  Effective average MLB level  (mean M over all MLB layers)
 8.  Compression ratio relative to FP32
     = M_avg / 32  (bits per weight: binary vs float32)
 9.  Estimated binary operation count
10.  Estimated MAC reduction
11.  Total binary bases used  (Σ M_i)
12.  Quantization error  ‖W − W_MLB‖_F  (summed over all MLB layers)
13.  Accuracy drop relative to FP32 baseline
     (read from config if available, else recorded as N/A)

Layer-level CSV (layer_quantization_report.csv)
------------------------------------------------
    layer_name | M | num_parameters | quantization_error | compression_ratio
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, ExperimentMode
from evaluate import measure_inference_time
from quantization import (
    MLBConv2d,
    MLBLinear,
    HierarchicalMLBConv2d,
    HierarchicalMLBLinear,
    mlb_decompose,
    mlb_reconstruct,
    quantization_error,
    _ALL_MLB_TYPES,
)

# Convenience alias for layer iteration
_HIER_TYPES = (HierarchicalMLBConv2d, HierarchicalMLBLinear)


# ===========================================================================
# Helper: iterate MLB layers
# ===========================================================================

def _get_mlb_layers(model: nn.Module):
    """Yield (name, module) for all quantized layers (standard + hierarchical)."""
    for name, module in model.named_modules():
        if isinstance(module, _ALL_MLB_TYPES):
            yield name, module


# ===========================================================================
# Core metric computations
# ===========================================================================

def compute_effective_avg_M(model: nn.Module) -> float:
    """
    Compute the weighted-average M value across all MLB layers,
    weighted by the number of parameters in each layer.

    For FP32 models (no MLB layers) returns 0.
    """
    total_params  = 0
    weighted_M    = 0.0
    for _, module in _get_mlb_layers(model):
        n = module.weight.numel()
        weighted_M   += module.M * n
        total_params += n
    if total_params == 0:
        return 0.0
    return weighted_M / total_params


def compute_total_binary_bases(model: nn.Module) -> int:
    """
    Sum of M_i over all MLB layers.

    For uniform M=4 with K layers: returns 4K.
    """
    return sum(module.M for _, module in _get_mlb_layers(model))


def compute_compression_ratio(model: nn.Module) -> float:
    """
    Compression ratio of binary weights vs FP32.

    CR = 32 / M_avg     (bits per weight saved by using ±1 binary vs float32)

    Returns 1.0 for FP32 models (no compression).
    """
    m_avg = compute_effective_avg_M(model)
    if m_avg == 0.0:
        return 1.0
    return 32.0 / m_avg


def compute_per_layer_quantization_errors(
    model: nn.Module,
    layer_ptq_info: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """
    Return a dict mapping layer name -> quantization error (Frobenius norm)
    for every quantized layer.
    """
    errors: Dict[str, float] = {}
    for name, module in _get_mlb_layers(model):
        if layer_ptq_info and name in layer_ptq_info:
            errors[name] = layer_ptq_info[name].get("quantization_error", 0.0)
        else:
            with torch.no_grad():
                W = module.weight.data.float()
                if isinstance(module, _HIER_TYPES):
                    from quantization import hierarchical_mlb_decompose
                    _, _, W_q = hierarchical_mlb_decompose(W, module.M_per_stage)
                else:
                    alphas, bases = mlb_decompose(W, module.M)
                    W_q = mlb_reconstruct(alphas, bases)
                errors[name] = quantization_error(W, W_q).item()
    return errors


def compute_total_quantization_error(
    model: nn.Module,
    layer_ptq_info: Optional[Dict[str, Dict]] = None,
) -> float:
    """
    ||W - W_MLB||_F  summed over all quantized layers.
    """
    return sum(compute_per_layer_quantization_errors(model, layer_ptq_info).values())


def compute_binary_op_count(model: nn.Module) -> int:
    """
    Estimate the total number of binary operations (XNOR-popcount pairs)
    when all MLB layers are executed using bitwise hardware.

    Approximation
    -------------
    For each MLB conv layer with M levels:
        binary_ops ≈ M × (2 × out_C × in_C × kH × kW × out_H × out_W)
    where output spatial dims are estimated for a 28×28 input.

    For linear layers:
        binary_ops ≈ M × (2 × in_F × out_F)

    Factor of 2: one XNOR + one popcount per weight.
    """
    total = 0
    # Estimate spatial output sizes for ResNet-20 on 28×28 input
    # conv1: 28×28, layer1: 28×28, layer2: 14×14, layer3: 7×7
    spatial_map = {
        "conv1":  (28, 28),
        "layer1": (28, 28),
        "layer2": (14, 14),
        "layer3": (7,  7),
        "fc":     (1,  1),
    }

    for name, module in _get_mlb_layers(model):
        M = module.M
        # Identify which group this layer belongs to
        group = "fc"
        for key in spatial_map:
            if key in name:
                group = key
                break

        if isinstance(module, MLBConv2d):
            oH, oW = spatial_map.get(group, (7, 7))
            out_C = module.out_channels
            in_C  = module.in_channels
            kH    = module.kernel_size
            kW    = module.kernel_size
            # ops per level: 2 × out_C × in_C × kH × kW × oH × oW
            ops_per_level = 2 * out_C * in_C * kH * kW * oH * oW
        else:  # MLBLinear
            ops_per_level = 2 * module.in_features * module.out_features

        total += M * ops_per_level

    return total


def compute_mac_reduction(model: nn.Module) -> float:
    """
    Estimated fraction of MACs replaced by binary ops (XNOR+popcount).

    mac_reduction = 1 − (binary_op_bits / fp32_op_bits)

    Where:
        fp32_op_bits  = total_binary_ops × (32 / M_avg)   [if done in FP32]
        binary_op_bits = total_binary_ops × 1              [1-bit XNOR]

    Simplification: MAC_reduction ≈ 1 − 1/32  per quantized weight.
    """
    m_avg = compute_effective_avg_M(model)
    if m_avg == 0.0:
        return 0.0  # FP32 — no MAC reduction
    # Fraction of original FP32 MACs saved by using 1-bit representations
    return 1.0 - (1.0 / 32.0)  # ≈ 96.875 % reduction in bit-level ops


# ===========================================================================
# Master metrics aggregator
# ===========================================================================

def compute_all_metrics(
    model: nn.Module,
    cfg: Config,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    train_time_seconds: float,
    val_acc_final: float,
    val_loss_final: float,
    layer_ptq_info: Optional[Dict[str, Dict]] = None,
    fp32_baseline_accuracy: Optional[float] = None,
    logger=None,
) -> Dict[str, Any]:
    """
    Compute all 13 required experiment metrics.

    Parameters
    ----------
    model                   : trained (and possibly quantized) model
    cfg                     : Config object
    val_loader              : validation DataLoader
    criterion               : loss function
    device                  : torch.device
    train_time_seconds      : wall-clock training time
    val_acc_final           : final validation accuracy  [0, 1]
    val_loss_final          : final validation loss
    layer_ptq_info          : dict from apply_ptq() (empty for FP32/QAT)
    fp32_baseline_accuracy  : optional known FP32 accuracy for acc-drop
    logger                  : optional logger

    Returns
    -------
    metrics : dict  {metric_name: value}
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    metrics: Dict[str, Any] = {}

    # 1. Final validation accuracy
    metrics["final_val_accuracy"] = val_acc_final

    # 2. Final validation loss
    metrics["final_val_loss"] = val_loss_final

    # 3. Training time
    metrics["training_time_seconds"] = train_time_seconds

    # 4. Inference time per image (ms)
    _log("  Benchmarking inference latency …")
    ms_per_image = measure_inference_time(model, device)
    metrics["inference_time_ms_per_image"] = ms_per_image

    # 5. Total parameter count
    total_params = sum(p.numel() for p in model.parameters())
    metrics["total_parameters"] = total_params

    # 6. Model size in MB (FP32)
    model_size_mb = total_params * 4 / (1024 ** 2)
    metrics["model_size_mb"] = model_size_mb

    # 7. Effective average MLB level
    m_avg = compute_effective_avg_M(model)
    metrics["effective_avg_M"] = m_avg

    # 8. Compression ratio
    cr = compute_compression_ratio(model)
    metrics["compression_ratio"] = cr

    # 9. Estimated binary operation count
    bin_ops = compute_binary_op_count(model)
    metrics["estimated_binary_op_count"] = bin_ops

    # 10. Estimated MAC reduction
    mac_red = compute_mac_reduction(model)
    metrics["estimated_mac_reduction"] = mac_red

    # 11. Total binary bases used
    total_bases = compute_total_binary_bases(model)
    metrics["total_binary_bases"] = total_bases

    # 12. Quantization error ||W - W_MLB||_F
    per_layer_errors = compute_per_layer_quantization_errors(model, layer_ptq_info)
    q_err = sum(per_layer_errors.values())
    metrics["quantization_error_frobenius"] = q_err

    # New: per-layer stats
    if per_layer_errors:
        metrics["average_quantization_error"] = q_err / len(per_layer_errors)
        metrics["max_layer_quantization_error"] = max(per_layer_errors.values())
        metrics["min_layer_quantization_error"] = min(per_layer_errors.values())
    else:
        metrics["average_quantization_error"] = 0.0
        metrics["max_layer_quantization_error"] = 0.0
        metrics["min_layer_quantization_error"] = 0.0

    # New: effective_M (alias for effective_avg_M for CSV readability)
    metrics["effective_M"] = m_avg

    # Hierarchical-specific metrics (stage1/stage2/combined errors)
    if cfg.mode.is_hierarchical:
        stage1_errors, stage2_errors = [], []
        for name, module in model.named_modules():
            if isinstance(module, _HIER_TYPES):
                if layer_ptq_info and name in layer_ptq_info:
                    stage1_errors.append(layer_ptq_info[name].get("stage1_error", 0.0))
                    stage2_errors.append(layer_ptq_info[name].get("stage2_error", 0.0))
                elif module._stage1_info is not None:
                    stage1_errors.append(module._stage1_info.get("error", 0.0))
                    stage2_errors.append(module._stage2_info.get("error", 0.0))
        metrics["stage1_error"]   = sum(stage1_errors)
        metrics["stage2_error"]   = sum(stage2_errors)
        metrics["combined_error"] = q_err
    else:
        metrics["stage1_error"]   = "N/A"
        metrics["stage2_error"]   = "N/A"
        metrics["combined_error"] = "N/A"

    # 13. Accuracy drop relative to FP32 baseline
    if fp32_baseline_accuracy is not None:
        acc_drop = fp32_baseline_accuracy - val_acc_final
    elif cfg.mode == ExperimentMode.A:
        acc_drop = 0.0
    else:
        acc_drop = None
    metrics["accuracy_drop_vs_fp32"] = acc_drop

    # Additional metadata
    metrics["experiment_name"]  = cfg.experiment_name
    metrics["mode"]             = cfg.mode.value
    metrics["mode_description"] = cfg.mode.description
    metrics["seed"]             = cfg.seed
    metrics["M"] = (
        cfg.M if not (cfg.mode.is_mixed_precision or cfg.mode.is_hierarchical)
        else ("mixed" if cfg.mode.is_mixed_precision else "hierarchical_8")
    )

    _log(f"  Metrics summary:")
    _log(f"    val_acc          = {val_acc_final * 100:.2f}%")
    _log(f"    inference_lat    = {ms_per_image:.3f} ms/image")
    _log(f"    model_size       = {model_size_mb:.2f} MB")
    _log(f"    avg_M            = {m_avg:.2f}")
    _log(f"    compression_ratio= {cr:.2f}×")
    _log(f"    quant_error      = {q_err:.4f}")
    if acc_drop is not None:
        _log(f"    acc_drop_vs_fp32 = {acc_drop * 100:.2f}%")

    return metrics


# ===========================================================================
# CSV savers
# ===========================================================================

def save_experiment_summary(metrics: Dict[str, Any], path: str):
    """
    Append the experiment metrics to *path* as a single CSV row.

    If the file does not exist, the header is written first.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Ordered fieldnames for the CSV
    fieldnames = [
        "experiment_name",
        "mode",
        "mode_description",
        "seed",
        "M",
        "effective_M",
        "final_val_accuracy",
        "final_val_loss",
        "accuracy_drop_vs_fp32",
        "training_time_seconds",
        "inference_time_ms_per_image",
        "total_parameters",
        "model_size_mb",
        "effective_avg_M",
        "compression_ratio",
        "estimated_binary_op_count",
        "estimated_mac_reduction",
        "total_binary_bases",
        "quantization_error_frobenius",
        "average_quantization_error",
        "max_layer_quantization_error",
        "min_layer_quantization_error",
        "stage1_error",
        "stage2_error",
        "combined_error",
    ]

    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        # Format floats nicely
        row = {}
        for k in fieldnames:
            v = metrics.get(k, "N/A")
            if isinstance(v, float):
                row[k] = f"{v:.6f}"
            elif v is None:
                row[k] = "N/A"
            else:
                row[k] = v
        writer.writerow(row)


def save_layer_report(
    model: nn.Module,
    cfg: Config,
    path: str,
    layer_ptq_info: Optional[Dict[str, Dict]] = None,
):
    """
    Write per-layer quantization details to *path*.

    Columns
    -------
    layer_name | M | num_parameters | quantization_error | compression_ratio
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "layer_name",
        "M",
        "num_parameters",
        "quantization_error",
        "compression_ratio",
    ]

    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, (MLBConv2d, MLBLinear)):
            continue

        M        = module.M
        n_params = module.weight.numel()

        # Quantization error
        if layer_ptq_info and name in layer_ptq_info:
            q_err = layer_ptq_info[name].get("quantization_error", 0.0)
        else:
            with torch.no_grad():
                W = module.weight.data.float()
                if isinstance(module, _HIER_TYPES):
                    from quantization import hierarchical_mlb_decompose
                    _, _, W_q = hierarchical_mlb_decompose(W, module.M_per_stage)
                else:
                    alphas, bases = mlb_decompose(W, M)
                    W_q = mlb_reconstruct(alphas, bases)
                q_err = quantization_error(W, W_q).item()

        # Compression ratio for this layer: 32 bits (FP32) / M bits (binary)
        cr = 32.0 / M if M > 0 else 1.0

        rows.append({
            "layer_name":        name,
            "M":                 M,
            "num_parameters":    n_params,
            "quantization_error": f"{q_err:.6f}",
            "compression_ratio":  f"{cr:.4f}",
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Training plots
# ===========================================================================

def generate_training_plots(log_csv_path: str, plots_dir: str, experiment_name: str):
    """
    Read training_log.csv and generate four PNG plots:
        1. Training loss vs epoch
        2. Validation loss vs epoch
        3. Training accuracy vs epoch
        4. Validation accuracy vs epoch

    Uses matplotlib with a clean, publication-quality style.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for server environments
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[metrics] matplotlib not available – skipping plots.")
        return

    if not os.path.isfile(log_csv_path):
        print(f"[metrics] training log not found at {log_csv_path} – skipping plots.")
        return

    # Read CSV
    epochs, train_losses, val_losses, train_accs, val_accs = [], [], [], [], []
    with open(log_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_accs.append(float(row["train_accuracy"]) * 100)
            val_accs.append(float(row["val_accuracy"]) * 100)

    if not epochs:
        return

    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ---- Colour palette ----
    C_TRAIN = "#4C9BE8"   # steel blue
    C_VAL   = "#E8774C"   # coral

    # ---- Style ----
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "font.family":       "DejaVu Sans",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })

    def _save_fig(fig, fname: str):
        out = os.path.join(plots_dir, fname)
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] saved: {out}")

    # ------------------------------------------------------------------
    # 1. Training loss
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, color=C_TRAIN, lw=2, label="Train Loss")
    ax.set_title(f"{experiment_name} — Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    _save_fig(fig, f"{experiment_name}_train_loss.png")

    # ------------------------------------------------------------------
    # 2. Validation loss
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, val_losses, color=C_VAL, lw=2, label="Val Loss")
    ax.set_title(f"{experiment_name} — Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    _save_fig(fig, f"{experiment_name}_val_loss.png")

    # ------------------------------------------------------------------
    # 3. Training accuracy
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_accs, color=C_TRAIN, lw=2, label="Train Acc")
    ax.set_title(f"{experiment_name} — Training Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    _save_fig(fig, f"{experiment_name}_train_acc.png")

    # ------------------------------------------------------------------
    # 4. Validation accuracy
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, val_accs, color=C_VAL, lw=2, label="Val Acc")
    ax.set_title(f"{experiment_name} — Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.legend()
    _save_fig(fig, f"{experiment_name}_val_acc.png")

    # ------------------------------------------------------------------
    # 5. Combined loss + accuracy overview (bonus)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax_l, ax_a = axes

    ax_l.plot(epochs, train_losses, color=C_TRAIN, lw=2, label="Train")
    ax_l.plot(epochs, val_losses,   color=C_VAL,   lw=2, linestyle="--", label="Val")
    ax_l.set_title("Loss")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Cross-Entropy Loss")
    ax_l.legend()

    ax_a.plot(epochs, train_accs, color=C_TRAIN, lw=2, label="Train")
    ax_a.plot(epochs, val_accs,   color=C_VAL,   lw=2, linestyle="--", label="Val")
    ax_a.set_title("Accuracy")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax_a.legend()

    fig.suptitle(f"{experiment_name}", fontsize=14, fontweight="bold")
    _save_fig(fig, f"{experiment_name}_overview.png")
