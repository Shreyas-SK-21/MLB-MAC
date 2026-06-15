"""
train.py
========
Training engine for all five MLB-quantization experiment modes.

Usage (command line)
--------------------
    # Mode A – FP32 baseline
    python train.py --mode A

    # Mode B – Uniform MLB, post-training quantization (M=3)
    python train.py --mode B --M 3

    # Mode C – Uniform MLB, QAT (M=4)
    python train.py --mode C --M 4 --epochs 80

    # Mode D – Mixed-precision MLB, PTQ
    python train.py --mode D

    # Mode E – Mixed-precision MLB, QAT
    python train.py --mode E --lr 5e-4

    # Resume from checkpoint
    python train.py --mode C --M 4 --resume --resume-checkpoint results/checkpoints/ckpt_epoch_50.pth

Training outputs (written to results/)
---------------------------------------
    checkpoints/   – per-epoch + best-model .pth files
    logs/          – training_log.csv, experiment config JSON, text log
    plots/         – PNG plots auto-generated after training
    metrics/       – experiment_summary.csv, layer_quantization_report.csv
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local modules
from config import Config, ExperimentMode, parse_args
from resnet20 import (
    build_fp32_resnet20,
    build_uniform_mlb_resnet20,
    build_mixed_precision_resnet20,
)
from quantization import apply_ptq, set_qat_mode
from utils import (
    CSVLogger,
    Timer,
    create_output_dirs,
    get_data_loaders,
    get_logger,
    load_checkpoint,
    progress_bar,
    save_checkpoint,
    save_run_config,
    set_seed,
)
from evaluate import evaluate_model
from metrics import compute_all_metrics, save_experiment_summary, save_layer_report


# ===========================================================================
# Model factory
# ===========================================================================

def build_model(cfg: Config) -> nn.Module:
    """
    Instantiate the appropriate ResNet-20 variant for the given mode.

    Mode A → FP32
    Mode B → Uniform MLB (PTQ, so qat=False at build time)
    Mode C → Uniform MLB (QAT enabled)
    Mode D → Mixed-precision MLB (PTQ)
    Mode E → Mixed-precision MLB (QAT enabled)
    """
    mode = cfg.mode
    if mode == ExperimentMode.A:
        return build_fp32_resnet20(num_classes=cfg.num_classes)
    elif mode in (ExperimentMode.B, ExperimentMode.C):
        return build_uniform_mlb_resnet20(
            M=cfg.M,
            qat=mode.is_qat,
            num_classes=cfg.num_classes,
        )
    else:  # D or E
        return build_mixed_precision_resnet20(
            mixed_precision_config=cfg.mixed_precision_config,
            qat=mode.is_qat,
            num_classes=cfg.num_classes,
        )


# ===========================================================================
# Optimizer factory
# ===========================================================================

def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Return Adam or SGD based on cfg.optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}. Choose 'adam' or 'sgd'.")


# ===========================================================================
# LR scheduler factory
# ===========================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> Optional[object]:
    """Return a learning-rate scheduler (or None for 'none')."""
    name = cfg.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.num_epochs,
            eta_min=cfg.lr_min,
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.lr_step_size,
            gamma=cfg.lr_gamma,
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_gamma,
            patience=cfg.patience // 2,
            min_lr=cfg.lr_min,
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler!r}.")


# ===========================================================================
# Early stopping
# ===========================================================================

class EarlyStopping:
    """
    Monitor validation loss and stop training when no improvement is
    observed for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update state.

        Returns True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ===========================================================================
# Single training epoch
# ===========================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger,
) -> Tuple[float, float]:
    """
    Run one full epoch of training.

    Returns
    -------
    avg_loss : float   – mean cross-entropy loss over all batches
    accuracy : float   – fraction of correctly classified samples [0, 1]
    """
    model.train()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0
    n_batches     = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Accumulate statistics
        total_loss    += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

        # Periodic console progress
        if (batch_idx + 1) % max(1, n_batches // 5) == 0 or (batch_idx + 1) == n_batches:
            bar = progress_bar(batch_idx + 1, n_batches,
                               f"loss={loss.item():.4f}")
            logger.debug(f"  Epoch {epoch:3d}  Batch {batch_idx+1:4d}/{n_batches}  {bar}")

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ===========================================================================
# Main training loop
# ===========================================================================

def train(cfg: Config, logger=None):
    """
    Full training pipeline for the given Config.

    Steps
    -----
    1. Set up directories, seed, device.
    2. Build datasets and data loaders.
    3. Build model (variant depends on cfg.mode).
    4. Optionally resume from checkpoint.
    5. Build optimizer, scheduler, early stopper.
    6. Run training epochs (with per-epoch CSV logging).
    7. For PTQ modes (B, D): apply post-training quantization after training.
    8. Final evaluation on validation set.
    9. Compute and save all metrics / summary.
    10. Generate plots.

    Parameters
    ----------
    cfg    : populated Config object
    logger : optional pre-configured logger; one will be created if None

    Returns
    -------
    model     : trained (and possibly quantized) nn.Module
    metrics   : dict of final metrics
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    create_output_dirs(cfg)
    set_seed(cfg.seed)
    device = torch.device(cfg.resolve_device())

    if logger is None:
        log_file = os.path.join(cfg.logs_dir, f"{cfg.experiment_name}.log")
        logger   = get_logger(cfg.experiment_name, log_file=log_file)

    logger.info("=" * 70)
    logger.info(f"  MLB Quantization Experiment")
    logger.info(f"  Mode  : {cfg.mode.value} — {cfg.mode.description}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Seed  : {cfg.seed}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Loading Fashion-MNIST …")
    train_loader, val_loader, test_loader = get_data_loaders(cfg)
    logger.info(
        f"  Train={len(train_loader.dataset):,}  "
        f"Val={len(val_loader.dataset):,}  "
        f"Test={len(test_loader.dataset):,}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Building model (mode={cfg.mode.value}) …")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser / scheduler / loss
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    stopper   = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # ------------------------------------------------------------------
    # (Optional) Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch   = 1
    best_val_acc  = 0.0
    best_val_loss = float("inf")

    if cfg.resume and cfg.resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {cfg.resume_checkpoint}")
        ckpt = load_checkpoint(cfg.resume_checkpoint, str(device))
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_acc  = ckpt.get("val_acc",  0.0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(f"  Resumed from epoch {start_epoch - 1}  "
                    f"best_val_acc={best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # CSV logger (per-epoch metrics)
    # ------------------------------------------------------------------
    log_csv_path = os.path.join(cfg.logs_dir, "training_log.csv")
    csv_logger   = CSVLogger(
        log_csv_path,
        fieldnames=["epoch", "train_loss", "train_accuracy",
                    "val_loss", "val_accuracy", "learning_rate"],
    )

    best_ckpt_path = os.path.join(cfg.checkpoints_dir, "best_model.pth")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info(f"Starting training for up to {cfg.num_epochs} epochs …")
    wall_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        # ---- Train ----
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )
        train_time = time.perf_counter() - t0

        # ---- Validate ----
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # ---- LR scheduler step ----
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ---- Logging ----
        logger.info(
            f"Epoch {epoch:3d}/{cfg.num_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  ({train_time:.1f}s)"
        )

        csv_logger.write_row({
            "epoch":          epoch,
            "train_loss":     f"{train_loss:.6f}",
            "train_accuracy": f"{train_acc:.6f}",
            "val_loss":       f"{val_loss:.6f}",
            "val_accuracy":   f"{val_acc:.6f}",
            "learning_rate":  f"{current_lr:.8f}",
        })

        # ---- Checkpoint ----
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc  = val_acc
            best_val_loss = val_loss

        ckpt_path = os.path.join(cfg.checkpoints_dir, f"ckpt_epoch_{epoch:04d}.pth")
        save_checkpoint(
            state={
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "val_acc":         val_acc,
                "val_loss":        val_loss,
                "config":          cfg.to_dict(),
            },
            filepath=ckpt_path,
            is_best=is_best,
            best_filepath=best_ckpt_path,
        )

        # ---- Early stopping ----
        if cfg.early_stopping and stopper.step(val_loss):
            logger.info(
                f"Early stopping triggered at epoch {epoch} "
                f"(no improvement for {cfg.patience} epochs)."
            )
            break

    total_train_time = time.perf_counter() - wall_start
    csv_logger.close()
    logger.info(f"Training complete.  Total wall time: {total_train_time:.1f}s")

    # ------------------------------------------------------------------
    # Reload best model weights for evaluation
    # ------------------------------------------------------------------
    if os.path.isfile(best_ckpt_path):
        logger.info("Loading best model for final evaluation …")
        best_ckpt = load_checkpoint(best_ckpt_path, str(device))
        model.load_state_dict(best_ckpt["model_state"])

    # ------------------------------------------------------------------
    # Post-Training Quantization (PTQ modes B and D)
    # ------------------------------------------------------------------
    if cfg.mode in (ExperimentMode.B, ExperimentMode.D):
        logger.info("Applying post-training MLB quantization …")
        layer_ptq_info = apply_ptq(model)
        logger.info(f"  Quantized {len(layer_ptq_info)} MLB layers.")
    else:
        layer_ptq_info = {}

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    logger.info("Final evaluation on validation set …")
    val_loss_final, val_acc_final = evaluate_model(
        model, val_loader, criterion, device
    )
    logger.info(
        f"  Final val_loss={val_loss_final:.4f}  val_acc={val_acc_final:.4f}"
    )

    # ------------------------------------------------------------------
    # Compute & save all metrics
    # ------------------------------------------------------------------
    logger.info("Computing experiment metrics …")
    metrics = compute_all_metrics(
        model=model,
        cfg=cfg,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        train_time_seconds=total_train_time,
        val_acc_final=val_acc_final,
        val_loss_final=val_loss_final,
        layer_ptq_info=layer_ptq_info,
        logger=logger,
    )

    summary_path = os.path.join(cfg.metrics_dir, "experiment_summary.csv")
    save_experiment_summary(metrics, summary_path)
    logger.info(f"  Experiment summary → {summary_path}")

    layer_report_path = os.path.join(cfg.metrics_dir, "layer_quantization_report.csv")
    save_layer_report(model, cfg, layer_report_path, layer_ptq_info)
    logger.info(f"  Layer quantization report → {layer_report_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("Generating training plots …")
    from metrics import generate_training_plots
    generate_training_plots(log_csv_path, cfg.plots_dir, cfg.experiment_name)
    logger.info(f"  Plots saved to {cfg.plots_dir}")

    # ------------------------------------------------------------------
    # Save run config (JSON)
    # ------------------------------------------------------------------
    config_path = save_run_config(cfg)
    logger.info(f"  Run config → {config_path}")

    logger.info("=" * 70)
    logger.info(f"  DONE.  Final val accuracy: {val_acc_final * 100:.2f}%")
    logger.info("=" * 70)

    return model, metrics


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
