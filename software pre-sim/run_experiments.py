"""
run_experiments.py
==================
Convenience script to run any subset (or all) of the seven MLB-quantization
experiment modes sequentially and produce a consolidated comparison CSV.

Mode table
----------
    A  =  FP32 Baseline
    B  =  Uniform MLB PTQ
    C  =  Uniform MLB QAT
    D  =  Mixed-Precision MLB PTQ
    E  =  Mixed-Precision MLB QAT
    F  =  Hierarchical M=8 PTQ
    G  =  Hierarchical M=8 QAT

Usage
-----
    # Run all seven modes:
    python run_experiments.py --all --epochs 100

    # Run specific modes:
    python run_experiments.py --modes A C E G --epochs 50

    # Run with custom M for uniform modes:
    python run_experiments.py --modes B C --M 3 --epochs 80

Outputs
-------
    results/
    ├── <timestamp>_mode_A/
    ├── <timestamp>_mode_C/
    ...
    └── comparison_runs/
        └── comparison_<timestamp>.csv   <- sorted by val accuracy
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Optional

from config import Config, ExperimentMode, DEFAULT_MIXED_PRECISION_CONFIG
from train import train
from utils import set_seed, get_logger


# ===========================================================================
# CLI
# ===========================================================================

def parse_run_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run MLB-quantization experiments (one or all modes)."
    )

    # Which modes to run
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all", action="store_true",
        help="Run all seven modes (A, B, C, D, E, F, G) sequentially.",
    )
    group.add_argument(
        "--modes", nargs="+", choices=["A", "B", "C", "D", "E", "F", "G"],
        help="One or more mode letters to run.",
    )

    # Shared hyperparameters
    parser.add_argument("--M",          type=int,   default=4,
                        help="Uniform binary levels for modes B / C.")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--optimizer",  type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--scheduler",  type=str,   default="cosine",
                        choices=["cosine", "step", "plateau", "none"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--results-root", type=str, default="./results",
                        help="Root directory for timestamped experiment runs.")
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--device",     type=str,   default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int,  default=2)

    return parser.parse_args(argv)


# ===========================================================================
# Build a Config for a specific mode (with its own timestamped results_dir)
# ===========================================================================

def make_config_for_mode(args, mode_letter: str, run_timestamp: str) -> Config:
    """
    Construct a Config for *mode_letter*.

    Each mode gets a unique timestamped directory so runs never overwrite
    each other.  The timestamp is shared across all modes in a single
    `run_experiments.py` invocation so the folders sort together.
    """
    mode = ExperimentMode(mode_letter)
    exp_name = f"mode_{mode_letter}"

    # Build a Config with results_dir pre-set (bypasses __post_init__ auto-ts)
    cfg = Config(
        experiment_name=exp_name,
        mode=mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        M=args.M,
        mixed_precision_config=DEFAULT_MIXED_PRECISION_CONFIG.copy(),
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        num_epochs=args.epochs,
        patience=args.patience,
        early_stopping=True,
        results_root=args.results_root,
        # Pre-set the timestamped directory
        results_dir=os.path.join(
            args.results_root, f"{run_timestamp}_{exp_name}"
        ),
        seed=args.seed,
        device=args.device,
    )
    return cfg


# ===========================================================================
# Comparison CSV (Change 6: sorted by val accuracy)
# ===========================================================================

COMPARISON_FIELDS = [
    "experiment_name",
    "mode",
    "mode_description",
    "effective_M",
    "final_val_accuracy",
    "accuracy_drop_vs_fp32",
    "model_size_mb",
    "compression_ratio",
    "quantization_error_frobenius",
    "inference_time_ms_per_image",
    # Extra detail columns
    "average_quantization_error",
    "max_layer_quantization_error",
    "min_layer_quantization_error",
    "stage1_error",
    "stage2_error",
    "combined_error",
    "training_time_seconds",
    "total_parameters",
]


def save_comparison_csv(all_metrics: List[dict], out_dir: str, run_timestamp: str) -> str:
    """
    Write a side-by-side comparison CSV sorted by descending validation accuracy.
    Written to results/comparison_runs/comparison_<timestamp>.csv.
    Never overwrites a previous comparison file.
    """
    comp_dir = os.path.join(out_dir, "comparison_runs")
    Path(comp_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(comp_dir, f"comparison_{run_timestamp}.csv")

    # Sort by final_val_accuracy descending (errors / N/A go to bottom)
    def _sort_key(m):
        v = m.get("final_val_accuracy", 0)
        try:
            return -float(v)
        except (TypeError, ValueError):
            return float("inf")

    sorted_metrics = sorted(all_metrics, key=_sort_key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for m in sorted_metrics:
            row = {}
            for k in COMPARISON_FIELDS:
                v = m.get(k, "N/A")
                if isinstance(v, float):
                    row[k] = f"{v:.6f}"
                elif v is None:
                    row[k] = "N/A"
                else:
                    row[k] = v
            writer.writerow(row)

    print(f"\n[run_experiments] Comparison CSV -> {path}")
    return path


# ===========================================================================
# Main runner
# ===========================================================================

def main(argv=None):
    args = parse_run_args(argv)

    modes: List[str] = (
        ["A", "B", "C", "D", "E", "F", "G"] if args.all else args.modes
    )

    # Shared timestamp for this entire run batch (modes sort together)
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Shared logger (goes to results_root/logs/)
    log_dir = os.path.join(args.results_root, "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        "run_experiments",
        log_file=os.path.join(log_dir, f"run_experiments_{run_timestamp}.log"),
    )

    logger.info("=" * 70)
    logger.info(f"  MLB Quantization -- running modes: {modes}")
    logger.info(f"  Results root: {args.results_root}")
    logger.info(f"  Run timestamp: {run_timestamp}")
    logger.info("=" * 70)

    all_metrics: List[dict] = []
    fp32_accuracy: Optional[float] = None

    for mode_letter in modes:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Starting MODE {mode_letter}  ({ExperimentMode(mode_letter).description})")
        logger.info(f"{'='*70}")

        cfg = make_config_for_mode(args, mode_letter, run_timestamp)
        set_seed(cfg.seed)

        try:
            _, metrics = train(cfg, logger=logger)
        except Exception as e:
            logger.error(f"  Mode {mode_letter} FAILED: {e}", exc_info=True)
            metrics = {
                "experiment_name":   f"mode_{mode_letter}",
                "mode":              mode_letter,
                "mode_description":  ExperimentMode(mode_letter).description,
                "M":                 cfg.M,
                "final_val_accuracy": "ERROR",
            }
            all_metrics.append(metrics)
            continue

        # Record FP32 baseline accuracy for acc-drop calculation
        if mode_letter == "A":
            fp32_accuracy = metrics.get("final_val_accuracy")

        # Backfill accuracy_drop_vs_fp32 now that we have the FP32 baseline
        if fp32_accuracy is not None and mode_letter != "A":
            val_acc = metrics.get("final_val_accuracy")
            if isinstance(val_acc, float):
                metrics["accuracy_drop_vs_fp32"] = fp32_accuracy - val_acc

        all_metrics.append(metrics)
        logger.info(
            f"  Mode {mode_letter} complete.  "
            f"val_acc={metrics.get('final_val_accuracy', 'N/A')}  "
            f"results -> {cfg.results_dir}"
        )

    # ------------------------------------------------------------------
    # Consolidated comparison (sorted by val accuracy)
    # ------------------------------------------------------------------
    if all_metrics:
        save_comparison_csv(all_metrics, args.results_root, run_timestamp)

    logger.info("\n  All requested modes completed.")
    return all_metrics


if __name__ == "__main__":
    main()
