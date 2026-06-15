"""
run_experiments.py
==================
Convenience script to run one or all five MLB-quantization experiment modes
sequentially and produce a consolidated comparison CSV.

Usage
-----
    # Run a single mode (e.g. mode C with M=4):
    python run_experiments.py --modes C --M 4 --epochs 60

    # Run all five modes with defaults:
    python run_experiments.py --all

    # Run modes A, B, C only:
    python run_experiments.py --modes A B C

    # Full research run (all modes, 100 epochs):
    python run_experiments.py --all --epochs 100

Outputs
-------
    results/
    ├── mode_A/   (each mode gets its own sub-directory)
    ├── mode_B/
    ├── mode_C/
    ├── mode_D/
    ├── mode_E/
    └── comparison_summary.csv   ← consolidated across all modes
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

# Local imports
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
        help="Run all five modes (A, B, C, D, E) sequentially.",
    )
    group.add_argument(
        "--modes", nargs="+", choices=["A", "B", "C", "D", "E"],
        help="One or more mode letters to run.",
    )

    # Shared hyperparameters
    parser.add_argument("--M",        type=int,   default=4,
                        help="Uniform binary levels for modes B / C.")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--optimizer",type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--scheduler",type=str,   default="cosine",
                        choices=["cosine", "step", "plateau", "none"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int,   default=15)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--data-dir",    type=str, default="./data")
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=2)

    return parser.parse_args(argv)


# ===========================================================================
# Build a Config for a specific mode
# ===========================================================================

def make_config_for_mode(args, mode_letter: str) -> Config:
    """Construct a Config for *mode_letter* from parsed CLI args."""
    mode = ExperimentMode(mode_letter)
    results_dir = os.path.join(args.results_dir, f"mode_{mode_letter}")
    return Config(
        experiment_name=f"mode_{mode_letter}",
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
        results_dir=results_dir,
        seed=args.seed,
        device=args.device,
    )


# ===========================================================================
# Comparison CSV
# ===========================================================================

COMPARISON_FIELDS = [
    "mode",
    "mode_description",
    "M",
    "effective_avg_M",
    "final_val_accuracy",
    "final_val_loss",
    "training_time_seconds",
    "inference_time_ms_per_image",
    "total_parameters",
    "model_size_mb",
    "compression_ratio",
    "quantization_error_frobenius",
    "accuracy_drop_vs_fp32",
    "estimated_binary_op_count",
    "estimated_mac_reduction",
    "total_binary_bases",
]


def save_comparison_csv(all_metrics: List[dict], out_dir: str):
    """Write a side-by-side comparison of all modes to comparison_summary.csv."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, "comparison_summary.csv")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for m in all_metrics:
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

    print(f"\n[run_experiments] Comparison CSV saved → {path}")
    return path


# ===========================================================================
# Main runner
# ===========================================================================

def main(argv=None):
    args = parse_run_args(argv)

    modes: List[str] = ["A", "B", "C", "D", "E"] if args.all else args.modes

    # Shared logger
    log_dir = os.path.join(args.results_dir, "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        "run_experiments",
        log_file=os.path.join(log_dir, "run_experiments.log"),
    )

    logger.info("=" * 70)
    logger.info(f"  MLB Quantization — running modes: {modes}")
    logger.info(f"  Results root: {args.results_dir}")
    logger.info("=" * 70)

    all_metrics: List[dict] = []
    fp32_accuracy: Optional[float] = None   # record Mode A accuracy for acc-drop

    for mode_letter in modes:
        logger.info(f"\n{'='*70}")
        logger.info(f"  Starting MODE {mode_letter}")
        logger.info(f"{'='*70}")

        cfg = make_config_for_mode(args, mode_letter)
        set_seed(cfg.seed)

        try:
            _, metrics = train(cfg, logger=logger)
        except Exception as e:
            logger.error(f"  Mode {mode_letter} FAILED: {e}", exc_info=True)
            # Record a placeholder row so comparison CSV still has all modes
            metrics = {
                "mode":             mode_letter,
                "mode_description": ExperimentMode(mode_letter).description,
                "M":                cfg.M,
                "final_val_accuracy": "ERROR",
            }
            all_metrics.append(metrics)
            continue

        # Capture FP32 baseline accuracy
        if mode_letter == "A":
            fp32_accuracy = metrics.get("final_val_accuracy")

        # Backfill accuracy_drop_vs_fp32 if we now know the FP32 baseline
        if fp32_accuracy is not None and mode_letter != "A":
            val_acc = metrics.get("final_val_accuracy")
            if isinstance(val_acc, float):
                metrics["accuracy_drop_vs_fp32"] = fp32_accuracy - val_acc

        all_metrics.append(metrics)
        logger.info(
            f"  Mode {mode_letter} complete. "
            f"val_acc={metrics.get('final_val_accuracy', 'N/A')}"
        )

    # ------------------------------------------------------------------
    # Consolidated comparison
    # ------------------------------------------------------------------
    if len(all_metrics) > 1:
        save_comparison_csv(all_metrics, args.results_dir)

    logger.info("\n  All requested modes completed.")
    return all_metrics


if __name__ == "__main__":
    main()
