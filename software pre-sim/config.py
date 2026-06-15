"""
config.py
=========
Central configuration for all MLB-quantization experiments.

Experiment Modes
----------------
    A  --  FP32 baseline (no quantization)
    B  --  Uniform MLB, inference-only  (PTQ)
    C  --  Uniform MLB, QAT
    D  --  Mixed-precision MLB, inference-only  (PTQ)
    E  --  Mixed-precision MLB, QAT
    F  --  Hierarchical M=8 MLB, inference-only (PTQ)
    G  --  Hierarchical M=8 MLB, QAT

Change log
----------
v2: Updated default mixed-precision allocation to sensitivity-guided
    ordering (deeper layers get higher M).
    Added Modes F and G for hierarchical M=8.
    Added timestamped per-run result directories.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, Optional


# ===========================================================================
# Mode enumeration
# ===========================================================================

class ExperimentMode(str, Enum):
    """The seven experiment modes supported by the framework."""
    A = "A"   # FP32 baseline
    B = "B"   # Uniform MLB PTQ
    C = "C"   # Uniform MLB QAT
    D = "D"   # Mixed-precision MLB PTQ
    E = "E"   # Mixed-precision MLB QAT
    F = "F"   # Hierarchical M=8 MLB PTQ
    G = "G"   # Hierarchical M=8 MLB QAT

    @property
    def is_quantized(self) -> bool:
        return self != ExperimentMode.A

    @property
    def is_qat(self) -> bool:
        return self in (ExperimentMode.C, ExperimentMode.E, ExperimentMode.G)

    @property
    def is_mixed_precision(self) -> bool:
        return self in (ExperimentMode.D, ExperimentMode.E)

    @property
    def is_hierarchical(self) -> bool:
        """True for modes that use hierarchical M=8 decomposition."""
        return self in (ExperimentMode.F, ExperimentMode.G)

    @property
    def description(self) -> str:
        _desc = {
            "A": "FP32 Baseline",
            "B": "Uniform MLB -- Inference-Only (PTQ)",
            "C": "Uniform MLB -- Quantization-Aware Training (QAT)",
            "D": "Mixed-Precision MLB -- Inference-Only (PTQ)",
            "E": "Mixed-Precision MLB -- Quantization-Aware Training (QAT)",
            "F": "Hierarchical M=8 MLB -- Inference-Only (PTQ)",
            "G": "Hierarchical M=8 MLB -- Quantization-Aware Training (QAT)",
        }
        return _desc[self.value]


# ===========================================================================
# Mixed-precision per-layer M allocation
# ===========================================================================

# Sensitivity-guided allocation:
# Earlier layers (conv1, layer1) are less sensitive to quantization and get
# lower M; deeper layers (layer3, fc) are more sensitive and get higher M.
# This is the REVERSE of the naive ordering and is the empirically better
# choice validated by layer-wise quantization-error analysis.
DEFAULT_MIXED_PRECISION_CONFIG: Dict[str, int] = {
    "conv1":  2,
    "layer1": 3,
    "layer2": 4,
    "layer3": 5,
    "fc":     5,
}


# ===========================================================================
# Main configuration dataclass
# ===========================================================================

@dataclass
class Config:
    # -----------------------------------------------------------------------
    # Experiment identity
    # -----------------------------------------------------------------------
    experiment_name: str = "mlb_experiment"
    mode: ExperimentMode = ExperimentMode.A

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    data_dir: str = "./data"
    batch_size: int = 128
    val_split: float = 0.2
    num_workers: int = 2
    pin_memory: bool = True

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    num_classes: int = 10

    # -----------------------------------------------------------------------
    # Uniform MLB
    # -----------------------------------------------------------------------
    M: int = 4                      # uniform binary levels for modes B / C

    # -----------------------------------------------------------------------
    # Mixed-precision MLB (modes D / E)
    # -----------------------------------------------------------------------
    mixed_precision_config: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_MIXED_PRECISION_CONFIG)
    )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    num_epochs: int = 100

    # LR scheduler
    scheduler: str = "cosine"
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    lr_min: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4

    # -----------------------------------------------------------------------
    # Checkpointing & resumption
    # -----------------------------------------------------------------------
    resume: bool = False
    resume_checkpoint: str = ""

    # -----------------------------------------------------------------------
    # Output directories
    # -----------------------------------------------------------------------
    # Base results root.  Each run automatically creates a timestamped subdir
    # under this root so no two runs ever overwrite each other.
    results_root: str = "./results"

    # Timestamped directory for this specific run (auto-set in __post_init__)
    results_dir: str = ""

    def __post_init__(self):
        # Auto-create a unique timestamped directory for this run if not set.
        if not self.results_dir:
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.results_dir = os.path.join(
                self.results_root,
                f"{ts}_{self.experiment_name}",
            )

    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.results_dir, "checkpoints")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.results_dir, "logs")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.results_dir, "plots")

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.results_dir, "metrics")

    @property
    def comparison_runs_dir(self) -> str:
        """Shared directory for aggregate comparison CSVs."""
        return os.path.join(self.results_root, "comparison_runs")

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------
    seed: int = 42

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device: str = "auto"

    # -----------------------------------------------------------------------
    # Computed helpers
    # -----------------------------------------------------------------------

    def resolve_device(self) -> str:
        """Return the actual torch device string."""
        if self.device == "auto":
            if __import__("torch").cuda.is_available():
                return "cuda"
            try:
                if __import__("torch").backends.mps.is_available():
                    return "mps"
            except AttributeError:
                pass
            return "cpu"
        return self.device

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    def save_json(self, path: str):
        """Persist the full config to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        d["mode"] = ExperimentMode(d["mode"])
        # results_dir is already set in the dict; skip __post_init__ re-generation
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


# ===========================================================================
# Command-line argument parser
# ===========================================================================

def parse_args(argv=None) -> Config:
    """
    Parse command-line arguments and return a populated Config.

    Examples
    --------
    python train.py --mode A
    python train.py --mode B --M 3
    python train.py --mode C --M 4 --epochs 80
    python train.py --mode D
    python train.py --mode E --lr 5e-4 --batch-size 64
    python train.py --mode F
    python train.py --mode G --epochs 100
    """
    parser = argparse.ArgumentParser(
        description="MLB Quantization Research Framework -- ResNet-20 / Fashion-MNIST"
    )

    # Experiment
    parser.add_argument(
        "--mode", type=str, default="A",
        choices=["A", "B", "C", "D", "E", "F", "G"],
        help=(
            "Experiment mode: "
            "A=FP32, "
            "B=Uniform-MLB-PTQ, "
            "C=Uniform-MLB-QAT, "
            "D=Mixed-MLB-PTQ, "
            "E=Mixed-MLB-QAT, "
            "F=Hierarchical-M8-PTQ, "
            "G=Hierarchical-M8-QAT"
        ),
    )
    parser.add_argument("--name", type=str, default="mlb_experiment",
                        help="Experiment name (used for naming output dirs/files).")

    # Dataset
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)

    # Uniform MLB
    parser.add_argument("--M", type=int, default=4,
                        help="Uniform binary levels for modes B/C.")

    # Mixed-precision MLB (JSON string or path)
    parser.add_argument(
        "--mp-config", type=str, default=None,
        help=(
            'Mixed-precision config as a JSON string, e.g. '
            '\'{"conv1":2,"layer1":3,"layer2":4,"layer3":5,"fc":5}\''
            ' or a path to a JSON file.'
        ),
    )

    # Training
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=100)

    # Scheduler
    parser.add_argument(
        "--scheduler", type=str, default="cosine",
        choices=["cosine", "step", "plateau", "none"],
    )
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--lr-min", type=float, default=1e-6)

    # Early stopping
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)

    # Checkpointing
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=str, default="")

    # Output
    parser.add_argument("--results-root", type=str, default="./results",
                        help="Root directory under which timestamped run dirs are created.")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Build Config from parsed args
    # ------------------------------------------------------------------
    mp_config = DEFAULT_MIXED_PRECISION_CONFIG.copy()
    if args.mp_config is not None:
        raw = args.mp_config.strip()
        if raw.startswith("{"):
            mp_config = json.loads(raw)
        else:
            with open(raw) as f:
                mp_config = json.load(f)

    # Derive experiment_name from mode if using default name
    exp_name = args.name if args.name != "mlb_experiment" else f"mode_{args.mode}"

    cfg = Config(
        experiment_name=exp_name,
        mode=ExperimentMode(args.mode),
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        M=args.M,
        mixed_precision_config=mp_config,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        num_epochs=args.epochs,
        scheduler=args.scheduler,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        lr_min=args.lr_min,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        min_delta=args.min_delta,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        results_root=args.results_root,
        seed=args.seed,
        device=args.device,
    )
    return cfg
