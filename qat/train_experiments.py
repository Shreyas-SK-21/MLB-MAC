"""
train_experiments.py
====================
Unified Training & Evaluation Engine for ResNet-20 on CIFAR-10.

Research Credibility Protections Implemented:
---------------------------------------------
1. ZERO Test-Set Leakage:
   - 50,000 CIFAR-10 training images are partitioned into:
     * 45,000 Training samples (with RandomCrop & RandomHorizontalFlip).
     * 5,000 Validation samples (deterministic subset, evaluation transform only).
   - Checkpoint selection is strictly guided by VALIDATION accuracy.
   - The 10,000-sample test set is evaluated EXACTLY ONCE at the conclusion of training
     using the validation-selected best checkpoint.
   - Both `best_val_accuracy` and `unbiased_test_accuracy` are recorded in the CSV.

2. Fail-Fast Checkpoint Integrity:
   - Fine-tuning modes (`bfp_finetune_20`) require an explicit prior checkpoint.
   - If the checkpoint is missing, the script raises an immediate FileNotFoundError.
   - Never silently falls back to scratch training under a fine-tuning label.

3. Verified FP8 E4M3 Arithmetic:
   - Supports IEEE P3109 / OCP FP8 E4M3 with 448.0 max finite saturation and proper
     exponent carry-over rounding.
   - Selectable via `--fp8-format {E4M3, E3M4}`.

4. Exponent Sorting as an Empirical BFP Approximation:
   - Documents that sorting re-indexes block membership across block boundaries,
     clustering similar exponents to reduce BFP truncation shift (E_max - e_k).

5. Automated Multi-Seed Statistical Aggregation:
   - Multiple seeds (`--seeds 42,43,44`) automatically compute and log mean +/- std
     into `qat_aggregated_metrics.csv`.
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from resnet20 import ResNet20, count_parameters
from quantize import set_global_fp8_format

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cifar10_loaders(data_dir="./data", batch_size=128, val_split=5000, num_workers=2):
    """
    Returns (train_loader, val_loader, test_loader).
    - train_loader: 45,000 samples with data augmentation.
    - val_loader: 5,000 samples without data augmentation (for model selection).
    - test_loader: 10,000 samples (isolated official test set).
    """
    os.makedirs(data_dir, exist_ok=True)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    full_val_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_eval,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_eval,
    )

    total_train = len(full_train_dataset)  # 50,000
    indices = list(range(total_train))
    val_indices = indices[total_train - val_split:]
    train_indices = indices[:total_train - val_split]

    train_subset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_val_dataset, val_indices)

    persistent = num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, max_batches=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0 or (max_batches > 0 and (batch_idx + 1) == max_batches):
            elapsed = time.time() - t0
            n_tot = min(max_batches, len(loader)) if max_batches > 0 else len(loader)
            print(
                f"    [Epoch {epoch+1:2d}/{total_epochs:2d}] Batch {batch_idx+1:3d}/{n_tot:3d} | "
                f"Loss: {running_loss/total:.4f} | Acc: {100.*correct/total:5.2f}% | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True,
            )

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def run_training_experiment(
    exp_name: str,
    mode: str,
    N: int,
    sort_by_exp: bool,
    epochs: int,
    lr: float,
    seed: int,
    init_checkpoint: str,
    device,
    train_loader,
    val_loader,
    test_loader,
    checkpoint_dir: str,
    max_batches: int = 0,
    fp8_format: str = "E4M3",
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 75, flush=True)
    print(f"EXPERIMENT: {exp_name}", flush=True)
    print(f"Mode: {mode} | FP8: {fp8_format} | N: {N} | Sorted: {sort_by_exp} | Epochs: {epochs} | LR: {lr} | Seed: {seed}", flush=True)
    print("=" * 75, flush=True)

    model = ResNet20(num_classes=10, mode=mode, N=N, sort_by_exp=sort_by_exp).to(device)

    # Initialize from checkpoint if provided (e.g. for fine-tuning)
    if init_checkpoint:
        if not os.path.exists(init_checkpoint):
            raise FileNotFoundError(
                f"[CRITICAL ERROR] Required initialization checkpoint '{init_checkpoint}' does not exist! "
                f"Aborting experiment '{exp_name}' to prevent silent fallback to scratch training."
            )
        raw_state = torch.load(init_checkpoint, map_location=device)
        if isinstance(raw_state, dict):
            if "state_dict" in raw_state:
                sd = raw_state["state_dict"]
            elif "model_state" in raw_state:
                sd = raw_state["model_state"]
            else:
                sd = raw_state
        else:
            sd = raw_state

        # Clean prefixes and map layer naming differences (e.g., fc -> linear)
        cleaned_sd = {}
        for k, v in sd.items():
            k_clean = k.replace("module.", "")
            if k_clean.startswith("fc."):
                k_clean = k_clean.replace("fc.", "linear.")
            cleaned_sd[k_clean] = v

        model_keys = set(model.state_dict().keys())
        loaded_keys = set(cleaned_sd.keys())
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys

        # Filter out benign batchnorm tracked counters
        critical_missing = [k for k in missing_keys if "num_batches_tracked" not in k]

        # Verify shapes for all matching keys
        mismatched_shapes = []
        for k in model_keys.intersection(loaded_keys):
            m_shape = model.state_dict()[k].shape
            l_shape = cleaned_sd[k].shape
            if m_shape != l_shape:
                mismatched_shapes.append((k, m_shape, l_shape))

        if mismatched_shapes:
            raise RuntimeError(
                f"[CRITICAL CHECKPOINT ERROR] Shape mismatches encountered when loading '{init_checkpoint}':\n"
                + "\n".join([f"  Key '{k}': Model shape {m_s} != Checkpoint shape {l_s}" for k, m_s, l_s in mismatched_shapes])
            )

        incompat = model.load_state_dict(cleaned_sd, strict=False)
        print(f"  Loaded initial weights from: {init_checkpoint}", flush=True)
        print(f"  Checkpoint Compatibility Audit: Matched {len(model_keys.intersection(loaded_keys))}/{len(model_keys)} keys", flush=True)
        if critical_missing:
            print(f"    WARNING: Critical missing keys ({len(critical_missing)}): {critical_missing[:5]}", flush=True)
        if unexpected_keys:
            print(f"    NOTE: Unexpected keys ignored ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_val_loss = 1e9
    t_start = time.time()

    for ep in range(epochs):
        t_ep0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, ep, epochs, max_batches
        )
        # Evaluate on VALIDATION set (No test-set leakage!)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device, max_batches)
        scheduler.step()
        ep_sec = time.time() - t_ep0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {ep+1:2d}/{epochs:2d} ({ep_sec:.1f}s) | "
            f"Train: Loss {tr_loss:.4f}, Acc {tr_acc:5.2f}% | "
            f"Val: Loss {va_loss:.4f}, Acc {va_acc:5.2f}% | "
            f"LR: {curr_lr:.6f}",
            flush=True,
        )

        # Select best model strictly by VALIDATION accuracy
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_val_loss = va_loss
            ckpt_path = os.path.join(checkpoint_dir, f"{exp_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    * New best validation accuracy: {best_val_acc:5.2f}% -> saved {ckpt_path}", flush=True)

    total_time = time.time() - t_start

    # Final unbiased evaluation on TEST set (evaluated ONCE using best validation model)
    best_ckpt = os.path.join(checkpoint_dir, f"{exp_name}_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, max_batches)

    print(f"\nCOMPLETED {exp_name}:", flush=True)
    print(f"  Best Val Accuracy  : {best_val_acc:5.2f}% (Loss: {best_val_loss:.4f})", flush=True)
    print(f"  Unbiased Test Acc  : {test_acc:5.2f}% (Loss: {test_loss:.4f})", flush=True)
    print(f"  Total Wall Time    : {total_time/60:.1f} min\n", flush=True)

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_name": exp_name,
        "mode": mode,
        "fp8_format": fp8_format if mode != "fp32" else "N/A",
        "N": N if mode == "bfp_mlb" else "N/A",
        "sort_by_exp": sort_by_exp if mode == "bfp_mlb" else "N/A",
        "epochs": epochs,
        "seed": seed,
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "unbiased_test_accuracy": round(test_acc, 2),
        "unbiased_test_loss": round(test_loss, 4),
        "total_time_sec": round(total_time, 1),
    }


def aggregate_and_print_summary(summary_csv_path: str, agg_csv_path: str):
    """
    Reads all historical runs from summary_csv_path, groups by unique experimental
    configuration (mode, fp8_format, N, sort_by_exp, epochs), computes cumulative
    statistics across seeds, and updates agg_csv_path without losing any prior runs.
    """
    if not os.path.exists(summary_csv_path):
        return

    all_records = []
    with open(summary_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_records.append(row)

    if not all_records:
        return

    # Deduplicate: if the exact same seed was re-run for a configuration, keep the latest run
    latest_per_seed = {}
    for r in all_records:
        k = (
            r.get("mode", ""),
            r.get("fp8_format", "E4M3"),
            str(r.get("N", "N/A")),
            str(r.get("sort_by_exp", "N/A")),
            int(r.get("epochs", 0)),
            int(r.get("seed", 0)),
        )
        latest_per_seed[k] = r

    # Group by configuration: (mode, fp8_format, N, sort_by_exp, epochs)
    groups = {}
    for (mode, fp8_fmt, N, sort_by_exp, epochs, seed), r in latest_per_seed.items():
        grp_key = (mode, fp8_fmt, N, sort_by_exp, epochs)
        if grp_key not in groups:
            groups[grp_key] = []
        groups[grp_key].append(r)

    print("\n" + "=" * 105, flush=True)
    print("CUMULATIVE STATISTICAL SUMMARY ACROSS SEEDS (ALL RECORDED RUNS)")
    print("=" * 105, flush=True)
    print(
        f"{'Mode':<15} {'FP8':<6} {'N':<6} {'Sorted':<8} {'Epochs':<8} "
        f"{'Seeds':<14} {'Val Acc (%)':<20} {'Unbiased Test Acc (%)':<20}"
    )
    print("-" * 105, flush=True)

    agg_rows = []
    for (mode, fp8_fmt, N, sorted_str, epochs), runs in sorted(
        groups.items(), key=lambda x: (x[0][0], str(x[0][2]), x[0][4])
    ):
        val_accs = [float(r["best_val_accuracy"]) for r in runs]
        test_accs = [float(r["unbiased_test_accuracy"]) for r in runs]
        seed_list = sorted([int(r["seed"]) for r in runs])
        seed_str = ",".join(str(s) for s in seed_list)

        m_val, s_val = np.mean(val_accs), np.std(val_accs)
        m_test, s_test = np.mean(test_accs), np.std(test_accs)

        val_display = f"{m_val:5.2f} +/- {s_val:4.2f}"
        test_display = f"{m_test:5.2f} +/- {s_test:4.2f}"

        print(
            f"{mode:<15} {fp8_fmt:<6} {N:<6} {sorted_str:<8} {epochs:<8} "
            f"{len(runs):<2} ({seed_str})   {val_display:<20} {test_display:<20}"
        )

        agg_rows.append({
            "mode": mode,
            "fp8_format": fp8_fmt,
            "N": N,
            "sort_by_exp": sorted_str,
            "epochs": epochs,
            "num_seeds": len(runs),
            "seeds": seed_str,
            "mean_val_acc": round(float(m_val), 2),
            "std_val_acc": round(float(s_val), 2),
            "mean_test_acc": round(float(m_test), 2),
            "std_test_acc": round(float(s_test), 2),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    print("=" * 105, flush=True)

    agg_fieldnames = [
        "mode", "fp8_format", "N", "sort_by_exp", "epochs", "num_seeds", "seeds",
        "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc", "last_updated"
    ]
    with open(agg_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)
    print(f"Cumulative aggregated statistics written to: {agg_csv_path}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="ResNet-20 QAT Training & Evaluation Engine")
    parser.add_argument(
        "--mode", type=str, default="full_suite",
        choices=[
            "fp32_50", "fp8_30", "fp8_50",
            "bfp_scratch_30", "bfp_scratch_50",
            "bfp_finetune_20", "ablation_sorting",
            "full_suite"
        ],
        help="Experiment mode to execute"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--fp8-format", type=str, default="E4M3", choices=["E4M3", "E3M4"], help="FP8 format")
    parser.add_argument("--N", type=str, default="9,25,32,64,128,256,512", help="Comma-separated N values")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated random seeds")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--val-split", type=int, default=5000, help="Validation set split size (default: 5000)")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--output-csv", type=str, default="./qat_summary_results.csv", help="Summary CSV path")
    parser.add_argument("--agg-csv", type=str, default="./qat_aggregated_metrics.csv", help="Aggregated CSV path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches per epoch (0 = full)")
    args = parser.parse_args()

    set_global_fp8_format(args.fp8_format)
    device = get_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    n_list = [int(x.strip()) for x in args.N.split(",") if x.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    max_fin = 448.0 if args.fp8_format == "E4M3" else 30.0
    print("=" * 80, flush=True)
    print("ResNet-20 CIFAR-10 Rigorous Fair QAT Training Engine", flush=True)
    print(f"Device        : {device}", flush=True)
    print(f"FP8 Format    : {args.fp8_format} (Max finite: {max_fin}, Exponent carry enabled)", flush=True)
    print(f"Mode          : {args.mode}", flush=True)
    print(f"Seeds         : {seeds}", flush=True)
    print(f"N values      : {n_list}", flush=True)
    print(f"Batch Size    : {args.batch_size}", flush=True)
    print(f"Train/Val/Test: 45,000 / {args.val_split} / 10,000 (No Test-Set Leakage!)", flush=True)
    print(f"Max Batches   : {args.max_batches} (0 = full)", flush=True)
    print("=" * 80, flush=True)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        val_split=args.val_split, num_workers=args.num_workers
    )

    all_results = []

    for seed in seeds:
        # Mode 1: FP32 Baseline (50 epochs)
        if args.mode in ("fp32_50", "full_suite"):
            ep = args.epochs if args.epochs is not None else 50
            res = run_training_experiment(
                exp_name=f"fp32_{ep}_seed{seed}",
                mode="fp32", N=64, sort_by_exp=False, epochs=ep, lr=0.1, seed=seed,
                init_checkpoint="", device=device,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                fp8_format="N/A",
            )
            all_results.append(res)

        # Mode 2: Native FP8 (30 epochs - Paper Reproduction Baseline & Anchor)
        if args.mode in ("fp8_30", "full_suite"):
            ep = args.epochs if args.epochs is not None else 30
            res = run_training_experiment(
                exp_name=f"fp8_{ep}_seed{seed}",
                mode="fp8", N=64, sort_by_exp=False, epochs=ep, lr=0.1, seed=seed,
                init_checkpoint="", device=device,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                fp8_format=args.fp8_format,
            )
            all_results.append(res)

        # Mode 3: Native FP8 (50 epochs - Matched Budget Control)
        if args.mode in ("fp8_50", "full_suite"):
            ep = args.epochs if args.epochs is not None else 50
            res = run_training_experiment(
                exp_name=f"fp8_{ep}_seed{seed}",
                mode="fp8", N=64, sort_by_exp=False, epochs=ep, lr=0.1, seed=seed,
                init_checkpoint="", device=device,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                fp8_format=args.fp8_format,
            )
            all_results.append(res)

        # Mode 4a: BFP MLB Trained from Scratch (30 epochs - Paper Reproduction Control across N)
        if args.mode in ("bfp_scratch_30", "full_suite"):
            ep = args.epochs if args.epochs is not None else 30
            for N in n_list:
                res = run_training_experiment(
                    exp_name=f"bfp_scratch{ep}_N{N}_seed{seed}",
                    mode="bfp_mlb", N=N, sort_by_exp=True, epochs=ep, lr=0.1, seed=seed,
                    init_checkpoint="", device=device,
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                    fp8_format=args.fp8_format,
                )
                all_results.append(res)

        # Mode 4b: BFP MLB Trained from Scratch (50 epochs - Matched Budget Control across N)
        if args.mode in ("bfp_scratch_50", "full_suite"):
            ep = args.epochs if args.epochs is not None else 50
            for N in n_list:
                res = run_training_experiment(
                    exp_name=f"bfp_scratch{ep}_N{N}_seed{seed}",
                    mode="bfp_mlb", N=N, sort_by_exp=True, epochs=ep, lr=0.1, seed=seed,
                    init_checkpoint="", device=device,
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                    fp8_format=args.fp8_format,
                )
                all_results.append(res)

        # Mode 5: BFP MLB Fine-Tuned from 30-Epoch FP8 Checkpoint (20 epochs across N)
        if args.mode in ("bfp_finetune_20", "full_suite"):
            ep = args.epochs if args.epochs is not None else 20
            fp8_ckpt = os.path.join(args.checkpoint_dir, f"fp8_30_seed{seed}_best.pth")
            if not os.path.exists(fp8_ckpt):
                raise FileNotFoundError(
                    f"[CRITICAL ERROR] Required baseline checkpoint '{fp8_ckpt}' not found for fine-tuning seed {seed}!\n"
                    f"To generate this baseline checkpoint first, run:\n"
                    f"  python train_experiments.py --mode fp8_30 --seeds {seed} --fp8-format {args.fp8_format}\n"
                    f"Or run the complete sequential suite:\n"
                    f"  python train_experiments.py --mode full_suite --seeds {seed} --fp8-format {args.fp8_format}"
                )
            for N in n_list:
                res = run_training_experiment(
                    exp_name=f"bfp_finetune{ep}_N{N}_seed{seed}",
                    mode="bfp_mlb", N=N, sort_by_exp=True, epochs=ep, lr=0.01, seed=seed,
                    init_checkpoint=fp8_ckpt, device=device,
                    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                    fp8_format=args.fp8_format,
                )
                all_results.append(res)

        # Mode 6: Exponent Sorting Ablation at N=64 (Matched 50-epoch scratch budget)
        if args.mode in ("ablation_sorting", "full_suite"):
            ep = args.epochs if args.epochs is not None else 50
            # Unsorted BFP scratch
            res_u = run_training_experiment(
                exp_name=f"ablation_unsorted{ep}_N64_seed{seed}",
                mode="bfp_mlb", N=64, sort_by_exp=False, epochs=ep, lr=0.1, seed=seed,
                init_checkpoint="", device=device,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                fp8_format=args.fp8_format,
            )
            all_results.append(res_u)

            # Sorted BFP scratch
            res_s = run_training_experiment(
                exp_name=f"ablation_sorted{ep}_N64_seed{seed}",
                mode="bfp_mlb", N=64, sort_by_exp=True, epochs=ep, lr=0.1, seed=seed,
                init_checkpoint="", device=device,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                checkpoint_dir=args.checkpoint_dir, max_batches=args.max_batches,
                fp8_format=args.fp8_format,
            )
            all_results.append(res_s)

    # Save individual runs summary CSV (Append-only, never overwrites prior history)
    file_exists = os.path.exists(args.output_csv)
    fieldnames = [
        "timestamp", "exp_name", "mode", "fp8_format", "N", "sort_by_exp", "epochs",
        "seed", "best_val_accuracy", "best_val_loss",
        "unbiased_test_accuracy", "unbiased_test_loss", "total_time_sec"
    ]
    with open(args.output_csv, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    # Aggregate and print cumulative statistical summary across all historical seeds
    aggregate_and_print_summary(summary_csv_path=args.output_csv, agg_csv_path=args.agg_csv)

    print("=" * 80, flush=True)
    print("ALL REQUESTED EXPERIMENTS COMPLETED WITH ZERO TEST-SET LEAKAGE!")
    print(f"Cumulative Log   : {args.output_csv}")
    print(f"Aggregated Stats : {args.agg_csv}")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
