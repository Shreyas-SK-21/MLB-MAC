# ResNet-20 CIFAR-10 Fair QAT Training & Evaluation Package

This self-contained experimental package accompanies the paper **"Multi-Level Binary (MLB) Block Floating Point (BFP) MAC Architecture for FP8 Deep Learning"**.

It resolves all methodological, fairness, and architectural concerns raised by **VLSID Reviewer 2** and **Reviewer 3**.

---

## 1. Reviewer Critiques & How This Package Resolves Them

### Reviewer 3 Critique: Training Duration Disparity Confound
* **Problem**: In the initial draft, the Native FP8 baseline was trained for 30 epochs, whereas the BFP models were initialized from that 30-epoch checkpoint and fine-tuned for an additional 20 epochs (total 50 effective epochs). Reviewers pointed out that part of BFP's accuracy advantage could simply stem from having 20 additional training epochs.
* **Resolution**: This package implements **matched-budget training controls**:
  1. **Control 1 (`fp8_50`)**: Native FP8 trained for **50 epochs from scratch** under the identical SGD/Cosine schedule.
  2. **Control 2 (`fp8_30`)**: Native FP8 trained for **30 epochs from scratch** (reproducing original paper baseline & fine-tuning anchor).
  3. **Control 3 (`fp32_50`)**: Full-precision FP32 trained for **50 epochs from scratch**.
  4. **Experiment A1 (`bfp_scratch_30`)**: BFP MLB models ($N \in \{9, 25, 32, 64, 128, 256, 512\}$) trained for **30 epochs directly from scratch** (reproducing paper 30-epoch scratch control).
  5. **Experiment A2 (`bfp_scratch_50`)**: BFP MLB models trained for **50 epochs directly from scratch** (matched 50-epoch budget control, eliminating any pre-training dependency).
  6. **Experiment B (`bfp_finetune_20`)**: BFP MLB models fine-tuned for 20 epochs from the 30-epoch FP8 checkpoint (original paper protocol).
  7. **Statistical Controls**: Multi-seed runs (`--seeds 42,43,44`) logging cumulative mean $\pm$ standard deviation to eliminate stochastic run-to-run noise.

### Reviewer 2 Critique: Exponent Sorting, Element Pairing, and Hardware Overhead
* **Problem**: Reviewer 2 questioned whether sorting exponents requires expensive runtime sorting networks/comparators and whether sorting scrambled activation-weight pairings $y = \sum_k x_k w_k$.
* **Resolution**:
  1. **Element Pairing Preservation**: Exponent sorting computes a permutation $\pi = \text{argsort}(\bar{e}_w)$ along reduction dimension $K$ and applies the **exact same permutation $\pi$ to BOTH weights and activations**:
     $$\sum_{k=0}^{K-1} x_k w_k \equiv \sum_{j=0}^{K-1} x_{\pi(j)} w_{\pi(j)}$$
     Because addition is associative and commutative, element pairings $(x_k, w_k)$ are strictly preserved ($0.00$ discrepancy without BFP).
  2. **BFP Block-Membership Optimization (Empirical Precision Preservation)**: Under BFP, the reduction dimension $K$ is partitioned into blocks of size $N$. Changing the permutation changes which elements share each block of size $N$, altering each block's shared maximum exponent $E_{\max}$ and mantissa truncation shifts $\Delta e_k = E_{\max} - e_k$. Sorting groups weights with similar magnitudes together, drastically reducing within-block dynamic range and preventing large weights from shifting smaller weights to zero truncation. Thus, sorting is an empirical precision-preserving technique rather than an invariant identity for BFP.
  3. **Zero Runtime Hardware Overhead**: Weights are **offline static constants** during inference. The permutation index $\pi$ is computed **once at compile-time (offline)**! Hardware needs **0 runtime comparators, 0 sorting networks, and 0 extra clock cycles**. The static compiler simply maps DMA/address generator fetch offsets.
  4. **Direct Empirical Ablation (`ablation_sorting`)**: Runs head-to-head experiments at $N=64$ comparing `sort_by_exp=True` vs `sort_by_exp=False` under matched training budgets.

### Methodological Rigor: Zero Test-Set Leakage & Statistical Controls
* **Dataset Splitting**: The 50,000 CIFAR-10 training images are deterministically split into **45,000 training images** and **5,000 validation images**. The 10,000 CIFAR-10 test set is held out.
* **Checkpoint Selection**: Epoch-by-epoch model checkpoints are selected **strictly based on validation accuracy**. The 10,000 test set is evaluated **strictly once at the conclusion of training**, guaranteeing 100% unbiased test accuracy with zero data leakage.
* **Multi-Seed Aggregation**: Running with multiple seeds (e.g., `--seeds 42,43,44`) automatically aggregates performance across runs into `qat_aggregated_metrics.csv` reporting mean $\pm$ standard deviation.

---

## 2. Package Structure

```
ResNet20_QAT_Package/
├── quantize.py             # FP8 E4M3 arithmetic, BFP alignment, and MLB dot product
├── layers.py               # PyTorch STE autograd layers (MLBConv2d, MLBLinear, FP8OnlyConv2d)
├── resnet20.py             # ResNet-20 for CIFAR-10 supporting FP32, FP8, and BFP modes
├── train_experiments.py    # Complete CLI runner for all experiments, controls, and sweeps
├── run_kaggle_colab.ipynb  # Turnkey Jupyter Notebook for 1-click execution on GPU
├── requirements.txt        # Python dependencies
└── SETUP_AND_RUN_GUIDE.md  # Complete setup and execution manual (this document)
```

---

## 3. Quickstart: Kaggle (Recommended - Free GPU T4)

1. **Upload Notebook**:
   - Go to [kaggle.com](https://www.kaggle.com) and create a New Notebook.
   - Click **File** $\rightarrow$ **Upload Notebook** and select `run_kaggle_colab.ipynb`.
2. **Enable GPU Accelerator**:
   - In the right-hand panel: **Settings** $\rightarrow$ **Accelerator** $\rightarrow$ Select **GPU T4 x2**.
   - Set **Internet** to **On** (required to download CIFAR-10 on the first run).
3. **Upload the Code Files**:
   - Upload `ResNet20_QAT_Package.zip` as a dataset or upload the python files directly.
   - Or simply run the first cell in `run_kaggle_colab.ipynb` which extracts the files.
4. **Run**:
   - Run the cells sequentially or click **Run All**.
   - Output logs and `qat_summary_results.csv` will be generated in `/kaggle/working/`.

---

## 4. Quickstart: Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com).
2. Upload `run_kaggle_colab.ipynb`.
3. In the menu: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU** (or A100).
4. Upload `ResNet20_QAT_Package.zip` to Colab's file browser and run:
   ```bash
   !unzip -q ResNet20_QAT_Package.zip
   !pip install -r requirements.txt
   ```
5. Execute the cells to run the desired training modes.

---

## 5. Quickstart: Local Linux / Windows GPU System

### Prerequisites
- Python 3.9+
- NVIDIA GPU with CUDA 11.8+ or 12.0+ (also works on CPU for testing with `--max-batches 5`).

### Installation
```bash
# Unzip package
unzip ResNet20_QAT_Package.zip -d ResNet20_QAT_Package
cd ResNet20_QAT_Package

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Execution Command Reference

### A. Quick Smoke Test (Verifies complete pipeline in ~30 seconds)
```bash
python train_experiments.py --mode full_suite --max-batches 3 --N 64
```

### B. Matched-Budget Baseline Controls (Reviewer 3)
```bash
# 1. FP32 Baseline (50 epochs)
python train_experiments.py --mode fp32_50 --epochs 50

# 2. Native FP8 Baseline (50 epochs - Matched Budget Control)
python train_experiments.py --mode fp8_50 --epochs 50

# 3. Native FP8 Baseline (30 epochs - Original Paper Protocol & Fine-Tuning Anchor)
python train_experiments.py --mode fp8_30 --epochs 30
```

### C. Proposed BFP MLB Trained from Scratch across N
```bash
# 1. 30-epoch Scratch Training (Paper Reproduction Control across N)
python train_experiments.py --mode bfp_scratch_30 --epochs 30 --N 9,25,32,64,128,256,512

# 2. 50-epoch Scratch Training (Matched Budget Control - No Fine-Tuning Confound)
python train_experiments.py --mode bfp_scratch_50 --epochs 50 --N 9,25,32,64,128,256,512
```

### D. Proposed BFP MLB Fine-Tuned (20 Epochs from 30-epoch FP8 Checkpoint)
```bash
# Note: Reuses checkpoints/fp8_30_seed{seed}_best.pth generated by mode fp8_30 without overwriting it.
python train_experiments.py --mode bfp_finetune_20 --epochs 20 --N 9,25,32,64,128,256,512
```

### E. Exponent Sorting Ablation (Reviewer 2)
```bash
# Head-to-head comparison of Sorted vs Unsorted at N=64 under 50 epochs scratch
python train_experiments.py --mode ablation_sorting --epochs 50 --N 64
```

### F. FP8 Format Selection (E4M3 vs E3M4)
By default, the simulation uses **E4M3** (1 sign, 4 exponent, 3 mantissa + 1 implicit leading bit = 4-bit integer mantissa, decomposing into the **$M=4$ binary bit-planes** synthesized in the Verilog RTL). If desired, you can also run **E3M4**:
```bash
# Default: Canonical E4M3 matching paper and M=4 RTL
python train_experiments.py --mode bfp_scratch_50 --fp8-format E4M3

# Optional: E3M4 format (Max finite: 30.0, bias=3)
python train_experiments.py --mode bfp_scratch_50 --fp8-format E3M4
```

### G. Zero FP32 Fallback Guarantee
All forward-pass MAC operations inside `MLBConv2d` and `MLBLinear` are strictly hardware-accurate:
1. Inputs are snapped to discrete FP8 states (254 legal values).
2. Exponents are discrete integers in $[0 \dots 15]$.
3. BFP alignment performs exact integer truncation right-shifts into 4-bit unsigned mantissas in $[0 \dots 15]$.
4. The dot product evaluates exact integer matrix multiplication, matching the RTL bitwise AND gates and popcount adder tree bit-for-bit (0 discrepancy).
5. Zero continuous floating-point multiplications occur in the MAC array.

### H. Multi-Seed Statistical Sweep
```bash
# Runs full evaluation sequence across multiple random seeds
python train_experiments.py --mode full_suite --seeds 42,43,44 --N 64
```

---

## 7. Results Format (Cumulative & Non-Overwriting)

### A. Individual Run Logs (`qat_summary_results.csv`)
The runner strictly appends every single run into `qat_summary_results.csv` with a timestamp:

| Column | Description |
| :--- | :--- |
| `timestamp` | Execution date and time (`YYYY-MM-DD HH:MM:SS`) |
| `exp_name` | Unique experiment identifier (e.g. `bfp_scratch50_N64_seed42`) |
| `mode` | `fp32`, `fp8`, or `bfp_mlb` |
| `fp8_format` | `E4M3`, `E3M4`, or `N/A` |
| `N` | Hardware block size ($9 \dots 512$, or `N/A`) |
| `sort_by_exp` | `True` (offline co-sorted), `False` (unsorted), or `N/A` |
| `epochs` | Training duration (20, 30, or 50) |
| `seed` | RNG seed (e.g. 42, 43, 44) |
| `best_val_accuracy` | Best validation accuracy (%) used strictly for checkpoint selection |
| `best_val_loss` | Validation loss corresponding to best validation accuracy |
| `unbiased_test_accuracy` | Final Top-1 test accuracy (%) on the 10,000 held-out test set |
| `unbiased_test_loss` | Test loss evaluated strictly once on best validation checkpoint |
| `total_time_sec` | Wall-clock execution time in seconds |

### B. Statistical Summary Table (`qat_aggregated_metrics.csv`)
Every run automatically reads all historical entries from `qat_summary_results.csv`, groups by `(mode, fp8_format, N, sort_by_exp, epochs)`, and compiles cumulative statistics across all seeds tested:

| Column | Description |
| :--- | :--- |
| `mode` | Model configuration mode |
| `fp8_format` | `E4M3`, `E3M4`, or `N/A` |
| `N` | Hardware block size |
| `sort_by_exp` | Sorting state |
| `epochs` | Training epochs |
| `num_seeds` | Count of distinct seeds evaluated |
| `seeds` | Comma-separated list of evaluated seeds (e.g. `42,43,44`) |
| `mean_val_acc` | Mean validation accuracy across seeds |
| `std_val_acc` | Sample standard deviation of validation accuracy |
| `mean_test_acc` | Mean unbiased test accuracy across seeds |
| `std_test_acc` | Sample standard deviation of unbiased test accuracy |
| `last_updated` | Timestamp of latest aggregation |

> [!NOTE]
> Even when executing experiments in separate invocations or on different days, previous runs are never overwritten or dropped from `qat_aggregated_metrics.csv`. The summary dashboard always reflects the complete cumulative state of all experiments executed.
