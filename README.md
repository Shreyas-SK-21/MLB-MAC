# AND-Popcount MLB-MAC: The Story of Building a Better FP8 Dot-Product Unit

[![Hardware](https://img.shields.io/badge/RTL-Verilog-blue.svg)](#hardware-architecture)
[![PDK](https://img.shields.io/badge/PDK-SkyWater%20130nm-green.svg)](#asic-synthesis--reproduction)
[![Framework](https://img.shields.io/badge/QAT-PyTorch-orange.svg)](#quantization-aware-training-qat)

Welcome to the **AND-Popcount Multi-Level Binary (MLB) MAC** repository. 

This repository contains synthesizable Verilog RTL, Open PDK (SkyWater 130 nm) synthesis automation, and a PyTorch Quantization-Aware Training (QAT) framework for an energy-efficient 8-bit floating-point (FP8 E4M3) dot-product datapath designed for deep neural network inference.

---

## The Engineering Story: How We Got Here

Building efficient arithmetic for deep learning hardware is rarely a straight line. This project grew out of an attempt to solve a specific problem in FP8 inference, hit unexpected roadblocks with existing literature, and eventually converged on a cleaner, mathematically exact architecture.

Here is the story of how the architecture evolved, what failed, and why this design works.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     THE EVOLUTION OF OUR ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. Conventional FP8 MAC                                                     │
│    [Exp Compare] ──► [Per-Element Barrel Shifters] ──► [Multipliers] ──► FP32 Accum │
│    Problem: Expensive barrel shifters and high dynamic switching per lane.  │
│                                                                             │
│ 2. Block Floating Point (BFP)                                               │
│    [Block Exponent e_max] ──► [Narrow Shifts] ──► [Integer Multipliers]     │
│    Problem: Exponents are aligned, but 4x4 integer multipliers remain.      │
│                                                                             │
│ 3. The Pitfall: Published MLB (TCAS-I 2025)                                 │
│    Decompose operands into binary planes ──► XNOR-Popcount (2P - N)         │
│    Problem: Built for bipolar {-1, +1}; requires messy C1, C2, C3 offset    │
│    correction trees on unsigned FP8 significands that eat away savings.     │
│                                                                             │
│ 4. The Breakthrough: AND-Popcount MLB-MAC                                   │
│    Unsigned 4-bit significands ──► Direct bitwise AND ──► Equal-Weight Popcount │
│    + Dual-Array Complementary Sign Masking (II = 1 sustained throughput)    │
│    Result: ZERO correction terms, 39.65% smaller, 51.11% lower power.       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Act I: The Problem with Standard FP8 Dot Products
Modern deep learning accelerators are shifting from FP16 and INT8 to **FP8 (E4M3)** because it preserves dynamic range while halving memory bandwidth. But inside the hardware datapath, conventional FP8 execution is still surprisingly heavy:
- Every lane requires dedicated exponent comparison logic.
- Variable-length barrel shifters must align significands across every input pair.
- Wide 32-bit floating-point accumulators with normalization trees dominate area and power.

To tackle this, researchers often turn to **Block Floating Point (BFP)**. BFP groups a block of $N$ inputs under a single shared maximum exponent ($e_{\max}$). This replaces per-element variable shifters with simple, block-wide alignment logic. 

**However, BFP only solves the exponent problem.** Once operands are aligned, you still need $N$ dedicated integer multipliers to process the significands.

---

### Act II: The Pitfall of Published Multi-Level Binary (MLB)
To eliminate multipliers altogether, we turned to the published **Multi-Level Binary (MLB)** formulation (*IEEE TCAS-I 2025*). 

The core thesis of MLB is appealing: decompose numbers into bit planes and convert multiplication into bitwise logical operations followed by population counts (popcounts). In the published formulation, bit planes were mapped to bipolar values $\{-1, +1\}$ using **XNOR** gates and scaled popcounts:

$$\text{XNOR-Popcount}(A, B) = 2P - N$$

**Where it went wrong:**  
While bipolar mapping works for symmetric binary neural networks, FP8 E4M3 significands (with the hidden bit restored) are strictly **unsigned integer values in the range $[8, 15]$**. 

Forcing an unsigned representation into a bipolar XNOR framework required awkward algebraic corrections:
- Multiple correction terms ($C_1, C_2, C_3$) had to be calculated across planes.
- Additional adder trees and constant offsets were needed to undo the bipolar bias.
- When synthesized, the overhead of the correction circuitry largely erased the power and area advantages of eliminating multipliers.

---

### Act III: The Epiphany — Direct AND-Popcount Factorization
Looking closely at the arithmetic, we realized a simple truth: **we were solving a problem created by the representation itself.**

If significands are unsigned, we do not need bipolar transformation at all. The product of two unsigned binary numbers $X$ and $W$ decomposes directly into weighted conjunctions of their constituent bit planes:

$$\sum_{k=0}^{N-1} \widetilde{m}_{x,k} \widetilde{m}_{w,k} = \sum_{i=0}^{M-1} \sum_{j=0}^{M-1} 2^{i+j} \left( \mathbf{a}_i \cdot \mathbf{b}_j \right)$$

For FP8 E4M3, the significand precision is $M=4$ (3 stored mantissa bits + 1 hidden bit), creating exactly $4 \times 4 = 16$ plane pairs. 

By replacing XNOR with a simple bitwise **AND**:
1. **Zero correction terms:** Cross-products are exact by construction. No offsets, no bias adjustment.
2. **Wiring-only weighting:** Terms with equal weight ($i+j$) are grouped before accumulation, and $2^{i+j}$ scalings are implemented purely as bit-shifts in hardware wiring.
3. **Simpler reduction:** The reduction tree handles narrow unsigned totals without sign-extension sign-bit propagation until the final output.

---

### Act IV: The Sign Bottleneck and the Dual-Array Solution
In deep learning dot products, inputs and weights have signs ($s_k = s_{x,k} \oplus s_{w,k}$). How do you handle signs without turning the entire datapath into an expensive signed array?

#### The First Attempt: Two-Phase Sign FSM (Failed Throughput)
Our initial implementation used a single AND-popcount array and a state machine:
- Cycle 1: Process all positive terms ($s_k = 0$).
- Cycle 2: Process all negative terms ($s_k = 1$), and subtract.

While this minimized cell area, it halved the throughput: **Initiation Interval $\text{II} = 2$** (0.5 blocks per cycle). In tensor engines, stalling every other cycle degrades effective efficiency.

#### The Breakthrough: Dual-Array with Complementary Masking
To achieve $\text{II} = 1$ while keeping arithmetic unsigned, we split the architecture into parallel positive and negative reduction arrays:
- The product sign $s_k$ drives complementary masks: $p_k = \overline{s_k}$ and $n_k = s_k$.
- Inactive lanes are masked to zero, which suppresses dynamic switching in those gates.
- Both arrays run concurrently in unsigned arithmetic.
- A single final subtractor at the output forms the signed integer result: $S_{\text{int}} = S_+ - S_-$.

**The Trade-off:** While the dual array uses 19.18% more area than the FSM, it doubles sustained throughput to **1.0 block/cycle**, delivering a **67.8% higher throughput density per unit area**.

---

### Act V: Honest, Controlled Benchmarking
Many accelerator papers report massive gains by comparing an optimized datapath against an unoptimized or unaligned baseline. We wanted to isolate exactly where every micro-watt and square micron was coming from. 

We implemented and synthesized three controlled baselines under identical SkyWater 130 nm standard-cell library constraints:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SUMMARY OF HARDWARE ABLATION                       │
│                           (Block Size N = 64, 40 MHz)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Baseline 2: Native FP8 MAC (No BFP, FP32 accumulator)                      │
│  ├── Area:  256,955 µm²                                                     │
│  └── Power: 3.75 mW                                                         │
│         │                                                                   │
│         ▼  -39.46% Area, -43.47% Power (Gain from BFP Alignment)            │
│                                                                             │
│  Baseline 1: BFP + Integer Multipliers MAC                                  │
│  ├── Area:  155,568 µm²                                                     │
│  └── Power: 2.12 mW                                                         │
│         │                                                                   │
│         ▼  -7.61% Area, -27.36% Power (Gain from AND-Popcount Core)         │
│                                                                             │
│  Proposed: AND-Popcount MLB-MAC (Dual-Array, II = 1)                        │
│  ├── Area:  143,725 µm²                                                     │
│  └── Power: 1.54 mW  ──► Total Savings vs Native FP8: 44.07% Area, 58.93% Pwr│
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Ablation vs Published MLB (Baseline 3, XNOR-popcount with correction):     │
│  ├── Baseline 3: 238,155 µm² area, 3.15 mW power, 17,033 IPS/W             │
│  └── Proposed:   143,725 µm² area, 1.54 mW power, 34,841 IPS/W             │
│      ► 39.65% Area Reduction | 51.11% Power Reduction | 2.05x Higher IPS/W │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Baseline 1 (BFP + Integer Multipliers):** Same BFP front end, but uses inferred $4 \times 4$ multipliers. Replacing them with our AND-popcount core saves **7.61% area and 27.36% power** at $N=64$.
2. **Baseline 2 (Native FP8):** Standard individual exponent alignment and FP32 accumulator. Combined savings reach **44.07% area and 58.93% power**.
3. **Baseline 3 (Matched-II XNOR Reconstruction):** Synthesizable adaptation of published MLB with matched $\text{II}=1$. Proves that eliminating bipolar correction circuitry saves **39.65% area and 51.11% power ($2.05\times$ higher IPS/W)**.

---

### Act VI: Closing the Loop with Quantization-Aware Training (QAT)
Hardware efficiency is meaningless if network accuracy collapses. 

We developed a bit-accurate PyTorch emulator that faithfully models E4M3 representation, block-level exponent sharing, right-shift truncation, and dual-array popcount summation. We tested **ResNet-20 on CIFAR-10** across three independent random seeds (7, 42, 123) under a strict 50-epoch matched budget:

| Configuration | Test Accuracy (Mean $\pm s$, $n=3$) | $\Delta$ vs. Native FP8 |
|---|:---:|:---:|
| **FP32 Baseline** | $90.17\%$ | $+0.28\text{ pp}$ |
| **Native FP8 Baseline** | $89.89\%$ | $0.00\text{ pp}$ |
| **BFP-MLB ($N=9$)** | $89.30 \pm 0.13\%$ | $-0.59\text{ pp}$ |
| **BFP-MLB ($N=25$)** | $89.11 \pm 0.09\%$ | $-0.78\text{ pp}$ |
| **BFP-MLB ($N=32$)** | $89.23 \pm 0.34\%$ | $-0.66\text{ pp}$ |
| **BFP-MLB ($N=64$)** | $88.79 \pm 0.15\%$ | $-1.10\text{ pp}$ |
| **BFP-MLB ($N=128$)** | $88.71 \pm 0.21\%$ | $-1.18\text{ pp}$ |
| **BFP-MLB ($N=256$)** | $88.88 \pm 0.08\%$ | $-1.01\text{ pp}$ |

**Key Takeaways:**
- At $N=64$, accuracy remains within **1.10 percentage points** of native FP8.
- Across block sizes from $N=9$ to $256$, test accuracy remains within a tight **0.59 pp window** (88.71% to 89.30%), proving that exponent sharing does not induce catastrophic truncation noise.

---

## Repository Structure

```text
MLB-MAC/
├── hardware/
│   ├── proposed_and_popcount/     # Golden 3-cycle, II=1 AND-popcount datapath
│   │   ├── FP8_corrected.v        # Top-level datapath
│   │   ├── and_popcount_4_bit.v   # Unsigned bit-plane AND & reduction
│   │   ├── bpf_aligner.v          # BFP shared exponent aligner
│   │   ├── MLB_MAC_4_bit.v        # Plane-pair grouping logic
│   │   └── MLB_MAC_unit.v         # MAC unit wrapper
│   ├── baselines/                 # Reference hardware implementations
│   │   ├── baseline1_bfp_int/     # BFP + integer multiplier MAC
│   │   ├── baseline2_native_fp8/  # Conventional FP8 MAC with FP32 accumulator
│   │   └── baseline3_xnor_reconstruction/ # Matched-II XNOR-popcount ablation
│   ├── variants/                  # Architectural exploration variants
│   │   ├── single_array_fsm/      # 2-phase sign-multiplexed FSM (II=2)
│   │   └── optimized_1cycle/      # Unpipelined 1-cycle exploration
│   ├── sweep_blocks/              # Pre-configured blocks across N in {9..512}
│   └── testbenches/               # Unit and co-verification testbenches
│       ├── tb_FP8_corrected.v     # Verification testbench with streaming checks
│       └── tb_co_verify.v         # Co-simulation verification testbench
│
├── synthesis/                     # Open ASIC synthesis flow (SkyWater 130 nm)
│   ├── scripts/                   # Yosys (.ys) and OpenSTA (.sdc / .tcl) scripts
│   ├── run_full_sweep.py          # Sweep orchestration across all N
│   └── compute_vcd_power_and_update_csv.py # Cycle-accurate VCD power parser
│
├── qat/                           # PyTorch Quantization-Aware Training suite
│   ├── layers.py                  # Bit-accurate hardware emulator layer
│   ├── quantize.py                # Straight-Through Estimator (STE) quantizer
│   ├── resnet20.py                # ResNet-20 model for CIFAR-10
│   ├── train_experiments.py       # Multi-seed training engine
│   ├── run_kaggle_colab.ipynb     # Interactive cloud notebook
│   ├── requirements.txt           # Python dependencies
│   └── SETUP_AND_RUN_GUIDE.md     # QAT execution walkthrough
│
├── results/                       # Empirical measurement data
│   ├── MLB_MAC_Master_Unified_Results.csv # Full synthesis, power, and IPS/W data
│   ├── qat_summary_results.csv    # Individual QAT runs across seeds
│   └── qat_aggregated_metrics.csv # Multi-seed statistical aggregations
│
└── docs/
    └── assets/                    # Architectural diagrams (.svg and .pdf)
```

---

## Quickstart & Reproduction

### 1. Simulating the Hardware RTL
Simulate the proposed AND-popcount datapath using [Icarus Verilog](http://iverilog.icarus.com/):

```bash
# Compile and run testbench
iverilog -o sim_mlb \
  hardware/testbenches/tb_FP8_corrected.v \
  hardware/proposed_and_popcount/FP8_corrected.v \
  hardware/proposed_and_popcount/MLB_MAC_unit.v \
  hardware/proposed_and_popcount/MLB_MAC_4_bit.v \
  hardware/proposed_and_popcount/and_popcount_4_bit.v \
  hardware/proposed_and_popcount/bpf_aligner.v

vvp sim_mlb
```

The testbench verifies back-to-back streaming across multiple blocks, validating the 3-cycle latency and $\text{II}=1$ initiation interval.

---

### 2. ASIC Synthesis Flow (SkyWater 130 nm)
Synthesis uses [Yosys](https://yosyshq.net/yosys/) and [OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) with the open-source `sky130_fd_sc_hd` standard-cell library:

```bash
# Run synthesis and timing analysis across all block sizes (N = 9 to 512)
python synthesis/run_full_sweep.py

# Parse simulation VCD switching activity and compute power & IPS/W
python synthesis/compute_vcd_power_and_update_csv.py
```

---

### 3. Running Quantization-Aware Training (QAT)
Set up the Python environment:

```bash
cd qat
pip install -r requirements.txt
```

Train ResNet-20 on CIFAR-10 from scratch using the BFP-MLB emulator:

```bash
# Train N=64 block configuration under Seed 42 for 50 epochs
python train_experiments.py --mode bfp_mlb --N 64 --epochs 50 --seed 42

# Train native FP8 reference baseline
python train_experiments.py --mode fp8 --epochs 50 --seed 42

# Train FP32 floating-point control baseline
python train_experiments.py --mode fp32 --epochs 50 --seed 42
```

Alternatively, open `qat/run_kaggle_colab.ipynb` in Google Colab or Kaggle to run training on cloud GPUs.

---

## License & Attribution
This project is open-source under the MIT License. If you use or build upon this architecture in your research, please cite:

```bibtex
@inproceedings{kulkarni2026andpopcount,
  author    = {Shreyas S. Kulkarni and Heer and Madhav Rao},
  title     = {AND-Popcount MLB-MAC: An Energy-Efficient FP8 Dot-Product Unit for DNN Inference},
  booktitle = {IEEE International Symposium on Circuits and Systems (ISCAS)},
  year      = {2027}
}
```
