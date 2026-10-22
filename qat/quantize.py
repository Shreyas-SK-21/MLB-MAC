"""
quantize.py
===========
Hardware-accurate numerical simulation of FP8 (E4M3 / E3M4), Block Floating Point (BFP)
exponent alignment, and Multi-Level Binary (MLB) AND-popcount dot products.

Key Technical Highlights:
-------------------------
1. FP8 Format Specifications (IEEE P3109 / OCP FP8 Compliant):
   - E4M3 (Canonical Paper & RTL Hardware Specification, M=4):
     * 1 Sign bit, 4 Exponent bits (bias = 7), 3 Explicit Mantissa bits.
     * Max finite representable value: 448.0 (E=15, M=6).
     * Pattern E=15, M=7 (0b1111111) is strictly reserved for NaN (not a finite number).
     * Exponent carry-over on rounding: e.g., 1.99 rounds up to 2.0 (E=8, M=0) rather than
       clamping to 1.875.
     * Restored normal mantissa with implicit bit: 1 + 3 = 4 bits (integers 8..15),
       directly forming M = 4 binary bit-planes in hardware.
   - E3M4 (Supported Alternative, M=5):
     * 1 Sign bit, 3 Exponent bits (bias = 3), 4 Explicit Mantissa bits.
     * Max finite representable value: 30.0 (E=7, M=14, bias=3).
     * Pattern E=7, M=15 (0b1111111) is strictly reserved for NaN (not a finite number).

2. Two Verified Levels of MLB Emulation:
   - `batched_mlb_dot_product`: Fast, vectorized integer-matrix functional emulator
     for high-throughput QAT training and validation.
   - `explicit_bitplane_mlb_dot_product`: Golden reference model that explicitly decomposes
     4-bit aligned mantissas into 4 binary planes, performs 16 pairwise bitwise ANDs,
     and computes popcounts and hardwired shifts matching the Verilog logic gates bit-for-bit.

3. Exponent Sorting Clarification (BFP Block-Membership Approximation):
   - Element Pairing Invariance: The exact same permutation pi is applied to both activations
     and weights (x_pi(j) * w_pi(j)), so activation-weight pairings are never scrambled.
   - Block Alignment Impact: BFP divides the reduction dimension K into blocks of size N.
     Sorting rearranges which elements share a block, altering intra-block maximum exponents (E_max)
     and right-shift truncation amounts. Exponent sorting is thus an empirical precision-preserving
     strategy that clusters similar exponents to minimize truncation loss, not an invariant identity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FP8FormatConfig:
    def __init__(self, format_name="E4M3", flush_subnormals=True):
        self.flush_subnormals = flush_subnormals
        self.set_format(format_name)

    def set_format(self, format_name="E4M3"):
        fmt = format_name.upper()
        if fmt == "E4M3":
            self.format_name = "E4M3"
            self.exp_bits = 4
            self.mant_bits = 3
            self.bias = 7
            self.max_exp = 15
            self.num_planes = 4  # 1 implicit + 3 explicit = 4 bits (M=4)
            # In OCP E4M3, E=15, M=7 is NaN. Max finite value is E=15, M=6:
            # 2^(15 - 7) * (1 + 6/8) = 256 * 1.75 = 448.0
            self.max_val = 448.0
            self.max_mant_at_max_exp = 6
        elif fmt == "E3M4":
            self.format_name = "E3M4"
            self.exp_bits = 3
            self.mant_bits = 4
            self.bias = 3
            self.max_exp = 7
            self.num_planes = 5  # 1 implicit + 4 explicit = 5 bits (M=5)
            # In E3M4 (1 sign, 3 exp, 4 mantissa), pattern E=7, M=15 (0b1111111) is reserved for NaN.
            # Max finite representable value is E=7, M=14:
            # 2^(7 - 3) * (1 + 14/16) = 16 * 1.875 = 30.0
            self.max_val = 30.0
            self.max_mant_at_max_exp = 14
        else:
            raise ValueError(f"Unsupported FP8 format: {format_name}. Choose 'E4M3' or 'E3M4'.")

        self.min_normal = 2.0 ** (1 - self.bias)
        self.shared_exp_offset = 2 * self.bias + 2 * self.mant_bits


# Global active FP8 configuration (Default: E4M3 matching paper and synthesized RTL)
GLOBAL_FP8_CONFIG = FP8FormatConfig("E4M3", flush_subnormals=True)


def set_global_fp8_format(fmt: str, flush_subnormals: bool = True):
    """Dynamically set the global FP8 simulation format ('E4M3' or 'E3M4') and FTZ behavior."""
    GLOBAL_FP8_CONFIG.flush_subnormals = flush_subnormals
    GLOBAL_FP8_CONFIG.set_format(fmt)


def _fp8_extract_fields(abs_val: torch.Tensor, cfg: FP8FormatConfig = None):
    """
    Extract biased integer exponent, integer mantissa, and normal mask from magnitude tensor.
    Correctly implements:
      - Max finite value saturation (448.0 for E4M3, 30.0 for E3M4, preventing NaN encodings).
      - Mantissa carry-over into exponent upon rounding (e.g., 1.99 -> 2.0).
      - Strict RTL Flush-To-Zero (FTZ) contract matching bpf_aligner.v (flush_subnormals).
    """
    if cfg is None:
        cfg = GLOBAL_FP8_CONFIG

    # 1. Clamp magnitude to max finite representable value
    abs_clamped = abs_val.clamp(min=0.0, max=cfg.max_val)

    # 2. Floor log2 for initial tentative exponent
    log2_val = torch.log2(abs_clamped.clamp(min=1e-45))
    e_floor = torch.floor(log2_val).long()
    E_tentative = (e_floor + cfg.bias).clamp(min=0, max=cfg.max_exp)

    # 3. Normal vs subnormal condition
    is_normal = (E_tentative >= 1) & (abs_clamped > 0)
    scale_normal = 2.0 ** (E_tentative.float() - cfg.bias)
    mant_frac_normal = (abs_clamped / scale_normal.clamp(min=1e-45) - 1.0).clamp(min=0.0)

    # Subnormals (E == 0): value = (m / 2^mant_bits) * 2^(1 - bias)
    scale_subnormal = 2.0 ** (1 - cfg.bias)
    mant_frac_subnormal = abs_clamped / scale_subnormal

    steps = float(2 ** cfg.mant_bits)
    m_float = torch.where(is_normal, mant_frac_normal * steps, mant_frac_subnormal * steps)
    m_round = torch.round(m_float).long()

    # 4. Handle rounding carry-over: if m_round == steps, carry into exponent
    carry_mask = (m_round >= int(steps))
    E_final = torch.where(carry_mask, E_tentative + 1, E_tentative)
    m_final = torch.where(carry_mask, torch.zeros_like(m_round), m_round)

    # 5. Saturate at max representable finite value (strictly prevent NaN generation)
    if cfg.format_name == "E4M3":
        # E=15, M=7 is NaN; max finite is E=15, M=6 (448.0)
        overflow_mask = (E_final > 15) | ((E_final == 15) & (m_final > cfg.max_mant_at_max_exp))
        E_final = torch.where(overflow_mask, torch.full_like(E_final, 15), E_final)
        m_final = torch.where(overflow_mask, torch.full_like(m_final, cfg.max_mant_at_max_exp), m_final)
    elif cfg.format_name == "E3M4":
        # E=7, M=15 is NaN; max finite is E=7, M=14 (30.0)
        overflow_mask = (E_final > 7) | ((E_final == 7) & (m_final > cfg.max_mant_at_max_exp))
        E_final = torch.where(overflow_mask, torch.full_like(E_final, 7), E_final)
        m_final = torch.where(overflow_mask, torch.full_like(m_final, cfg.max_mant_at_max_exp), m_final)
    else:
        overflow_mask = (E_final > cfg.max_exp) | ((E_final == cfg.max_exp) & (m_final > cfg.max_mant_at_max_exp))
        E_final = torch.where(overflow_mask, torch.full_like(E_final, cfg.max_exp), E_final)
        m_final = torch.where(overflow_mask, torch.full_like(m_final, cfg.max_mant_at_max_exp), m_final)

    # 6. Flush subnormals if enabled (matches RTL bpf_aligner.v where mants_w = (exps_w != 0) ? {1'b1, mant} : 0)
    if cfg.flush_subnormals:
        subnorm_flush_mask = (E_final == 0)
        E_final = torch.where(subnorm_flush_mask, torch.zeros_like(E_final), E_final)
        m_final = torch.where(subnorm_flush_mask, torch.zeros_like(m_final), m_final)

    # 7. Zero mask for true zeros or subnormal underflow
    zero_mask = (abs_clamped == 0) | ((E_final == 0) & (m_final == 0))
    E_final = torch.where(zero_mask, torch.zeros_like(E_final), E_final)
    m_final = torch.where(zero_mask, torch.zeros_like(m_final), m_final)
    is_normal_final = (E_final >= 1) & (~zero_mask)

    return E_final, m_final, is_normal_final


class FP8QuantizeSTE(torch.autograd.Function):
    """
    FP8 Quantizer with Straight-Through Estimator (STE) for gradient backpropagation.
    Snaps continuous floats to discrete FP8 representable values.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        cfg = GLOBAL_FP8_CONFIG
        sign = x.sign()
        abs_x = x.abs()
        E, M, is_norm = _fp8_extract_fields(abs_x, cfg)

        # Reconstruct discrete values
        steps = float(2 ** cfg.mant_bits)
        norm_val = (1.0 + M.float() / steps) * (2.0 ** (E.float() - cfg.bias))
        subnorm_val = (M.float() / steps) * (2.0 ** (1 - cfg.bias))
        val = torch.where(is_norm, norm_val, subnorm_val)
        val = torch.where((E == 0) & (M == 0), torch.zeros_like(val), val)
        return sign * val

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-Through Estimator (STE) passes upstream gradient unaltered
        return grad_output


def fp8_quantize(x: torch.Tensor) -> torch.Tensor:
    """Snaps input tensor to discrete FP8 states using STE."""
    return FP8QuantizeSTE.apply(x)


def batched_bfp_align(vals_abs: torch.Tensor, cfg: FP8FormatConfig = None):
    """
    Block Floating Point (BFP) alignment across block dimension N.
    Matches hardware bfp_aligner.v barrel-shifter logic with exact integer truncation.
    """
    if cfg is None:
        cfg = GLOBAL_FP8_CONFIG

    exp, mant, is_normal = _fp8_extract_fields(vals_abs, cfg)
    max_exp = exp.max(dim=1).values
    implicit_weight = 2 ** cfg.mant_bits
    full_mant = torch.where(is_normal, implicit_weight + mant, torch.zeros_like(mant))
    shift_amt = (max_exp.unsqueeze(1) - exp).clamp(min=0, max=15)
    aligned_mant = torch.div(
        full_mant, (2 ** shift_amt.clamp(max=15)), rounding_mode="trunc"
    )
    return aligned_mant, max_exp


def batched_mlb_dot_product(
    act_fp8: torch.Tensor,
    wgt_fp8: torch.Tensor,
    N: int,
    sort_by_exp: bool = True,
    cfg: FP8FormatConfig = None
) -> torch.Tensor:
    """
    Fast functional emulator of BFP-aligned Multi-Level Binary (MLB) dot product:
      Y = X @ W^T

    Computes dot products on discrete 4-bit integer mantissas using vectorized matrix multiplication.
    Mathematically proven to produce the identical result to explicit bit-plane popcount.
    """
    if cfg is None:
        cfg = GLOBAL_FP8_CONFIG

    batch, K = act_fp8.shape
    C_out = wgt_fp8.shape[0]
    assert wgt_fp8.shape[1] == K, f"Shape mismatch: act K={K}, wgt K={wgt_fp8.shape[1]}"

    # Exponent-sorted permutation (applied to both inputs and weights)
    if sort_by_exp:
        wgt_exp, _, _ = _fp8_extract_fields(wgt_fp8.abs(), cfg)
        mean_wgt_exp = wgt_exp.float().mean(dim=0)
        sort_idx = torch.argsort(mean_wgt_exp)
        act_fp8 = act_fp8[:, sort_idx]
        wgt_fp8 = wgt_fp8[:, sort_idx]

    num_chunks = math.ceil(K / N)
    K_padded = num_chunks * N

    if K_padded > K:
        act_padded = F.pad(act_fp8, (0, K_padded - K))
        wgt_padded = F.pad(wgt_fp8, (0, K_padded - K))
    else:
        act_padded = act_fp8
        wgt_padded = wgt_fp8

    act_reshaped = act_padded.view(batch, num_chunks, N)
    wgt_reshaped = wgt_padded.view(C_out, num_chunks, N)

    sign_act = act_reshaped < 0
    sign_wgt = wgt_reshaped < 0

    aligned_act, max_exp_x = batched_bfp_align(act_reshaped.abs().view(batch * num_chunks, N), cfg)
    aligned_wgt, max_exp_w = batched_bfp_align(wgt_reshaped.abs().view(C_out * num_chunks, N), cfg)

    aligned_act = aligned_act.view(batch, num_chunks, N)
    max_exp_x = max_exp_x.view(batch, num_chunks)
    aligned_wgt = aligned_wgt.view(C_out, num_chunks, N)
    max_exp_w = max_exp_w.view(C_out, num_chunks)

    sign_act_f = torch.where(sign_act, -1.0, 1.0)
    sign_wgt_f = torch.where(sign_wgt, -1.0, 1.0)

    signed_act = aligned_act.float() * sign_act_f
    signed_wgt = aligned_wgt.float() * sign_wgt_f

    act_perm = signed_act.permute(1, 0, 2)
    wgt_perm = signed_wgt.permute(1, 0, 2)

    # Core MLB Dot Product: Matrix multiplication of BFP-aligned integer mantissas
    raw_dot = torch.bmm(act_perm, wgt_perm.transpose(1, 2))

    # Shared Exponent Rescaling: 2^(max_exp_x + max_exp_w - offset)
    scale_x = (2.0 ** max_exp_x.float()).t().unsqueeze(2)
    scale_w = (2.0 ** max_exp_w.float()).t().unsqueeze(1)

    chunk_results = raw_dot * scale_x * scale_w * (2.0 ** -cfg.shared_exp_offset)
    return chunk_results.sum(dim=0)


def explicit_bitplane_mlb_dot_product(
    act_fp8: torch.Tensor,
    wgt_fp8: torch.Tensor,
    N: int,
    sort_by_exp: bool = True,
    cfg: FP8FormatConfig = None
) -> torch.Tensor:
    """
    Golden Gate-Level Reference Model:
    Explicitly decomposes aligned mantissas into 4 binary bit-planes (M=4),
    evaluates all 16 pairwise bitwise AND operations, performs popcount reductions,
    and accumulates shifted bit-plane sums.
    Matches the Verilog RTL gates bit-for-bit.
    """
    if cfg is None:
        cfg = GLOBAL_FP8_CONFIG

    batch, K = act_fp8.shape
    C_out = wgt_fp8.shape[0]

    if sort_by_exp:
        wgt_exp, _, _ = _fp8_extract_fields(wgt_fp8.abs(), cfg)
        mean_wgt_exp = wgt_exp.float().mean(dim=0)
        sort_idx = torch.argsort(mean_wgt_exp)
        act_fp8 = act_fp8[:, sort_idx]
        wgt_fp8 = wgt_fp8[:, sort_idx]

    num_chunks = math.ceil(K / N)
    K_padded = num_chunks * N

    if K_padded > K:
        act_padded = F.pad(act_fp8, (0, K_padded - K))
        wgt_padded = F.pad(wgt_fp8, (0, K_padded - K))
    else:
        act_padded = act_fp8
        wgt_padded = wgt_fp8

    act_reshaped = act_padded.view(batch, num_chunks, N)
    wgt_reshaped = wgt_padded.view(C_out, num_chunks, N)

    aligned_act, max_exp_x = batched_bfp_align(act_reshaped.abs().view(batch * num_chunks, N), cfg)
    aligned_wgt, max_exp_w = batched_bfp_align(wgt_reshaped.abs().view(C_out * num_chunks, N), cfg)

    aligned_act = aligned_act.view(batch, num_chunks, N)
    max_exp_x = max_exp_x.view(batch, num_chunks)
    aligned_wgt = aligned_wgt.view(C_out, num_chunks, N)
    max_exp_w = max_exp_w.view(C_out, num_chunks)

    sign_act = act_reshaped < 0
    sign_wgt = wgt_reshaped < 0

    # Output accumulator
    final_output = torch.zeros(batch, C_out, dtype=torch.float32, device=act_fp8.device)
    M_planes = cfg.num_planes

    for b in range(batch):
        for o in range(C_out):
            block_sum = 0.0
            for c in range(num_chunks):
                a_block = aligned_act[b, c].long() # N integers in [0..15]
                w_block = aligned_wgt[o, c].long() # N integers in [0..15]
                s_a = sign_act[b, c]
                s_w = sign_wgt[o, c]
                sign_prod = s_a ^ s_w

                # Extract explicit binary planes (axi[0..3], awi[0..3])
                a_planes = [(a_block >> p) & 1 for p in range(M_planes)]
                w_planes = [(w_block >> q) & 1 for q in range(M_planes)]

                pos_total = 0
                neg_total = 0

                # 16 pairwise AND-popcounts
                for p in range(M_planes):
                    for q in range(M_planes):
                        and_bits = a_planes[p] & w_planes[q]
                        # Zero-mask partitioning (as implemented in Verilog pos_mask / neg_mask)
                        pos_popcount = (and_bits & (~sign_prod).long()).sum().item()
                        neg_popcount = (and_bits & (sign_prod).long()).sum().item()

                        shift = p + q
                        pos_total += pos_popcount << shift
                        neg_total += neg_popcount << shift

                net_integer_sum = pos_total - neg_total
                scale = (2.0 ** (max_exp_x[b, c].item() + max_exp_w[o, c].item() - cfg.shared_exp_offset))
                block_sum += net_integer_sum * scale

            final_output[b, o] = block_sum

    return final_output


def batched_native_fp8_dot_product(act_fp8: torch.Tensor, wgt_fp8: torch.Tensor) -> torch.Tensor:
    """
    Native FP8 dot product baseline (standard float matmul on discrete FP8 values).
    """
    return act_fp8 @ wgt_fp8.t()
