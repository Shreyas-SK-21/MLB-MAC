#!/usr/bin/env python3
"""
gen_fp8_mac.py
==============
Generates parametric FP8 (E4M3-style, matching the original fp8_native_mac_64.v
bit-level algorithm) native MAC Verilog modules for arbitrary N, and
self-checking testbenches (11 fixed edge cases + 5 random cases per N).

Usage:
    python3 gen_fp8_mac.py

Outputs (per N in N_LIST):
    FP8_NATIVE_MAC_<N>.v        - the DUT
    tb_FP8_NATIVE_MAC_<N>.v     - self-checking testbench

Only the bit-widths / loop bounds change between instances; the
micro-architecture (3-stage: per-lane multiply+align -> pairwise adder
reduction tree -> registered output) is identical to the reference N=64
design. For non-power-of-two N, the reduction tree pairs elements level by
level and simply carries forward (sign-extended) any element that has no
partner at that level -- this is the natural generalisation of the N=64
tree and does not change the arithmetic result.
"""

import os
import random
import math

N_LIST = [9, 25, 32, 64, 128, 256, 512]
OUT_DIR = "/Users/grover.heer/Documents/IIITB/Projects/MLB(MultiLevelBinary)/FP8/actual implementation"

# ----------------------------------------------------------------------
# Golden (bit-exact) software model of the RTL arithmetic
# ----------------------------------------------------------------------
def decode_fp8(byte):
    """Replicates the exact extraction/adjustment logic used in the RTL."""
    sign = (byte >> 7) & 1
    exp  = (byte >> 3) & 0xF
    man  = byte & 0x7
    if exp == 0:
        true_man = man          # {1'b0, man}
        true_exp = 1
    else:
        true_man = 0x8 | man    # {1'b1, man}
        true_exp = exp
    return sign, true_man, true_exp


def lane_shifted_value(byte_a, byte_w):
    """Bit-exact replica of one fp8_mult_lane -> shifted_prod value."""
    sign_a, man_a, exp_a = decode_fp8(byte_a)
    sign_w, man_w, exp_w = decode_fp8(byte_w)

    sign_out  = sign_a ^ sign_w
    prod_mant = man_a * man_w            # 4b x 4b -> up to 8b (max 0xF*0xF=225)
    exp_sum   = exp_a + exp_w            # range 2..30
    shift_amt = exp_sum - 2              # range 0..28

    aligned_mag = prod_mant << shift_amt  # fits in 36 bits (max ~6.0e10)

    return -aligned_mag if sign_out else aligned_mag


def golden_final_acc(act_bytes, wgt_bytes):
    """act_bytes / wgt_bytes: lists of N ints (0..255), lane 0 first."""
    assert len(act_bytes) == len(wgt_bytes)
    return sum(lane_shifted_value(a, w) for a, w in zip(act_bytes, wgt_bytes))


# ----------------------------------------------------------------------
# Verilog generation helpers
# ----------------------------------------------------------------------
def acc_width(n):
    """Bits needed for the final signed accumulator: 37 (single lane) +
    ceil(log2(n)) levels of pairwise growth."""
    growth = max(0, math.ceil(math.log2(n))) if n > 1 else 0
    return 37 + growth


def build_reduction_tree(n, base_width):
    """
    Returns (verilog_text, final_wire_name, final_width).
    Builds a generic pairwise binary reduction tree over `n` signed
    elements named shifted_prod[0..n-1] (width base_width, i.e. [base_width-1:0]).
    Any unpaired (odd) element at a level is carried forward unchanged
    (sign-extended by Verilog's automatic extension on assignment to a
    wider signed wire).
    """
    lines = []
    level = 0
    # level 0 "virtual" wires are shifted_prod[i], base_width bits
    cur_names = [f"shifted_prod[{i}]" for i in range(n)]
    cur_width = base_width

    if n == 1:
        # trivial: no reduction needed
        return "", cur_names[0], cur_width

    while len(cur_names) > 1:
        level += 1
        nxt_width = cur_width + 1
        nxt_names = []
        m = len(cur_names)
        pairs = m // 2
        has_leftover = (m % 2 == 1)

        lines.append(f"    // ---- Reduction level {level}: {m} -> {pairs + (1 if has_leftover else 0)} "
                      f"(width {cur_width} -> {nxt_width}) ----")

        src_is_base = (level == 1)
        src_array = "shifted_prod" if src_is_base else f"sum_L{level-1}"

        lines.append(f"    wire signed [{nxt_width-1}:0] sum_L{level} [0:{pairs + (1 if has_leftover else 0) - 1}];")
        lines.append(f"    genvar j_L{level};")
        lines.append(f"    generate")
        lines.append(f"        for (j_L{level}=0; j_L{level}<{pairs}; j_L{level}=j_L{level}+1) begin : lvl{level}_pair")
        lines.append(f"            assign sum_L{level}[j_L{level}] = $signed({src_array}[2*j_L{level}]) + $signed({src_array}[2*j_L{level}+1]);")
        lines.append(f"        end")
        lines.append(f"    endgenerate")

        for p in range(pairs):
            nxt_names.append(f"sum_L{level}[{p}]")

        if has_leftover:
            lines.append(f"    assign sum_L{level}[{pairs}] = $signed({src_array}[{m-1}]);")
            nxt_names.append(f"sum_L{level}[{pairs}]")

        cur_names = nxt_names
        cur_width = nxt_width

    return "\n".join(lines), cur_names[0], cur_width


def gen_dut(n):
    base_w = 37
    tree_text, final_name, final_w = build_reduction_tree(n, base_w)
    assert final_w == acc_width(n), f"width mismatch {final_w} vs {acc_width(n)}"

    bus_w = n * 8

    module = f"""`timescale 1ns/1ps

module FP8_NATIVE_MAC_{n} (
    input clk,
    input rst,
    input valid_in,
    input [{bus_w-1}:0] fp8_act,      // {n} parallel FP8 E4M3 activations
    input [{bus_w-1}:0] fp8_wgt,      // {n} parallel FP8 E4M3 weights
    output reg done,
    output reg signed [{final_w-1}:0] final_acc // {final_w}-bit exact sum (implicit scale 2^-18)
);

    reg signed [{base_w-1}:0] shifted_prod [0:{n-1}];
    reg stage1_valid;

    genvar i;
    generate
        for (i = 0; i < {n}; i = i + 1) begin : fp8_mult_lane
            // Wires for extraction
            wire sign_a = fp8_act[i*8 + 7];
            wire [3:0] exp_a = fp8_act[i*8 + 3 +: 4];
            wire [2:0] man_a = fp8_act[i*8 +: 3];

            wire sign_w = fp8_wgt[i*8 + 7];
            wire [3:0] exp_w = fp8_wgt[i*8 + 3 +: 4];
            wire [2:0] man_w = fp8_wgt[i*8 +: 3];
            wire [3:0] true_man_a = (exp_a == 4'd0) ? {{1'b0, man_a}} : {{1'b1, man_a}};
            wire [3:0] true_man_w = (exp_w == 4'd0) ? {{1'b0, man_w}} : {{1'b1, man_w}};
            wire [3:0] true_exp_a = (exp_a == 4'd0) ? 4'd1 : exp_a;
            wire [3:0] true_exp_w = (exp_w == 4'd0) ? 4'd1 : exp_w;

            wire sign_out = sign_a ^ sign_w;
            wire [7:0] prod_mant = true_man_a * true_man_w; // 4b x 4b = 8b
            wire [4:0] exp_sum = true_exp_a + true_exp_w;   // Range: 2 to 30

            wire [4:0] shift_amt = exp_sum - 5'd2;
            wire [35:0] aligned_mag = {{28'd0, prod_mant}} << shift_amt;
            
            always @(posedge clk) begin
                if (rst) begin
                    shifted_prod[i] <= {base_w}'sd0;
                end else if (valid_in) begin
                    if (sign_out)
                        shifted_prod[i] <= -$signed({{1'b0, aligned_mag}});
                    else
                        shifted_prod[i] <=  $signed({{1'b0, aligned_mag}});
                end
            end
        end
    endgenerate

    // Stage 1 Control
    always @(posedge clk) begin
        if (rst) stage1_valid <= 1'b0;
        else     stage1_valid <= valid_in;
    end


{tree_text}

    always @(posedge clk) begin
        if (rst) begin
            final_acc <= {final_w}'sd0;
            done      <= 1'b0;
        end else begin
            done <= stage1_valid;
            if (stage1_valid) begin
                final_acc <= {final_name};
            end
        end
    end

endmodule
"""
    return module, final_w


# ----------------------------------------------------------------------
# Test-vector generation
# ----------------------------------------------------------------------
def edge_case_bytes(n, seed_idx):
    """
    Returns a list of n FP8 bytes (0..255) representing one of the 11
    canonical edge-case vectors. Each edge case fills the whole vector
    with a themed pattern so lane-parallel behaviour and the reduction
    tree corner cases (odd leftovers, max magnitude, etc.) are exercised.
    """
    POS_ZERO      = 0b0_0000_000
    NEG_ZERO      = 0b1_0000_000
    MAX_NORMAL    = 0b0_1111_110  # E4M3 max normal (exp=15,man=6 -> avoids NaN pattern if reserved)
    MAX_NORMAL_N  = 0b1_1111_110
    MIN_NORMAL    = 0b0_0001_000  # exp=1, man=0
    MAX_SUBNORMAL = 0b0_0000_111  # exp=0, man=7
    MIN_SUBNORMAL = 0b0_0000_001  # exp=0, man=1
    ONE           = 0b0_0111_000  # exp bias-ish value used consistently by this custom format (see note)
    NEG_ONE       = 0b1_0111_000
    ALT_SIGN_HI   = 0b0_1111_111
    ALT_SIGN_HI_N = 0b1_1111_111

    def fill(byte_a, byte_w):
        return [byte_a] * n, [byte_w] * n

    def alternate(byte_a0, byte_a1, byte_w0, byte_w1):
        a = [byte_a0 if (i % 2 == 0) else byte_a1 for i in range(n)]
        w = [byte_w0 if (i % 2 == 0) else byte_w1 for i in range(n)]
        return a, w

    cases = []
    # 1. all zeros
    cases.append(("all_zero", *fill(POS_ZERO, POS_ZERO)))
    # 2. act=0, wgt=max  -> result must be 0
    cases.append(("act_zero_wgt_max", *fill(POS_ZERO, MAX_NORMAL)))
    # 3. act=max, wgt=0
    cases.append(("act_max_wgt_zero", *fill(MAX_NORMAL, POS_ZERO)))
    # 4. max * max (positive) -> largest positive accumulation
    cases.append(("max_pos_times_max_pos", *fill(MAX_NORMAL, MAX_NORMAL)))
    # 5. max * max (negative*positive) -> largest negative accumulation
    cases.append(("max_neg_times_max_pos", *fill(MAX_NORMAL_N, MAX_NORMAL)))
    # 6. max_neg * max_neg -> large positive (neg*neg)
    cases.append(("max_neg_times_max_neg", *fill(MAX_NORMAL_N, MAX_NORMAL_N)))
    # 7. min normal * min normal
    cases.append(("min_normal_sq", *fill(MIN_NORMAL, MIN_NORMAL)))
    # 8. max subnormal * max subnormal
    cases.append(("max_subnormal_sq", *fill(MAX_SUBNORMAL, MAX_SUBNORMAL)))
    # 9. min subnormal * min subnormal (smallest nonzero result)
    cases.append(("min_subnormal_sq", *fill(MIN_SUBNORMAL, MIN_SUBNORMAL)))
    # 10. alternating sign lanes -> exercises reduction tree cancellation
    cases.append(("alternating_sign", *alternate(MAX_NORMAL, MAX_NORMAL_N, MAX_NORMAL, MAX_NORMAL)))
    # 11. alternating max/zero -> exercises odd-leftover carry paths
    cases.append(("alternating_max_zero", *alternate(MAX_NORMAL, POS_ZERO, MAX_NORMAL, POS_ZERO)))

    assert len(cases) == 11
    return cases


def random_case_bytes(n, rng):
    a = [rng.randint(0, 255) for _ in range(n)]
    w = [rng.randint(0, 255) for _ in range(n)]
    return a, w


def bytes_to_hex_concat(byte_list):
    """lane 0 is the LOW byte of the concatenated bus, matching fp8_act[i*8 +:8]."""
    val = 0
    for i, b in enumerate(byte_list):
        val |= (b & 0xFF) << (8 * i)
    return val


def gen_testbench(n):
    final_w = acc_width(n)
    bus_w = n * 8
    rng = random.Random(1000 + n)  # deterministic per-N

    vectors = []  # (name, act_int, wgt_int, expected_int)

    for name, a, w in edge_case_bytes(n, None):
        act_int = bytes_to_hex_concat(a)
        wgt_int = bytes_to_hex_concat(w)
        exp = golden_final_acc(a, w)
        vectors.append((name, act_int, wgt_int, exp))

    for k in range(5):
        a, w = random_case_bytes(n, rng)
        act_int = bytes_to_hex_concat(a)
        wgt_int = bytes_to_hex_concat(w)
        exp = golden_final_acc(a, w)
        vectors.append((f"random_{k}", act_int, wgt_int, exp))

    assert len(vectors) == 16

    def hexstr(v, width_bits):
        nyb = (width_bits + 3) // 4
        return f"{v & ((1<<width_bits)-1):0{nyb}x}"

    tv_lines = []
    for idx, (name, act_int, wgt_int, exp) in enumerate(vectors):
        tv_lines.append(
            f'        test_name[{idx}] = "{name}";\n'
            f"        act_vec[{idx}]  = {bus_w}'h{hexstr(act_int, bus_w)};\n"
            f"        wgt_vec[{idx}]  = {bus_w}'h{hexstr(wgt_int, bus_w)};\n"
            f"        exp_vec[{idx}]  = {final_w}'sh{hexstr(exp if exp>=0 else (exp + (1<<final_w)), final_w)};"
        )
    tv_block = "\n".join(tv_lines)

    tb = f"""`timescale 1ns/1ps

module tb_FP8_NATIVE_MAC_{n};

    localparam N       = {n};
    localparam BUS_W   = {bus_w};
    localparam ACC_W   = {final_w};
    localparam NUM_VEC = 16; // 11 edge cases + 5 random

    reg clk, rst, valid_in;
    reg  [BUS_W-1:0] fp8_act, fp8_wgt;
    wire done;
    wire signed [ACC_W-1:0] final_acc;

    integer i;
    integer errors;

    reg [BUS_W-1:0] act_vec [0:NUM_VEC-1];
    reg [BUS_W-1:0] wgt_vec [0:NUM_VEC-1];
    reg signed [ACC_W-1:0] exp_vec [0:NUM_VEC-1];
    reg [255:0] test_name [0:NUM_VEC-1]; // enough room for readable ascii names

    FP8_NATIVE_MAC_{n} dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .fp8_act(fp8_act),
        .fp8_wgt(fp8_wgt),
        .done(done),
        .final_acc(final_acc)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; valid_in = 0;
        fp8_act = 0; fp8_wgt = 0;
        errors = 0;

{tv_block}

        @(negedge clk); @(negedge clk);
        rst = 0;

        for (i = 0; i < NUM_VEC; i = i + 1) begin
            @(negedge clk);
            fp8_act  = act_vec[i];
            fp8_wgt  = wgt_vec[i];
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            @(negedge clk);
            wait (done == 1'b1 || 1); // stage 3 registers same cycle 'done' asserts
            #0.1;

            if (final_acc !== exp_vec[i]) begin
                errors = errors + 1;
                $display("[FAIL] vec %0d (%0s): expected=%0d got=%0d",
                          i, test_name[i], exp_vec[i], final_acc);
            end else begin
                $display("[PASS] vec %0d (%0s): got=%0d", i, test_name[i], final_acc);
            end
        end

        if (errors == 0)
            $display("\\n==== ALL %0d TESTS PASSED for FP8_NATIVE_MAC_{n} ====", NUM_VEC);
        else
            $display("\\n==== %0d / %0d TESTS FAILED for FP8_NATIVE_MAC_{n} ====", errors, NUM_VEC);

        $finish;
    end

endmodule
"""
    return tb


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in N_LIST:
        dut_text, final_w = gen_dut(n)
        tb_text = gen_testbench(n)

        dut_path = os.path.join(OUT_DIR, f"FP8_NATIVE_MAC_{n}.v")
        tb_path  = os.path.join(OUT_DIR, f"tb_FP8_NATIVE_MAC_{n}.v")

        with open(dut_path, "w") as f:
            f.write(dut_text)
        with open(tb_path, "w") as f:
            f.write(tb_text)

        print(f"N={n:4d}  bus={n*8:4d} bits  acc_width={final_w:3d} bits  -> {dut_path}, {tb_path}")


if __name__ == "__main__":
    main()
