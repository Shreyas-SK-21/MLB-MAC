"""
generate_and_verify_optimized_sweep.py
======================================
1. Creates `MLB BFP Optimized_1Cycle` for all N in {9, 25, 32, 64, 128, 256, 512}
   without overwriting any prior runs.
2. Implements the 4 Task-1 RTL optimizations for all N:
   - Exact (2^k - 1) carry-in full-adder popcount compressor tree
   - Zero-gate hardwired bit shifts << (i+j) (no MLB_unit barrel shifters or 16-bit regs)
   - Unsigned equal-weight shift grouping (G0..G6) + single narrow subtractor
   - Complementary pos_mask / neg_mask preserving 50% zero-masking
3. Runs iverilog + vvp on all 7 block sizes to verify 100% functional accuracy,
   3-cycle pipeline latency, II=1 streaming, and generates VCD switching traces.
4. Runs Yosys 0.33 synthesis (SkyWater 130nm HD) for all 7 block sizes.
"""

import os
import shutil
import subprocess
import math

BASE_CORR = r"c:\Users\shrey\Desktop\Projects\MLB-MAC\FP8_Corrected"
LIB_WSL = "/mnt/c/Users/shrey/Desktop/Projects/MLB-MAC/FP8_Corrected/AND_Popcount/Synth files/sky130_fd_sc_hd__tt_025C_1v80.lib"
IVERILOG = r"C:\iverilog\bin\iverilog.exe"
VVP = r"C:\iverilog\bin\vvp.exe"

N_CONFIGS = {
    9:   {"pop_w": 4,  "wu": 11, "sum_w": 17},
    25:  {"pop_w": 5,  "wu": 13, "sum_w": 18},
    32:  {"pop_w": 6,  "wu": 13, "sum_w": 19},
    64:  {"pop_w": 7,  "wu": 14, "sum_w": 21},
    128: {"pop_w": 8,  "wu": 15, "sum_w": 22},
    256: {"pop_w": 9,  "wu": 16, "sum_w": 23},
    512: {"pop_w": 10, "wu": 17, "sum_w": 24},
}


def gen_popcount_module(N: int, pop_w: int) -> str:
    """Generates exact (2^k - 1) carry-in full-adder compressor tree for any N."""
    lines = []
    lines.append(f"module and_popcount_{N} (")
    lines.append(f"    output reg [{pop_w-1}:0] f_output,")
    lines.append(f"    input      [{N-1}:0] a,")
    lines.append(f"    input      [{N-1}:0] b,")
    lines.append(f"    input             clk,")
    lines.append(f"    input             rst,")
    lines.append(f"    input             valid_in")
    lines.append(f");")
    lines.append(f"    wire [{N-1}:0] x = a & b;")
    lines.append("")

    if N in (32, 64, 128, 256, 512):
        K = int(round(math.log2(N)))
        # Stages m = 1 .. K-1: p_{2^(m+1)-1}
        # m=1: p3 (N//4 groups, width 2, consumes 3*(N//4) bits: x[0 .. 3*N//4 - 1])
        num_p3 = N // 4
        lines.append(f"    wire [1:0] p3 [0:{num_p3-1}];")
        for m in range(2, K):
            w = m + 1
            cnt = N // (2 ** (m + 1))
            tag = (2 ** (m + 1)) - 1
            if cnt > 1:
                lines.append(f"    wire [{w-1}:0] p{tag} [0:{cnt-1}];")
            else:
                lines.append(f"    wire [{w-1}:0] p{tag};")
        lines.append(f"    wire [{pop_w-1}:0] final_popcount;")
        lines.append("    genvar i, k;")
        lines.append("    generate")
        lines.append(f"        for (i = 0; i < {num_p3}; i = i + 1) begin : gen_p3")
        lines.append("            assign p3[i][0] = x[3*i] ^ x[3*i+1] ^ x[3*i+2];")
        lines.append("            assign p3[i][1] = (x[3*i] & x[3*i+1]) | (x[3*i+1] & x[3*i+2]) | (x[3*i] & x[3*i+2]);")
        lines.append("        end")

        bit_idx = 3 * num_p3
        for m in range(2, K):
            w_in = m
            w_out = m + 1
            cnt = N // (2 ** (m + 1))
            prev_tag = (2 ** m) - 1
            curr_tag = (2 ** (m + 1)) - 1
            if cnt > 1:
                lines.append(f"        for (i = 0; i < {cnt}; i = i + 1) begin : gen_p{curr_tag}")
                lines.append(f"            wire [{w_in}:0] c;")
                lines.append(f"            assign c[0] = x[{bit_idx} + i];")
                lines.append(f"            for (k = 0; k < {w_in}; k = k + 1) begin : bit_p{curr_tag}")
                lines.append(f"                assign p{curr_tag}[i][k] = p{prev_tag}[2*i][k] ^ p{prev_tag}[2*i+1][k] ^ c[k];")
                lines.append(f"                assign c[k+1] = (p{prev_tag}[2*i][k] & p{prev_tag}[2*i+1][k]) | (p{prev_tag}[2*i][k] & c[k]) | (p{prev_tag}[2*i+1][k] & c[k]);")
                lines.append(f"            end")
                lines.append(f"            assign p{curr_tag}[i][{w_in}] = c[{w_in}];")
                lines.append(f"        end")
                bit_idx += cnt
            else:
                lines.append("    endgenerate")
                lines.append(f"    wire [{w_in}:0] c{curr_tag};")
                lines.append(f"    assign c{curr_tag}[0] = x[{bit_idx}];")
                lines.append("    generate")
                lines.append(f"        for (k = 0; k < {w_in}; k = k + 1) begin : bit_p{curr_tag}")
                lines.append(f"            assign p{curr_tag}[k] = p{prev_tag}[0][k] ^ p{prev_tag}[1][k] ^ c{curr_tag}[k];")
                lines.append(f"            assign c{curr_tag}[k+1] = (p{prev_tag}[0][k] & p{prev_tag}[1][k]) | (p{prev_tag}[0][k] & c{curr_tag}[k]) | (p{prev_tag}[1][k] & c{curr_tag}[k]);")
                lines.append(f"        end")
                lines.append("    endgenerate")
                lines.append(f"    assign p{curr_tag}[{w_in}] = c{curr_tag}[{w_in}];")
                bit_idx += 1

        last_tag = N - 1
        lines.append(f"    wire [{K}:0] c{N};")
        lines.append(f"    assign c{N}[0] = x[{N-1}];")
        lines.append("    generate")
        lines.append(f"        for (k = 0; k < {K}; k = k + 1) begin : bit_p{N}")
        lines.append(f"            assign final_popcount[k] = p{last_tag}[k] ^ c{N}[k];")
        lines.append(f"            assign c{N}[k+1]          = p{last_tag}[k] & c{N}[k];")
        lines.append(f"        end")
        lines.append("    endgenerate")
        lines.append(f"    assign final_popcount[{K}] = c{N}[{K}];")

    elif N == 9:
        # 9 = 7 (p3[0] + p3[1] + x[6]) + 2 (x[7] + x[8])
        lines.append("    wire [1:0] p3 [0:1];")
        lines.append("    wire [2:0] p7;")
        lines.append("    wire [1:0] p2;")
        lines.append("    wire [3:0] final_popcount;")
        lines.append("    genvar i;")
        lines.append("    generate")
        lines.append("        for (i = 0; i < 2; i = i + 1) begin : gen_p3")
        lines.append("            assign p3[i][0] = x[3*i] ^ x[3*i+1] ^ x[3*i+2];")
        lines.append("            assign p3[i][1] = (x[3*i] & x[3*i+1]) | (x[3*i+1] & x[3*i+2]) | (x[3*i] & x[3*i+2]);")
        lines.append("        end")
        lines.append("    endgenerate")
        lines.append("    wire c0 = x[6];")
        lines.append("    wire c1 = (p3[0][0] & p3[1][0]) | (p3[0][0] & c0) | (p3[1][0] & c0);")
        lines.append("    assign p7[0] = p3[0][0] ^ p3[1][0] ^ c0;")
        lines.append("    assign p7[1] = p3[0][1] ^ p3[1][1] ^ c1;")
        lines.append("    assign p7[2] = (p3[0][1] & p3[1][1]) | (p3[0][1] & c1) | (p3[1][1] & c1);")
        lines.append("    assign p2[0] = x[7] ^ x[8];")
        lines.append("    assign p2[1] = x[7] & x[8];")
        lines.append("    assign final_popcount = {1'b0, p7} + {2'b0, p2};")

    elif N == 25:
        # 25 = 15 (x[0..14]) + 7 (x[15..21]) + 3 (x[22..24])
        lines.append("    wire [1:0] p3 [0:6];")
        lines.append("    wire [2:0] p7 [0:2];")
        lines.append("    wire [3:0] p15;")
        lines.append("    wire [4:0] final_popcount;")
        lines.append("    genvar i, k;")
        lines.append("    generate")
        lines.append("        for (i = 0; i < 7; i = i + 1) begin : gen_p3")
        lines.append("            assign p3[i][0] = x[3*i] ^ x[3*i+1] ^ x[3*i+2];")
        lines.append("            assign p3[i][1] = (x[3*i] & x[3*i+1]) | (x[3*i+1] & x[3*i+2]) | (x[3*i] & x[3*i+2]);")
        lines.append("        end")
        lines.append("        for (i = 0; i < 3; i = i + 1) begin : gen_p7")
        lines.append("            wire c0 = x[21 + i];")
        lines.append("            wire c1 = (p3[2*i][0] & p3[2*i+1][0]) | (p3[2*i][0] & c0) | (p3[2*i+1][0] & c0);")
        lines.append("            assign p7[i][0] = p3[2*i][0] ^ p3[2*i+1][0] ^ c0;")
        lines.append("            assign p7[i][1] = p3[2*i][1] ^ p3[2*i+1][1] ^ c1;")
        lines.append("            assign p7[i][2] = (p3[2*i][1] & p3[2*i+1][1]) | (p3[2*i][1] & c1) | (p3[2*i+1][1] & c1);")
        lines.append("        end")
        lines.append("    endgenerate")
        lines.append("    wire [3:0] c15;")
        lines.append("    assign c15[0] = x[24];")
        lines.append("    generate")
        lines.append("        for (k = 0; k < 3; k = k + 1) begin : bit_p15")
        lines.append("            assign p15[k]   = p7[0][k] ^ p7[1][k] ^ c15[k];")
        lines.append("            assign c15[k+1] = (p7[0][k] & p7[1][k]) | (p7[0][k] & c15[k]) | (p7[1][k] & c15[k]);")
        lines.append("        end")
        lines.append("    endgenerate")
        lines.append("    assign p15[3] = c15[3];")
        lines.append("    wire [3:0] p10 = {1'b0, p7[2]} + {2'b0, p3[6]};")
        lines.append("    assign final_popcount = {1'b0, p15} + {1'b0, p10};")

    lines.append("")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (rst) begin")
    lines.append(f"            f_output <= {pop_w}'d0;")
    lines.append("        end else if (valid_in) begin")
    lines.append("            f_output <= final_popcount;")
    lines.append("        end")
    lines.append("    end")
    lines.append("endmodule")
    return "\n".join(lines)


def gen_full_fp8_optimized_verilog(N: int, cfg: dict) -> str:
    pop_w = cfg["pop_w"]
    wu = cfg["wu"]
    sum_w = cfg["sum_w"]
    diff_w = wu + 1
    ext_w = sum_w - diff_w
    act_w = 8 * N - 1
    axi_w = 4 * N - 1

    pop_mod = gen_popcount_module(N, pop_w)

    w_g0 = pop_w
    w_g1 = pop_w + 1
    w_g2 = pop_w + 2
    w_g3 = pop_w + 2
    w_g4 = pop_w + 2
    w_g5 = pop_w + 1
    w_g6 = pop_w

    pad0 = wu - w_g0
    pad1 = wu - (w_g1 + 1)
    pad2 = wu - (w_g2 + 2)
    pad3 = wu - (w_g3 + 3)
    pad4 = wu - (w_g4 + 4)
    pad5 = wu - (w_g5 + 5)
    pad6 = wu - (w_g6 + 6)

    def fmt_shifted(name, w_g, shift, wu_total):
        top_zeros = wu_total - (w_g + shift)
        parts = []
        if top_zeros > 0:
            parts.append(f"{top_zeros}'b0")
        parts.append(name)
        if shift > 0:
            parts.append(f"{shift}'b0")
        return "{" + ", ".join(parts) + "}"

    pos_terms = [
        fmt_shifted("pos_g0", w_g0, 0, wu),
        fmt_shifted("pos_g1", w_g1, 1, wu),
        fmt_shifted("pos_g2", w_g2, 2, wu),
        fmt_shifted("pos_g3", w_g3, 3, wu),
        fmt_shifted("pos_g4", w_g4, 4, wu),
        fmt_shifted("pos_g5", w_g5, 5, wu),
        fmt_shifted("pos_g6", w_g6, 6, wu),
    ]
    neg_terms = [t.replace("pos_", "neg_") for t in pos_terms]

    pos_expr = "\n                          + ".join(pos_terms)
    neg_expr = "\n                          + ".join(neg_terms)

    sign_ext_line = (
        f"    assign mlb_sum = {{{{{ext_w}{{mlb_sum_r[{diff_w-1}]}}}}, mlb_sum_r}};"
        if ext_w > 0
        else "    assign mlb_sum = mlb_sum_r;"
    )

    return f"""// ===========================================================================
// FP8_corrected.v  --  Optimized 1-Cycle AND-Popcount MLB-MAC (N={N}, M=4)
// ===========================================================================
// Architectural Optimizations Applied (Task 1):
//   1. Exact (2^k - 1) Carry-In Compressor Tree in and_popcount_{N}.
//   2. Eliminated redundant MLB_unit dynamic barrel shifters and duplicate
//      16-bit registers; replaced with zero-gate hardwired shifts << (i+j).
//   3. Unsigned Equal-Weight Shift Grouping (G0..G6) + Single {wu}-bit Subtractor.
//   4. Preserved complementary sign-masked AND-popcount cores (50% zero-masked).
// ===========================================================================

`timescale 1ns/1ps

{pop_mod}


module MLB_4_grouped (
    output signed [{sum_w-1}:0] mlb_sum,
    output reg           done,
    input  [{axi_w}:0]       axi_pos,
    input  [{axi_w}:0]       axi_neg,
    input  [{axi_w}:0]       awi,
    input                clk,
    input                rst,
    input                valid_in
);
    wire [{pop_w-1}:0] pop_pos [0:3][0:3];
    wire [{pop_w-1}:0] pop_neg [0:3][0:3];

    genvar i, j;
    generate
        for (i = 0; i < 4; i = i + 1) begin : rows
            for (j = 0; j < 4; j = j + 1) begin : cols
                and_popcount_{N} u_pos (
                    .f_output(pop_pos[i][j]),
                    .a       (axi_pos[i*{N} +: {N}]),
                    .b       (awi[j*{N} +: {N}]),
                    .clk     (clk),
                    .rst     (rst),
                    .valid_in(valid_in)
                );
                and_popcount_{N} u_neg (
                    .f_output(pop_neg[i][j]),
                    .a       (axi_neg[i*{N} +: {N}]),
                    .b       (awi[j*{N} +: {N}]),
                    .clk     (clk),
                    .rst     (rst),
                    .valid_in(valid_in)
                );
            end
        end
    endgenerate

    reg pop_valid;
    always @(posedge clk) begin
        if (rst) pop_valid <= 1'b0;
        else     pop_valid <= valid_in;
    end

    // Group unsigned positive popcounts by shift weight s = i + j in {{0..6}}
    wire [{w_g0-1}:0] pos_g0 = pop_pos[0][0];
    wire [{w_g1-1}:0] pos_g1 = pop_pos[0][1] + pop_pos[1][0];
    wire [{w_g2-1}:0] pos_g2 = (pop_pos[0][2] + pop_pos[1][1]) + pop_pos[2][0];
    wire [{w_g3-1}:0] pos_g3 = (pop_pos[0][3] + pop_pos[1][2]) + (pop_pos[2][1] + pop_pos[3][0]);
    wire [{w_g4-1}:0] pos_g4 = (pop_pos[1][3] + pop_pos[2][2]) + pop_pos[3][1];
    wire [{w_g5-1}:0] pos_g5 = pop_pos[2][3] + pop_pos[3][2];
    wire [{w_g6-1}:0] pos_g6 = pop_pos[3][3];

    // Group unsigned negative popcounts by shift weight s = i + j in {{0..6}}
    wire [{w_g0-1}:0] neg_g0 = pop_neg[0][0];
    wire [{w_g1-1}:0] neg_g1 = pop_neg[0][1] + pop_neg[1][0];
    wire [{w_g2-1}:0] neg_g2 = (pop_neg[0][2] + pop_neg[1][1]) + pop_neg[2][0];
    wire [{w_g3-1}:0] neg_g3 = (pop_neg[0][3] + pop_neg[1][2]) + (pop_neg[2][1] + pop_neg[3][0]);
    wire [{w_g4-1}:0] neg_g4 = (pop_neg[1][3] + pop_neg[2][2]) + pop_neg[3][1];
    wire [{w_g5-1}:0] neg_g5 = pop_neg[2][3] + pop_neg[3][2];
    wire [{w_g6-1}:0] neg_g6 = pop_neg[3][3];

    wire [{wu-1}:0] pos_total = {pos_expr};

    wire [{wu-1}:0] neg_total = {neg_expr};

    wire signed [{diff_w-1}:0] diff_r = $signed({{1'b0, pos_total}}) - $signed({{1'b0, neg_total}});

    reg signed [{diff_w-1}:0] mlb_sum_r;
    always @(posedge clk) begin
        if (rst) begin
            mlb_sum_r <= {diff_w}'sd0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;
            if (pop_valid) begin
                mlb_sum_r <= diff_r;
                done      <= 1'b1;
            end
        end
    end

{sign_ext_line}

endmodule


module fp8_mlb_top_corrected (
    output signed [{sum_w-1}:0] wide_integer_sum,
    output signed [8:0]  shared_exponent,
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,
    input [{act_w}:0]        fp8_activations,
    input [{act_w}:0]        fp8_weights
);

    wire [{axi_w}:0] axi_planes;
    wire [{axi_w}:0] awi_planes;
    wire [3:0]   max_exp_x;
    wire [3:0]   max_exp_w;

    bfp_aligner align_x (
        .clk           (clk),
        .fp8_vec       (fp8_activations),
        .aligned_planes(axi_planes),
        .max_exp       (max_exp_x)
    );

    bfp_aligner align_w (
        .clk           (clk),
        .fp8_vec       (fp8_weights),
        .aligned_planes(awi_planes),
        .max_exp       (max_exp_w)
    );

    genvar i;
    wire [{N-1}:0] sign_k;
    generate
        for (i = 0; i < {N}; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [{N-1}:0] sign_k_d1;
    reg        valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= {N}'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    wire [{N-1}:0] pos_mask = ~sign_k_d1;
    wire [{N-1}:0] neg_mask =  sign_k_d1;

    wire [{axi_w}:0] axi_planes_pos;
    wire [{axi_w}:0] axi_planes_neg;

    assign axi_planes_pos[{N-1}:0]         = axi_planes[{N-1}:0]         & pos_mask;
    assign axi_planes_pos[{2*N-1}:{N}]     = axi_planes[{2*N-1}:{N}]     & pos_mask;
    assign axi_planes_pos[{3*N-1}:{2*N}]   = axi_planes[{3*N-1}:{2*N}]   & pos_mask;
    assign axi_planes_pos[{4*N-1}:{3*N}]   = axi_planes[{4*N-1}:{3*N}]   & pos_mask;

    assign axi_planes_neg[{N-1}:0]         = axi_planes[{N-1}:0]         & neg_mask;
    assign axi_planes_neg[{2*N-1}:{N}]     = axi_planes[{2*N-1}:{N}]     & neg_mask;
    assign axi_planes_neg[{3*N-1}:{2*N}]   = axi_planes[{3*N-1}:{2*N}]   & neg_mask;
    assign axi_planes_neg[{4*N-1}:{3*N}]   = axi_planes[{4*N-1}:{3*N}]   & neg_mask;

    MLB_4_grouped mac_core (
        .mlb_sum (wide_integer_sum),
        .done    (mac_done),
        .axi_pos (axi_planes_pos),
        .axi_neg (axi_planes_neg),
        .awi     (awi_planes),
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    wire signed [8:0] shared_exponent_comb = $signed({{5'b0, max_exp_x}})
                                           + $signed({{5'b0, max_exp_w}})
                                           - 9'sd20;

    reg signed [8:0] exp_pipe_1, exp_pipe_2;

    always @(posedge clk) begin
        if (rst) begin
            exp_pipe_1 <= 9'sd0;
            exp_pipe_2 <= 9'sd0;
        end else begin
            exp_pipe_1 <= shared_exponent_comb;
            exp_pipe_2 <= exp_pipe_1;
        end
    end

    assign shared_exponent = exp_pipe_2;

endmodule
"""


def main():
    for N, cfg in N_CONFIGS.items():
        n_dir = os.path.join(BASE_CORR, f"M=4, N={N}(and implementation)")
        src_corr_v = os.path.join(n_dir, "MLB BFP Corrected", "Verilog codes")
        opt_dir = os.path.join(n_dir, "MLB BFP Optimized_1Cycle")
        opt_v_dir = os.path.join(opt_dir, "Verilog codes")
        opt_s_dir = os.path.join(opt_dir, "Synth files")
        os.makedirs(opt_v_dir, exist_ok=True)
        os.makedirs(opt_s_dir, exist_ok=True)

        # Copy bpf_aligner.v and tb_FP8_corrected.v
        for fname in os.listdir(src_corr_v):
            if "aligner" in fname:
                shutil.copy2(os.path.join(src_corr_v, fname), os.path.join(opt_v_dir, fname))
        tb_src = os.path.join(src_corr_v, "tb_FP8_corrected.v")
        tb_dst = os.path.join(opt_v_dir, "tb_FP8_corrected.v")
        with open(tb_src, "r", encoding="utf-8") as f:
            tb_code = f.read()
        # Update VCD filename so it creates mlb_mac_opt_sim.vcd in the Verilog codes dir
        tb_code = tb_code.replace('dumpfile("mlb_mac_sim.vcd")', 'dumpfile("mlb_mac_opt_sim.vcd")')
        tb_code = tb_code.replace('dumpfile("mlb_mac_corrected_sim.vcd")', 'dumpfile("mlb_mac_opt_sim.vcd")')
        with open(tb_dst, "w", encoding="utf-8") as f:
            f.write(tb_code)

        # Write FP8_corrected.v (skip N=64 if already present, or write identical)
        if N != 64:
            v_code = gen_full_fp8_optimized_verilog(N, cfg)
            with open(os.path.join(opt_v_dir, "FP8_corrected.v"), "w", encoding="utf-8") as f:
                f.write(v_code)

        # Write Yosys synthesis script
        aligner_file = "bpf_aligner.v" if os.path.exists(os.path.join(opt_v_dir, "bpf_aligner.v")) else "bfp_aligner.v"
        ys_code = f"""read_verilog "../Verilog codes/FP8_corrected.v"
read_verilog "../Verilog codes/{aligner_file}"

synth -top fp8_mlb_top_corrected

dfflibmap -liberty "{LIB_WSL}"
abc -liberty "{LIB_WSL}"
opt_clean

tee -o fp8_mlb_opt_area.rpt stat -liberty "{LIB_WSL}"
write_verilog fp8_mlb_opt_synth.v
"""
        with open(os.path.join(opt_s_dir, "synth_opt_130.ys"), "w", encoding="utf-8") as f:
            f.write(ys_code)

        # Run iverilog + vvp
        sim_bin = os.path.join(opt_v_dir, "sim_opt")
        cmd_iv = [
            IVERILOG, "-o", sim_bin,
            os.path.join(opt_v_dir, "FP8_corrected.v"),
            os.path.join(opt_v_dir, aligner_file),
            tb_dst
        ]
        subprocess.run(cmd_iv, check=True)
        res_vvp = subprocess.run([VVP, "sim_opt"], cwd=opt_v_dir, capture_output=True, text=True, check=True)
        passed = "ALL TESTS PASSED" in res_vvp.stdout
        print(f"[N={N:3d}] Iverilog/VVP Simulation: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(res_vvp.stdout[-500:])
            raise RuntimeError(f"Simulation failed for N={N}")

        # Run Yosys synthesis in WSL (skip N=64 if rpt already exists)
        rpt_path = os.path.join(opt_s_dir, "fp8_mlb_opt_area.rpt")
        if N != 64 or not os.path.exists(rpt_path):
            wsl_s_dir = "/mnt/c" + opt_s_dir[2:].replace("\\", "/")
            cmd_ys = ["wsl", "-d", "Ubuntu", "-u", "root", "--", "bash", "-c", f"cd '{wsl_s_dir}' && yosys -q -s synth_opt_130.ys"]
            subprocess.run(cmd_ys, check=True)
        print(f"[N={N:3d}] Yosys Synthesis Complete -> {rpt_path}")


if __name__ == "__main__":
    main()
