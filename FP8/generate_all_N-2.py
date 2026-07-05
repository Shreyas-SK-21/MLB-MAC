#!/usr/bin/env python3
"""
Generate M=4, N=K=<N>(and implementation) folders for all required N values.
N values: 9, 25, 32, 64, 128, 256, 512  (N=64 already exists, skipped)

Coding style exactly mirrors the original N=64 files:
  - and_popcount_4_bit: single genvar i; generate with multiple for-assign lines;
    final add done OUTSIDE as plain assign (stops when 2 elements remain).
  - int_mac_N reduction: single genvar j; generate with named begin:gen_rN blocks;
    final assign s_final = $signed(s_last[0]) + $signed(s_last[1]) outside.
  - All other files: only bit-widths change, structure identical to N=64.
"""

import os, math

BASE = "/Users/grover.heer/Documents/IIITB/Projects/MLB(MultiLevelBinary)/New-FP8"
N_VALUES = [9, 25, 32, 64, 128, 256, 512]

# ─── helpers ─────────────────────────────────────────────────────────────────
def pc_bits(N):
    """floor(log2(N))+1 — number of bits to count N ones."""
    return int(math.floor(math.log2(max(N, 1)))) + 1


def popcount_tree(N):
    """
    Build the reduction tree for and_popcount_4_bit.

    Rule (matches original N=64 style):
      • Run generate loops while count > 2  (so last loop always leaves ≥2 elements).
      • Track odd-element remainders at each step.
      • The final assign (outside generate) sums the last 1 or 2 wire-array
        elements plus any accumulated remainders.

    Returns:
      wire_decls  : list of "wire [w:0] sum_Lk [0:pairs-1];" strings (no indent)
      gen_lines   : list of lines for the single generate block (no indent)
      final_assign: string like "sum_L5[0] + sum_L5[1]" for assign final_popcount = ...
      PC          : number of output bits
    """
    PC = pc_bits(N)

    levels     = []      # (lev, pairs, src, in_w, out_w)
    remainders = []      # (verilog_expr_str, width)

    cur_count = N
    cur_width = 1
    lev       = 0
    src       = "xnn"

    while cur_count > 2:
        pairs = cur_count // 2
        odd   = cur_count % 2
        new_w = cur_width + 1
        lev  += 1

        levels.append((lev, pairs, src, cur_width, new_w))

        if odd:
            rem_expr = "%s[%d]" % (src, cur_count - 1)
            remainders.append((rem_expr, cur_width))

        src       = "sum_L%d" % lev
        cur_count = pairs
        cur_width = new_w

    # cur_count is now 1 or 2
    last_lev = lev
    last_src = src          # points to sum_L<lev> (or "xnn" if no levels ran)
    last_w   = cur_width    # width of each element in the last wire array

    # ── wire declarations ────────────────────────────────────────────────────
    wire_decls = []
    for (l, p, s, iw, ow) in levels:
        wire_decls.append("wire [%d:0] sum_L%d [0:%d];" % (ow - 1, l, p - 1))
    wire_decls.append("wire [%d:0] final_popcount;" % (PC - 1))

    # ── single generate block ────────────────────────────────────────────────
    gen_lines = ["genvar i;", "generate"]
    for (l, p, s, iw, ow) in levels:
        gen_lines.append(
            "    for(i=0; i<%d; i=i+1) assign sum_L%d[i] = %s[2*i] + %s[2*i+1];"
            % (p, l, s, s)
        )
    gen_lines.append("endgenerate")

    # ── final assign (outside generate) ─────────────────────────────────────
    # Collect terms; all must be zero-extended to PC bits before adding.
    terms = []

    if cur_count == 2 and lev == 0:
        # Edge case: N=2, no levels, final = xnn[0] + xnn[1]
        terms.append("{%d'b0, xnn[0]}" % (PC - 1) if PC > 1 else "xnn[0]")
        terms.append("{%d'b0, xnn[1]}" % (PC - 1) if PC > 1 else "xnn[1]")
    elif cur_count == 2:
        # Normal case: 2 elements remain in sum_Llast
        ext = PC - last_w
        if ext > 0:
            terms.append("{%d'b0, %s[0]}" % (ext, last_src))
            terms.append("{%d'b0, %s[1]}" % (ext, last_src))
        else:
            terms.append("%s[0]" % last_src)
            terms.append("%s[1]" % last_src)
    else:
        # cur_count == 1: one element remains
        ext = PC - last_w
        if ext > 0:
            terms.append("{%d'b0, %s[0]}" % (ext, last_src))
        else:
            terms.append("%s[0]" % last_src)

    # Add remainders
    for (rexpr, rw) in remainders:
        ext = PC - rw
        if ext > 0:
            terms.append("{%d'b0, %s}" % (ext, rexpr))
        else:
            terms.append(rexpr)

    final_assign = " + ".join(terms)
    return wire_decls, gen_lines, final_assign, PC


def int_mac_tree(N, start_w=11):
    """
    Build the reduction tree for int_mac_N (signed accumulators).

    Matches int_mac_64.v style exactly:
      • Single genvar j; generate with named begin:gen_rN blocks.
      • Loops run while count > 2.
      • Final assign uses $signed(s_last[0]) + $signed(s_last[1]) outside.
      • Remainders from odd counts are added to the final assign.

    Returns:
      wire_decls   : list of "wire signed [w:0] sk [0:pairs-1];" strings
      gen_lines    : lines for the single generate block
      final_expr   : RHS string for the final wire assignment
      final_wire   : "wire signed [w:0] s_final;" declaration string
      res_w        : bit-width of s_final
    """
    levels     = []   # (lev, pairs, src, in_w, out_w)
    remainders = []   # (verilog_expr_str, width)

    cur_count = N
    cur_width = start_w
    lev       = 0
    src       = "acc"

    while cur_count > 2:
        pairs = cur_count // 2
        odd   = cur_count % 2
        new_w = cur_width + 1
        lev  += 1

        levels.append((lev, pairs, src, cur_width, new_w))

        if odd:
            rem_expr = "%s[%d]" % (src, cur_count - 1)
            remainders.append((rem_expr, cur_width))

        src       = "s%d" % lev
        cur_count = pairs
        cur_width = new_w

    last_lev = lev
    last_src = src
    last_w   = cur_width

    # ── wire declarations ────────────────────────────────────────────────────
    wire_decls = []
    for (l, p, s, iw, ow) in levels:
        wire_decls.append("wire signed [%d:0] s%d [0:%d];" % (ow - 1, l, p - 1))

    # ── single generate block ────────────────────────────────────────────────
    gen_lines = ["genvar j;", "generate"]
    for (l, p, s, iw, ow) in levels:
        gen_lines.append("    for (j = 0; j < %d; j = j + 1) begin : gen_r%d" % (p, l))
        gen_lines.append("        assign s%d[j] = $signed(%s[2*j]) + $signed(%s[2*j+1]);" % (l, s, s))
        gen_lines.append("    end")
    gen_lines.append("endgenerate")

    # ── final expression (outside generate) ──────────────────────────────────
    # Result width = last_w (+ 1 if we need to accommodate remainders)
    # Determine result width conservatively: last_w + enough for remainders
    res_w = last_w + (1 if remainders else 0)

    # Build terms for final assign
    terms = []

    if cur_count == 2 and lev == 0:
        # Edge: N=2
        terms.append("$signed(acc[0])")
        terms.append("$signed(acc[1])")
    elif cur_count == 2:
        terms.append("$signed(%s[0])" % last_src)
        terms.append("$signed(%s[1])" % last_src)
    else:
        terms.append("$signed(%s[0])" % last_src)

    for (rexpr, rw) in remainders:
        ext = res_w - rw
        if ext > 0:
            terms.append("$signed({{{%d{{%s[%d]}}}}, %s})" % (ext, rexpr, rw - 1, rexpr))
        else:
            terms.append("$signed(%s)" % rexpr)

    final_expr = " + ".join(terms)
    final_wire = "wire signed [%d:0] s_final;" % (res_w - 1)

    return wire_decls, gen_lines, final_expr, final_wire, res_w


# =============================================================================
# bfp_aligner.v  (only N changes)
# =============================================================================
def exp_max_tree(N):
    """
    Build a LOG-DEPTH tree-based max reduction over N 4-bit exponent values.

    This replaces a naive procedural 'for(i) if (exps[i] > max) max = exps[i]'
    loop, which synthesizes as an O(N) chain of N-1 comparators in series
    (each iteration waits on the previous result). At N=512 that is up to
    511 sequential 4-bit comparisons in ONE combinational block -- the actual
    cause of the timing failure inside bfp_aligner even after pipelining its
    output. A binary max-tree instead gives O(log2 N) comparator depth
    (9 levels at N=512), with all comparisons at a given level independent
    and parallel.

    Structure mirrors popcount_tree(): run pairwise-max generate loops while
    count > 2, carrying forward any odd leftover element as a "remainder"
    that is combined at the very end.

    Returns:
      wire_decls  : list of "wire [3:0] maxL<k> [0:pairs-1];" strings
      gen_lines   : lines for the generate block(s) (single genvar 'm', reused)
      final_expr  : Verilog ternary-chain expression evaluating to the global max
    """
    levels     = []   # (lev, pairs, src)
    remainders = []   # list of verilog expr strings

    cur_count = N
    lev       = 0
    src       = "exps_w"

    while cur_count > 2:
        pairs = cur_count // 2
        odd   = cur_count % 2
        lev  += 1

        levels.append((lev, pairs, src))

        if odd:
            remainders.append("%s[%d]" % (src, cur_count - 1))

        src       = "maxL%d" % lev
        cur_count = pairs

    last_src = src

    # ── wire declarations ────────────────────────────────────────────────
    wire_decls = []
    for (l, p, s) in levels:
        wire_decls.append("wire [3:0] maxL%d [0:%d];" % (l, p - 1))

    # ── generate block(s) ────────────────────────────────────────────────
    gen_lines = ["genvar m;", "generate"]
    for (l, p, s) in levels:
        gen_lines.append(
            "    for(m=0; m<%d; m=m+1) assign maxL%d[m] = (%s[2*m] > %s[2*m+1]) ? %s[2*m] : %s[2*m+1];"
            % (p, l, s, s, s, s)
        )
    gen_lines.append("endgenerate")

    # ── final combine (few terms: last 1-2 tree outputs + any remainders) ──
    if cur_count == 2 and lev == 0:
        terms = ["exps_w[0]", "exps_w[1]"]
    elif cur_count == 2:
        terms = ["%s[0]" % last_src, "%s[1]" % last_src]
    else:
        terms = ["%s[0]" % last_src]
    terms += remainders

    expr = terms[0]
    for t in terms[1:]:
        expr = "(((%s) > (%s)) ? (%s) : (%s))" % (expr, t, expr, t)

    return wire_decls, gen_lines, expr


# =============================================================================
# bfp_aligner.v  (only N changes)
# =============================================================================
def gen_bfp_aligner(N):
    VW = N * 8
    PW = N * 4

    wire_decls, gen_lines, max_expr = exp_max_tree(N)

    lines = [
        "module bfp_aligner (",
        "    input  clk,",
        "    input  [%d:0] fp8_vec," % (VW - 1),
        "    output reg [%d:0] aligned_planes, // Reorganized for MLB_4 axi/awi ports" % (PW - 1),
        "    output reg [3:0]   max_exp",
        ");",
        "    genvar g;",
        "",
        "    wire [3:0] exps_w [0:%d];" % (N - 1),
        "    generate",
        "        for (g=0; g<%d; g=g+1) begin : extract_exp" % N,
        "            assign exps_w[g] = fp8_vec[(g*8)+3 +: 4];",
        "        end",
        "    endgenerate",
        "",
    ]
    for wd in wire_decls:
        lines.append("    " + wd)
    lines.append("")
    for gl in gen_lines:
        lines.append("    " + gl)
    lines.append("")
    lines.append("    wire [3:0] max_exp_c = %s;" % max_expr)
    lines.append("")
    lines += [
        "    wire [3:0] mants_w        [0:%d];" % (N - 1),
        "    wire [3:0] shifted_mant_w [0:%d];" % (N - 1),
        "    generate",
        "        for (g=0; g<%d; g=g+1) begin : align_lane" % N,
        "            assign mants_w[g] = (exps_w[g] != 4'd0) ? {1'b1, fp8_vec[(g*8) +: 3]} : 4'd0;",
        "            assign shifted_mant_w[g] = mants_w[g] >> (max_exp_c - exps_w[g]);",
        "        end",
        "    endgenerate",
        "",
        "    wire [%d:0] aligned_planes_c;" % (PW - 1),
        "    generate",
        "        for (g=0; g<%d; g=g+1) begin : corner_turn" % N,
        "            assign aligned_planes_c[g]         = shifted_mant_w[g][0]; // Plane 0 (axi[%d:0])" % (N - 1),
        "            assign aligned_planes_c[%d + g]  = shifted_mant_w[g][1]; // Plane 1 (axi[%d:%d])" % (N, 2*N-1, N),
        "            assign aligned_planes_c[%d + g] = shifted_mant_w[g][2]; // Plane 2 (axi[%d:%d])" % (2*N, 3*N-1, 2*N),
        "            assign aligned_planes_c[%d + g] = shifted_mant_w[g][3]; // Plane 3 (axi[%d:%d])" % (3*N, 4*N-1, 3*N),
        "        end",
        "    endgenerate",
        "",
        "    always @(posedge clk) begin",
        "        aligned_planes <= aligned_planes_c;",
        "        max_exp        <= max_exp_c;",
        "    end",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# and_popcount_4_bit.v  — SINGLE generate block, exact original style
# =============================================================================
def gen_and_popcount(N):
    wire_decls, gen_lines, final_assign, PC = popcount_tree(N)

    lines = [
        "module and_popcount_4_bit(",
        "    output reg [%d:0] f_output," % (PC - 1),
        "    output reg done,",
        "    input [%d:0] a, b," % (N - 1),
        "    input clk, rst, valid_in",
        ");",
        "    wire [%d:0] xnn = a & b;" % (N - 1),
        "",
    ]
    for wd in wire_decls:
        lines.append("    " + wd)
    lines.append("")
    for gl in gen_lines:
        lines.append("    " + gl)
    lines.append("")
    lines.append("    assign final_popcount = %s;" % final_assign)
    lines.append("")
    lines.append("    // Pipeline Logic")
    lines.append("    reg cycle_cnt;")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if(rst) begin")
    lines.append("            f_output  <= %d'sd0;" % PC)
    lines.append("            done      <= 1'b0;")
    lines.append("            cycle_cnt <= 1'b0;")
    lines.append("        end else if(valid_in) begin")
    lines.append("            if(cycle_cnt == 1'b0) begin")
    lines.append("                f_output  <= final_popcount;")
    lines.append("                cycle_cnt <= 1'b0;")
    lines.append("                done      <= 1'b1;")
    lines.append("            end")
    lines.append("        end else begin")
    lines.append("            done <= 1'b0;")
    lines.append("        end")
    lines.append("    end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


# =============================================================================
# MLB_MAC_unit.v  (only widths change)
# =============================================================================
def gen_mlb_unit(N):
    PC   = pc_bits(N)
    OUTW = PC + 9   # 1 sign bit + PC bits popcount + 6 bits max shift (i+j ≤ 6)

    lines = [
        "module MLB_unit(",
        "    output signed [%d:0] out," % (OUTW - 1),
        "    output done,",
        "    input [%d:0] axi, awi," % (N - 1),
        "    input [2:0] shift_amt, // Replaces alpha_x and alpha_w",
        "    input clk, rst, valid_in",
        ");",
        "    wire [%d:0] inter;" % (PC - 1),
        "    wire xp_done;",
        "    ",
        "    and_popcount_4_bit xp(",
        "        .f_output(inter),",
        "        .done(xp_done),",
        "        .a(axi),",
        "        .b(awi),",
        "        .clk(clk),",
        "        .rst(rst),",
        "        .valid_in(valid_in)",
        "    );",
        "",
        "    reg signed [%d:0] shifted_out;" % (OUTW - 1),
        "    reg done_reg;",
        "",
        "    always @(posedge clk) begin",
        "        if (rst) begin",
        "            shifted_out <= %d'sd0;" % OUTW,
        "            done_reg <= 1'b0;",
        "        end else begin",
        "            done_reg <= xp_done;",
        "            if (xp_done) begin",
        "                shifted_out <= $signed({1'b0, inter}) << shift_amt;",
        "            end",
        "        end",
        "    end",
        "",
        "    assign out = shifted_out;",
        "    assign done = done_reg;",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# MLB_MAC_4_bit.v  (only widths / slice indices change)
# =============================================================================
def gen_mlb_4(N):
    PC    = pc_bits(N)
    UNITW = PC + 9
    PW    = N * 4
    L1W   = UNITW + 1
    L2W   = UNITW + 2
    L3W   = UNITW + 3
    MLBW  = UNITW + 4

    lines = [
        "module MLB_4(",
        "    output reg signed [%d:0] mlb," % (MLBW - 1),
        "    output reg done,",
        "    input [%d:0] axi, awi," % (PW - 1),
        "    input clk, rst, valid_in",
        ");",
        "    genvar i, j;",
        "    wire signed [%d:0] out[3:0][3:0];" % (UNITW - 1),
        "    wire [15:0] unit_done;",
        "",
        "    generate",
        "        for(i=0; i<4; i=i+1) begin : rows",
        "            for(j=0; j<4; j=j+1) begin : cols",
        "                // Create a strict 3-bit wire to silence the port size warning",
        "                wire [2:0] shift_val = i + j; ",
        "                ",
        "                MLB_unit u_ij(",
        "                    .out(out[i][j]),",
        "                    .done(unit_done[4*i+j]),",
        "                    .shift_amt(shift_val), ",
        "                    .axi(axi[i*%d+%d : %d*i])," % (N, N-1, N),
        "                    .awi(awi[j*%d+%d : %d*j])," % (N, N-1, N),
        "                    .clk(clk),",
        "                    .rst(rst),",
        "                    .valid_in(valid_in)",
        "                );",
        "            end",
        "        end",
        "    endgenerate",
        "",
        "    // Reduction tree (Unchanged)",
        "    wire signed [%d:0] s00,s01,s02,s03,s04,s05,s06,s07;" % (L1W - 1),
        "    wire signed [%d:0] s10,s11,s12,s13;" % (L2W - 1),
        "    wire signed [%d:0] s20,s21;" % (L3W - 1),
        "",
        "    assign s00=out[0][0]+out[0][1]; assign s01=out[0][2]+out[0][3];",
        "    assign s02=out[1][0]+out[1][1]; assign s03=out[1][2]+out[1][3];   ",
        "    assign s04=out[2][0]+out[2][1]; assign s05=out[2][2]+out[2][3];",
        "    assign s06=out[3][0]+out[3][1]; assign s07=out[3][2]+out[3][3];",
        "",
        "    assign s10=s00+s01; assign s11=s02+s03;",
        "    assign s12=s04+s05; assign s13=s06+s07;",
        "    assign s20=s10+s11; assign s21=s12+s13;",
        "",
        "    always @(posedge clk) begin",
        "        if(rst) begin",
        "            mlb  <= %d'sd0;" % MLBW,
        "            done <= 1'b0;",
        "        end else begin",
        "            done <= 1'b0; // Default state",
        "            if(&unit_done) begin",
        "                mlb  <= s20+s21;",
        "                done <= 1'b1;",
        "            end",
        "        end",
        "    end",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# FP8.v  (fp8_mlb_top) — only widths change
# =============================================================================
def gen_fp8_top(N):
    VW    = N * 8
    PW    = N * 4
    PC    = pc_bits(N)
    UNITW = PC + 9
    MLBW  = UNITW + 4
    SUMW  = MLBW + 1

    lines = [
        "module fp8_mlb_top(output signed [%d:0] wide_integer_sum,output signed [8:0] shared_exponent,output mac_done,input clk,rst,valid_in,input [%d:0] fp8_activations,fp8_weights);" % (SUMW-1, VW-1),
        "    wire [%d:0] axi_planes;" % (PW - 1),
        "    wire [%d:0] awi_planes;" % (PW - 1),
        "    wire [3:0] max_exp_x;",
        "    wire [3:0] max_exp_w;",
        "    bfp_aligner align_x (.clk(clk),.fp8_vec(fp8_activations),.aligned_planes(axi_planes),.max_exp(max_exp_x));",
        "    bfp_aligner align_w (.clk(clk),.fp8_vec(fp8_weights),.aligned_planes(awi_planes),.max_exp(max_exp_w));",
        "    wire [%d:0] sign_k;" % (N - 1),
        "    genvar i;",
        "    generate",
        "        for (i=0;i<%d;i=i+1) begin" % N,
        "            assign sign_k[i]=fp8_activations[i*8+7]^fp8_weights[i*8+7];",
        "        end",
        "    endgenerate",
        "    reg [%d:0] sign_k_d1;" % (N - 1),
        "    always @(posedge clk) sign_k_d1 <= sign_k;",
        "    wire [%d:0] pos_mask = ~sign_k_d1;" % (N - 1),
        "    wire [%d:0] neg_mask = sign_k_d1;" % (N - 1),
        "    wire [%d:0] axi_planes_pos;" % (PW - 1),
        "    wire [%d:0] axi_planes_neg;" % (PW - 1),
        "    assign axi_planes_pos[%d:0]    = axi_planes[%d:0]    & pos_mask;" % (N-1, N-1),
        "    assign axi_planes_pos[%d:%d]  = axi_planes[%d:%d]  & pos_mask;" % (2*N-1, N, 2*N-1, N),
        "    assign axi_planes_pos[%d:%d] = axi_planes[%d:%d] & pos_mask;" % (3*N-1, 2*N, 3*N-1, 2*N),
        "    assign axi_planes_pos[%d:%d] = axi_planes[%d:%d] & pos_mask;" % (4*N-1, 3*N, 4*N-1, 3*N),
        "    ",
        "    assign axi_planes_neg[%d:0]    = axi_planes[%d:0]    & neg_mask;" % (N-1, N-1),
        "    assign axi_planes_neg[%d:%d]  = axi_planes[%d:%d]  & neg_mask;" % (2*N-1, N, 2*N-1, N),
        "    assign axi_planes_neg[%d:%d] = axi_planes[%d:%d] & neg_mask;" % (3*N-1, 2*N, 3*N-1, 2*N),
        "    assign axi_planes_neg[%d:%d] = axi_planes[%d:%d] & neg_mask;" % (4*N-1, 3*N, 4*N-1, 3*N),
        "",
        "    reg valid_in_d1;",
        "    always @(posedge clk) begin",
        "        if (rst) valid_in_d1 <= 1'b0;",
        "        else     valid_in_d1 <= valid_in;",
        "    end",
        "",
        "    reg is_neg_phase;",
        "    reg internal_valid;",
        "    reg [1:0] state;",
        "    always @(posedge clk) begin",
        "        if (rst) begin",
        "            state <= 2'd0;",
        "            internal_valid <= 1'b0;",
        "            is_neg_phase <= 1'b0;",
        "        end else begin",
        "            case (state)",
        "                2'd0: begin",
        "                    if (valid_in_d1) begin",
        "                        internal_valid <= 1'b1;",
        "                        is_neg_phase <= 1'b0; // Launch Positive mask first",
        "                        state <= 2'd1;",
        "                    end else begin",
        "                        internal_valid <= 1'b0;",
        "                    end",
        "                end",
        "                2'd1: begin",
        "                    internal_valid <= 1'b1;",
        "                    is_neg_phase <= 1'b1; // Immediately launch Negative mask next clock",
        "                    state <= 2'd2;",
        "                end",
        "                2'd2: begin",
        "                    internal_valid <= 1'b0;",
        "                    state <= 2'd0; // Wait for next valid_in from testbench",
        "                end",
        "            endcase",
        "        end",
        "    end",
        "",
        "    wire [%d:0] muxed_axi_planes = is_neg_phase ? axi_planes_neg : axi_planes_pos;" % (PW - 1),
        "",
        "    wire signed [%d:0] shared_sum_out;" % (MLBW - 1),
        "    wire shared_done;",
        "",
        "    MLB_4 mac_shared (.mlb(shared_sum_out),.done(shared_done),.axi(muxed_axi_planes),.awi(awi_planes),.clk(clk),.rst(rst),.valid_in(internal_valid));",
        "",
        "    reg signed [%d:0] pos_reg;" % (MLBW - 1),
        "    reg got_pos;",
        "    reg signed [%d:0] result_reg;" % (SUMW - 1),
        "    reg done_reg;",
        "",
        "    always @(posedge clk) begin",
        "        if (rst) begin",
        "            pos_reg <= %d'sd0;" % MLBW,
        "            got_pos <= 1'b0;",
        "            result_reg <= %d'sd0;" % SUMW,
        "            done_reg <= 1'b0;",
        "        end else begin",
        "            done_reg <= 1'b0;",
        "            ",
        "            if (shared_done) begin",
        "                if (!got_pos) begin",
        "                    pos_reg <= shared_sum_out;",
        "                    got_pos <= 1'b1;",
        "                end else begin",
        "                    result_reg <= pos_reg - shared_sum_out; // pos_sum - neg_sum",
        "                    done_reg <= 1'b1; // Fire final done signal to testbench",
        "                    got_pos <= 1'b0;  // Reset for the next inference",
        "                end",
        "            end",
        "        end",
        "    end",
        "",
        "    assign wide_integer_sum = result_reg;",
        "    assign mac_done         = done_reg;",
        "    assign shared_exponent  = $signed({5'b0, max_exp_x}) + $signed({5'b0, max_exp_w}) - 9'sd20; // 14 (2x E4M3 bias) + 6 (2x hidden-bit integer scale 2^3)",
        "",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# tb_FP8.v  (MLB top testbench) — N-scaled, same structure
# =============================================================================
def gen_tb_mlb(N):
    VW    = N * 8
    PC    = pc_bits(N)
    UNITW = PC + 9
    MLBW  = UNITW + 4
    SUMW  = MLBW + 1

    code = """\
`timescale 1ns/1ps

module tb_fp8_mlb_top;
    reg  clk, rst, valid_in;
    reg  [%(VW_1)d:0] fp8_activations, fp8_weights;
    wire signed [%(SW_1)d:0] wide_integer_sum;
    wire signed [8:0]   shared_exponent;
    wire         mac_done;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    fp8_mlb_top dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .fp8_activations(fp8_activations),
        .fp8_weights(fp8_weights),
        .wide_integer_sum(wide_integer_sum),
        .shared_exponent(shared_exponent),
        .mac_done(mac_done)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg [7:0] act_lane [0:%(N_1)d];
    reg [7:0] wt_lane  [0:%(N_1)d];
    initial begin
        $dumpfile("mlb_mac_sim.vcd");
        $dumpvars(0, tb_fp8_mlb_top);
    end

    function [7:0] rand_fp8(input allow_zero);
        reg [31:0] r1, r2, r3;
        reg [3:0]  exp_f;
        begin
            r1 = $random;
            r2 = $random;
            r3 = $random;
            if (allow_zero)
                exp_f = r1 %% 16;
            else
                exp_f = (r1 %% 15) + 1;
            rand_fp8 = {r2[0], exp_f, r3[2:0]};
        end
    endfunction

    task clear_all_lanes;
        integer i;
        begin
            for (i = 0; i < %(N)d; i = i + 1) begin
                act_lane[i] = 8'b0;
                wt_lane[i]  = 8'b0;
            end
        end
    endtask

task set_lane(input integer is_weight, input integer idx, input sign, input [3:0] exp_f, input [2:0] mant_f);
        begin
            if (is_weight)
                wt_lane[idx] = {sign, exp_f, mant_f};
            else
                act_lane[idx] = {sign, exp_f, mant_f};
        end
    endtask

task fill_random_lanes(input allow_zero);
        integer i;
        begin
            for (i = 0; i < %(N)d; i = i + 1) begin
                act_lane[i] = rand_fp8(allow_zero);
                wt_lane[i]  = rand_fp8(allow_zero);
            end
        end
    endtask
    task pack_vectors;
        integer i;
        begin
            for (i = 0; i < %(N)d; i = i + 1) begin
                fp8_activations[i*8 +: 8] = act_lane[i];
                fp8_weights[i*8 +: 8]     = wt_lane[i];
            end
        end
    endtask

task compute_reference(output reg signed [%(SW_1)d:0] ref_result,
                            output reg [8:0]         ref_exponent);
        integer i;
        reg sx, sw;
        reg [3:0] ex, ew;
        reg [3:0] mx, mw;
        reg [3:0] max_ex, max_ew;
        integer shift_x, shift_w;
        reg [3:0] smx, smw;
        integer prod;
        integer pos_sum, neg_sum;
        begin
            max_ex = 4'd0;
            max_ew = 4'd0;
            for (i = 0; i < %(N)d; i = i + 1) begin
                ex = act_lane[i][6:3];
                ew = wt_lane[i][6:3];
                if (ex > max_ex) max_ex = ex;
                if (ew > max_ew) max_ew = ew;
            end

            pos_sum = 0;
            neg_sum = 0;
            for (i = 0; i < %(N)d; i = i + 1) begin
                sx = act_lane[i][7];
                ex = act_lane[i][6:3];
                mx = (ex != 4'd0) ? {1'b1, act_lane[i][2:0]} : 4'd0;

                sw = wt_lane[i][7];
                ew = wt_lane[i][6:3];
                mw = (ew != 4'd0) ? {1'b1, wt_lane[i][2:0]} : 4'd0;

                shift_x = max_ex - ex;
                shift_w = max_ew - ew;

                smx = mx >> shift_x;
                smw = mw >> shift_w;

                prod = smx * smw;

                if (sx ^ sw)
                    neg_sum = neg_sum + prod;
                else
                    pos_sum = pos_sum + prod;
            end

            ref_result   = pos_sum - neg_sum;
            ref_exponent = $signed({5'b0, max_ex}) + $signed({5'b0, max_ew}) - 9'sd20;
        end
    endtask
task apply_and_check(input [8*40-1:0] name);
        reg signed [%(SW_1)d:0] expected_result;
        reg [8:0]         expected_exponent;
        integer timeout;
        begin
            test_num = test_num + 1;
            pack_vectors;
            compute_reference(expected_result, expected_exponent);

            @(negedge clk);
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            timeout = 0;
            while (!mac_done && timeout < 2000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end

            @(negedge clk); // Safe read point after done
            if (timeout >= 2000) begin
                $display("[TEST %%0d] %%-0s : TIMEOUT waiting for mac_done", test_num, name);
                fail_count = fail_count + 1;
            end
            else if ((wide_integer_sum !== expected_result) ||
                     (shared_exponent !== expected_exponent)) begin
                $display("[TEST %%0d] %%-0s : FAIL", test_num, name);
                $display("           expected : sum=%%0d  exp=%%0d",
                          expected_result, expected_exponent);
                $display("           got      : sum=%%0d  exp=%%0d",
                          wide_integer_sum, shared_exponent);
                fail_count = fail_count + 1;
            end
            else begin
                $display("[TEST %%0d] %%-0s : PASS  (sum=%%0d, exp=%%0d)",
                          test_num, name, wide_integer_sum, shared_exponent);
                pass_count = pass_count + 1;
            end

            @(posedge clk);
        end
    endtask
    integer t;
    integer i;
    initial begin
        rst      = 1'b1;
        valid_in = 1'b0;
        fp8_activations = %(VW)d'b0;
        fp8_weights     = %(VW)d'b0;
        clear_all_lanes;

        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        clear_all_lanes;
        apply_and_check("All zeros");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b0, 4'd7, 3'b000);
        end
        apply_and_check("All positive equal value");


        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b1, 4'd7, 3'b000);
        end
        apply_and_check("All positive act, all negative weight");


        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b1, 4'd7, 3'b011);
            set_lane(1, i, 1'b1, 4'd7, 3'b011);
        end
        apply_and_check("Both negative (signs cancel)");

    
        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, i[0], 4'd9, 3'b010);
            set_lane(1, i, 1'b0,  4'd9, 3'b101);
        end
        apply_and_check("Alternating activation sign");


        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd15, 3'b111);
            set_lane(1, i, 1'b0, 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, all positive");


        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, i[0],  4'd15, 3'b111);
            set_lane(1, i, ~i[0], 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, alternating signs");


        for (i = 0; i < %(N)d; i = i + 1) begin
            if (i[0]) begin
                set_lane(0, i, 1'b0, 4'd0, 3'b101); // exp=0 -> flushed to 0
                set_lane(1, i, 1'b0, 4'd0, 3'b011); // exp=0 -> flushed to 0
            end else begin
                set_lane(0, i, 1'b0, 4'd8, 3'b001);
                set_lane(1, i, 1'b0, 4'd8, 3'b001);
            end
        end
        apply_and_check("Subnormal flush (half lanes exp=0)");


        clear_all_lanes;
        set_lane(0, 0, 1'b0, 4'd10, 3'b110);
        set_lane(1, 0, 1'b1, 4'd10, 3'b110);
        apply_and_check("Single nonzero lane");

        clear_all_lanes;
        set_lane(0, 0, 1'b0, 4'd15, 3'b111);
        set_lane(1, 0, 1'b0, 4'd15, 3'b111);
        for (i = 1; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd1, 3'b000);
            set_lane(1, i, 1'b0, 4'd1, 3'b000);
        end
        apply_and_check("Wide dynamic range (shift-to-zero stress)");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd0, 3'b111);
            set_lane(1, i, 1'b0, 4'd12, 3'b101);
        end
        apply_and_check("All activations flushed to zero");


        for (t = 0; t < 50; t = t + 1) begin
            if (t %% 3 == 0)
                fill_random_lanes(1'b1);  // allow zero-exponent lanes
            else
                fill_random_lanes(1'b0);  // normal numbers only
            apply_and_check("Random test");
        end
        $display("\\n======================================================");
        $display("  TEST SUMMARY: %%0d PASSED,  %%0d FAILED",
                  pass_count, fail_count);
        $display("======================================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  *** FAILURES DETECTED -- check waveform in mlb_mac_sim.vcd ***");
        $display("");

        $finish;
    end

endmodule
"""
    return code % dict(N=N, N_1=N-1, VW=VW, VW_1=VW-1, SW_1=SUMW-1)


# =============================================================================
# int_mac_N.v  — SINGLE generate block, exact original style
# =============================================================================
def gen_int_mac_n(N):
    VW = N * 4

    wire_decls, gen_lines, final_expr, final_wire, res_w = int_mac_tree(N)
    scaled_w = res_w + 9

    wd = "\n".join("    " + x for x in wire_decls)
    ga = "\n".join("    " + x for x in gen_lines)

    code = """\

module int_mac_%(N)d (
    input clk,
    input rst,
    input load,

    input [%(VW_1)d:0] a_flat,    // %(N)d x 4-bit UNSIGNED activations
    input [%(VW_1)d:0] b_flat,    // %(N)d x 4-bit UNSIGNED weights
    input [%(N_1)d:0]  sign_flat, // %(N)d x 1-bit product sign (sign_x XOR sign_w)

    input [3:0]        alpha_x,
    input [3:0]        alpha_w,
    input signed [7:0] beta_xw,

    output signed [20:0] result
);

localparam GROUP = 16;
localparam NGRP  = (%(N)d + GROUP - 1) / GROUP;

wire rst_buf  [0:NGRP-1];
wire load_buf [0:NGRP-1];

genvar g;
generate
    for (g = 0; g < NGRP; g = g + 1) begin : gen_ctrl_buf
        assign rst_buf[g]  = rst;
        assign load_buf[g] = load;
    end
endgenerate

wire [7:0]         product       [0:%(N_1)d]; // unsigned magnitude product
wire signed [8:0]  signed_product[0:%(N_1)d]; // sign applied after multiply
reg  signed [10:0] acc           [0:%(N_1)d]; // accumulator, 11-bit for M=4

genvar i;
generate
    for (i = 0; i < %(N)d; i = i + 1) begin : gen_mac_lane


        assign product[i] = a_flat[4*i +: 4] * b_flat[4*i +: 4];
        assign signed_product[i] = sign_flat[i]
                                    ? -$signed({1'b0, product[i]})
                                    :  $signed({1'b0, product[i]});
        always @(posedge clk) begin
            if (rst_buf[i/GROUP])
                acc[i] <= 11'sd0;
            else if (load_buf[i/GROUP])
                acc[i] <= acc[i] + {{2{signed_product[i][8]}}, signed_product[i]};
        end

    end
endgenerate


%(WD)s

%(GA)s

%(FW)s
assign s_final = %(FE)s;


wire [7:0]        alpha_prod;   // ax x aw (unsigned 8-bit)
wire signed [8:0] alpha_prod_s; // zero-extended to signed
wire signed [%(SW_1)d:0] scaled;

assign alpha_prod   = alpha_x * alpha_w;
assign alpha_prod_s = {1'b0, alpha_prod};
assign scaled       = $signed(s_final) * $signed(alpha_prod_s);

assign result = scaled[20:0] + {{13{beta_xw[7]}}, beta_xw};

endmodule
"""
    return code % dict(
        N=N, N_1=N-1, VW=VW, VW_1=VW-1,
        WD=wd, GA=ga,
        FW=final_wire, FE=final_expr,
        RW=res_w, SW=scaled_w, SW_1=scaled_w-1,
    )


# =============================================================================
# fp8_int_top.v  (only N changes)
# =============================================================================
def gen_fp8_int_top(N):
    VW = N * 8
    PW = N * 4

    lines = [
        "module fp8_int_top (",
        "    input  clk,",
        "    input  rst,",
        "    input  valid_in, // Connects to 'load' on the MAC array",
        "    input  [%d:0] fp8_activations," % (VW - 1),
        "    input  [%d:0] fp8_weights," % (VW - 1),
        "    output signed [20:0] wide_integer_sum,",
        "    output signed [8:0]   shared_exponent",
        ");",
        "",
        "    wire [%d:0] axi_planes;" % (PW - 1),
        "    wire [%d:0] awi_planes;" % (PW - 1),
        "    wire [3:0]   max_exp_x;",
        "    wire [3:0]   max_exp_w;",
        "",
        "    bfp_aligner align_x (",
        "        .clk(clk),",
        "        .fp8_vec(fp8_activations),",
        "        .aligned_planes(axi_planes),",
        "        .max_exp(max_exp_x)",
        "    );",
        "",
        "    bfp_aligner align_w (",
        "        .clk(clk),",
        "        .fp8_vec(fp8_weights),",
        "        .aligned_planes(awi_planes),",
        "        .max_exp(max_exp_w)",
        "    );",
        "",
        "    reg [%d:0] fp8_activations_d1;" % (VW - 1),
        "    reg [%d:0] fp8_weights_d1;" % (VW - 1),
        "    always @(posedge clk) begin",
        "        fp8_activations_d1 <= fp8_activations;",
        "        fp8_weights_d1     <= fp8_weights;",
        "    end",
        "",
        "    reg [3:0]  a_flat_reg  [0:%d]; // 4-bit unsigned mantissa per lane" % (N - 1),
        "    reg [3:0]  b_flat_reg  [0:%d]; // 4-bit unsigned mantissa per lane" % (N - 1),
        "    reg [%d:0] sign_flat_reg;      // product sign per lane: sign_x XOR sign_w" % (N - 1),
        "",
        "    wire [%d:0] a_flat_wire;" % (PW - 1),
        "    wire [%d:0] b_flat_wire;" % (PW - 1),
        "",
        "    integer i;",
        "    reg [3:0] mant_x, mant_w;",
        "",
        "    always @(*) begin",
        "        for (i = 0; i < %d; i = i + 1) begin" % N,
        "            // 1. Reconstruct full 4-bit unsigned mantissa from BFP bit-planes",
        "            mant_x = {axi_planes[%d+i], axi_planes[%d+i], axi_planes[%d+i], axi_planes[i]};" % (3*N, 2*N, N),
        "            mant_w = {awi_planes[%d+i], awi_planes[%d+i], awi_planes[%d+i], awi_planes[i]};" % (3*N, 2*N, N),
        "",
        "            a_flat_reg[i] = mant_x;",
        "            b_flat_reg[i] = mant_w;",
        "",
        "            // 2. Compute product sign (sign-magnitude rule: sign = XOR of input signs)",
        "            //    using the delayed (aligned-in-time) activation/weight copies.",
        "            sign_flat_reg[i] = fp8_activations_d1[i*8 + 7] ^ fp8_weights_d1[i*8 + 7];",
        "        end",
        "    end",
        "",
        "    // Pack mantissa registers into flat buses",
        "    genvar k;",
        "    generate",
        "        for (k = 0; k < %d; k = k + 1) begin : pack_flat" % N,
        "            assign a_flat_wire[4*k +: 4] = a_flat_reg[k];",
        "            assign b_flat_wire[4*k +: 4] = b_flat_reg[k];",
        "        end",
        "    endgenerate",
        "",
        "    reg valid_in_d1;",
        "    always @(posedge clk) begin",
        "        if (rst) valid_in_d1 <= 1'b0;",
        "        else     valid_in_d1 <= valid_in;",
        "    end",
        "",
        "    int_mac_%d mac_array (" % N,
        "        .clk(clk),",
        "        .rst(rst),",
        "        .load(valid_in_d1),",
        "        .a_flat(a_flat_wire),",
        "        .b_flat(b_flat_wire),",
        "        .sign_flat(sign_flat_reg),   // per-lane product sign",
        "        .alpha_x(4'd1),",
        "        .alpha_w(4'd1),",
        "        .beta_xw(8'sd0),",
        "        .result(wide_integer_sum)",
        "    );",
        "",
        "    assign shared_exponent = $signed({5'b0, max_exp_x}) + $signed({5'b0, max_exp_w}) - 9'sd20;",
        "",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# tb_FP8.v  (INT MAC testbench) — N-scaled, same structure
# =============================================================================
def gen_tb_int(N):
    VW = N * 8

    code = """\
`timescale 1ns/1ps

module tb_fp8_int_top;
    reg  clk, rst, valid_in;
    reg  [%(VW_1)d:0] fp8_activations, fp8_weights;
    wire signed [20:0] wide_integer_sum;
    wire signed [8:0]   shared_exponent;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    fp8_int_top dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .fp8_activations(fp8_activations),
        .fp8_weights(fp8_weights),
        .wide_integer_sum(wide_integer_sum),
        .shared_exponent(shared_exponent)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;


    reg [7:0] act_lane [0:%(N_1)d];
    reg [7:0] wt_lane  [0:%(N_1)d];


    initial begin
        $dumpfile("int_mac_sim.vcd");
        $dumpvars(0, tb_fp8_int_top);
    end

    function [7:0] rand_fp8(input allow_zero);
        reg [31:0] r1, r2, r3;
        reg [3:0]  exp_f;
        begin
            r1 = $random;
            r2 = $random;
            r3 = $random;
            if (allow_zero)
                exp_f = r1 %% 16;
            else
                exp_f = (r1 %% 15) + 1;
            rand_fp8 = {r2[0], exp_f, r3[2:0]};
        end
    endfunction

    task clear_all_lanes;
        integer i;
        begin
            for (i = 0; i < %(N)d; i = i + 1) begin
                act_lane[i] = 8'b0;
                wt_lane[i]  = 8'b0;
            end
        end
    endtask

task set_lane(input integer is_weight, input integer idx, input sign, input [3:0] exp_f, input [2:0] mant_f);
        begin
            if (is_weight)
                wt_lane[idx] = {sign, exp_f, mant_f};
            else
                act_lane[idx] = {sign, exp_f, mant_f};
        end
    endtask

task fill_random_lanes(input allow_zero);
        integer i;
        begin
            for (i = 0; i < %(N)d; i = i + 1) begin
                act_lane[i] = rand_fp8(allow_zero);
                wt_lane[i]  = rand_fp8(allow_zero);
            end
        end
    endtask

task pack_vectors;
        integer i;
        begin
            for (i = 0; i < %(N)d; i = i + 1) begin
                fp8_activations[i*8 +: 8] = act_lane[i];
                fp8_weights[i*8 +: 8]     = wt_lane[i];
            end
        end
    endtask

task compute_reference(output reg signed [20:0] ref_result,
                           output reg [8:0]         ref_exponent);
        integer i;
        reg sx, sw;
        reg [3:0] ex, ew;
        reg [3:0] mx, mw;
        reg [3:0] max_ex, max_ew;
        integer shift_x, shift_w;
        reg [3:0] smx, smw;
        
        integer prod;
        integer total_sum;
        begin
            max_ex = 4'd0;
            max_ew = 4'd0;
            for (i = 0; i < %(N)d; i = i + 1) begin
                ex = act_lane[i][6:3];
                ew = wt_lane[i][6:3];
                if (ex > max_ex) max_ex = ex;
                if (ew > max_ew) max_ew = ew;
            end

            total_sum = 0;
            for (i = 0; i < %(N)d; i = i + 1) begin
                sx = act_lane[i][7];
                ex = act_lane[i][6:3];
                mx = (ex != 4'd0) ? {1'b1, act_lane[i][2:0]} : 4'd0;

                sw = wt_lane[i][7];
                ew = wt_lane[i][6:3];
                mw = (ew != 4'd0) ? {1'b1, wt_lane[i][2:0]} : 4'd0;

                shift_x = max_ex - ex;
                shift_w = max_ew - ew;

                smx = mx >> shift_x;
                smw = mw >> shift_w;

                // Full mantissa magnitude used -- no >>1 truncation
                // Sign applied AFTER product (XOR rule), matching new int_mac_%(N)d
                prod = smx * smw;
                if (sx ^ sw)
                    total_sum = total_sum - prod;
                else
                    total_sum = total_sum + prod;
            end

            ref_result   = total_sum;
            // No >>1 compensation: same formula as FP8 MLB (14 bias + 6 mantissa scale = 20)
            ref_exponent = $signed({5'b0, max_ex}) + $signed({5'b0, max_ew}) - 9'sd20;
        end
    endtask

task apply_and_check(input [8*40-1:0] name);
        reg signed [20:0] expected_result;
        reg [8:0]         expected_exponent;
        begin
            test_num = test_num + 1;
            pack_vectors;
            compute_reference(expected_result, expected_exponent);

            // 1. Clear the Accumulators in the MAC array
            @(negedge clk);
            rst = 1'b1;
            
            // 2. Release reset and load data
            @(negedge clk);
            rst = 1'b0;
            valid_in = 1'b1;
            
            // 3. Stop loading (let reduction tree combinational logic settle)
            @(negedge clk);
            valid_in = 1'b0;
            @(negedge clk);
            // 4. Verify results immediately (Tree adder is combinational off the acc)
            if ((wide_integer_sum !== expected_result) ||
                (shared_exponent !== expected_exponent)) begin
                $display("[TEST %%0d] %%-0s : FAIL", test_num, name);
                $display("           expected : sum=%%0d  exp=%%0d", expected_result, expected_exponent);
                $display("           got      : sum=%%0d  exp=%%0d", wide_integer_sum, shared_exponent);
                fail_count = fail_count + 1;
            end
            else begin
                $display("[TEST %%0d] %%-0s : PASS  (sum=%%0d, exp=%%0d)", test_num, name, wide_integer_sum, shared_exponent);
                pass_count = pass_count + 1;
            end

            @(posedge clk);
        end
    endtask
    integer t;
    integer i;
    initial begin
        rst      = 1'b1;
        valid_in = 1'b0;
        fp8_activations = %(VW)d'b0;
        fp8_weights     = %(VW)d'b0;
        clear_all_lanes;

        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        // ---------------- Directed corner cases ----------------
        clear_all_lanes;
        apply_and_check("All zeros");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b0, 4'd7, 3'b000);
        end
        apply_and_check("All positive equal value");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b1, 4'd7, 3'b000);
        end
        apply_and_check("All positive act, all negative weight");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b1, 4'd7, 3'b011);
            set_lane(1, i, 1'b1, 4'd7, 3'b011);
        end
        apply_and_check("Both negative (signs cancel)");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, i[0], 4'd9, 3'b010);
            set_lane(1, i, 1'b0,  4'd9, 3'b101);
        end
        apply_and_check("Alternating activation sign");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd15, 3'b111);
            set_lane(1, i, 1'b0, 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, all positive");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, i[0],  4'd15, 3'b111);
            set_lane(1, i, ~i[0], 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, alternating signs");

        for (i = 0; i < %(N)d; i = i + 1) begin
            if (i[0]) begin
                set_lane(0, i, 1'b0, 4'd0, 3'b101);
                set_lane(1, i, 1'b0, 4'd0, 3'b011);
            end else begin
                set_lane(0, i, 1'b0, 4'd8, 3'b001);
                set_lane(1, i, 1'b0, 4'd8, 3'b001);
            end
        end
        apply_and_check("Subnormal flush (half lanes exp=0)");

        clear_all_lanes;
        set_lane(0, 0, 1'b0, 4'd10, 3'b110);
        set_lane(1, 0, 1'b1, 4'd10, 3'b110);
        apply_and_check("Single nonzero lane");

        clear_all_lanes;
        set_lane(0, 0, 1'b0, 4'd15, 3'b111);
        set_lane(1, 0, 1'b0, 4'd15, 3'b111);
        for (i = 1; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd1, 3'b000);
            set_lane(1, i, 1'b0, 4'd1, 3'b000);
        end
        apply_and_check("Wide dynamic range (shift-to-zero stress)");

        for (i = 0; i < %(N)d; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd0, 3'b111);
            set_lane(1, i, 1'b0, 4'd12, 3'b101);
        end
        apply_and_check("All activations flushed to zero");

        // ---------------- Randomized cases ----------------
        for (t = 0; t < 50; t = t + 1) begin
            if (t %% 3 == 0)
                fill_random_lanes(1'b1);  // allow zero-exponent lanes
            else
                fill_random_lanes(1'b0);  // normal numbers only
            apply_and_check("Random test");
        end
        $display("\\n======================================================");
        $display("  TEST SUMMARY: %%0d PASSED,  %%0d FAILED",
                  pass_count, fail_count);
        $display("======================================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED");
        else
            $display("  *** FAILURES DETECTED -- check waveform in int_mac_sim.vcd ***");
        $display("");

        $finish;
    end

endmodule
"""
    return code % dict(N=N, N_1=N-1, VW=VW, VW_1=VW-1)


# =============================================================================
# Main
# =============================================================================
def create_folder(N):

    folder     = os.path.join(BASE, "M=4, N=%d(and implementation)" % N)
    int_folder = os.path.join(folder, "INT MAC")
    os.makedirs(folder,     exist_ok=True)
    os.makedirs(int_folder, exist_ok=True)
    print("  N=%d: writing files..." % N)

    with open(os.path.join(folder, "bpf_aligner.v"),        "w") as f: f.write(gen_bfp_aligner(N))
    with open(os.path.join(folder, "and_popcount_4_bit.v"), "w") as f: f.write(gen_and_popcount(N))
    with open(os.path.join(folder, "MLB_MAC_unit.v"),        "w") as f: f.write(gen_mlb_unit(N))
    with open(os.path.join(folder, "MLB_MAC_4_bit.v"),       "w") as f: f.write(gen_mlb_4(N))
    with open(os.path.join(folder, "FP8.v"),                 "w") as f: f.write(gen_fp8_top(N))
    with open(os.path.join(folder, "tb_FP8.v"),              "w") as f: f.write(gen_tb_mlb(N))

    with open(os.path.join(int_folder, "bpf_aligner.v"),    "w") as f: f.write(gen_bfp_aligner(N))
    with open(os.path.join(int_folder, "int_mac_%d.v" % N), "w") as f: f.write(gen_int_mac_n(N))
    with open(os.path.join(int_folder, "fp8_int_top.v"),    "w") as f: f.write(gen_fp8_int_top(N))
    with open(os.path.join(int_folder, "tb_FP8.v"),         "w") as f: f.write(gen_tb_int(N))

    print("  N=%d: done." % N)


if __name__ == "__main__":
    print("Generating M=4 FP8 folders for N in", N_VALUES)
    for N in N_VALUES:
        create_folder(N)
    print("\nAll done!")
