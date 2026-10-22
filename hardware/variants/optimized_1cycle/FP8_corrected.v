// ===========================================================================
// FP8_corrected.v  --  Optimized 1-Cycle AND-Popcount MLB-MAC (N=64, M=4)
// ===========================================================================
// Architectural Optimizations Applied (Task 1):
//   1. Exact 57-FA Carry-In Compressor Tree in and_popcount_64:
//      Uses explicit 3:2 full-adder carry-in chains (p3 -> p7 -> p15 -> p31 -> p63)
//      so Yosys/ABC maps each stage to exact 1-FA-per-bit slices (57 FAs + 6 HAs)
//      instead of synthesizing two cascaded adders for (A + B + cin).
//   2. Eliminated redundant MLB_unit dynamic barrel shifters (input [2:0] shift_amt)
//      and duplicate 16-bit register stage (shifted_out[15:0]), replacing them
//      with true zero-gate hardwired bit shifts << (i+j) as described in Sec. III.
//   3. Unsigned Equal-Weight Shift Grouping + Single 14-bit Final Subtractor:
//      Groups same-shift unsigned popcounts (G0..G6) before shifting within a
//      14-bit unsigned range [0..14400], performing a single 14-bit subtraction
//      (pos_total - neg_total) at the output.
//   4. Preserved complementary sign-masked AND-popcount cores (and_popcount_64)
//      so that 50% of all bit-plane lanes remain zero-masked (0 & w = 0) with
//      zero dynamic switching activity every clock cycle.
// ===========================================================================

`timescale 1ns/1ps

module and_popcount_64 (
    output reg [6:0] f_output,
    input      [63:0] a,
    input      [63:0] b,
    input             clk,
    input             rst,
    input             valid_in
);
    wire [63:0] x = a & b;

    wire [1:0] p3  [0:15];
    wire [2:0] p7  [0:7];
    wire [3:0] p15 [0:3];
    wire [4:0] p31 [0:1];
    wire [5:0] p63;
    wire [6:0] final_popcount;

    genvar i, k;
    generate
        // L1: 16 Full Adders (48 bits consumed: x[0..47])
        for (i = 0; i < 16; i = i + 1) begin : gen_p3
            assign p3[i][0] = x[3*i] ^ x[3*i+1] ^ x[3*i+2];
            assign p3[i][1] = (x[3*i] & x[3*i+1]) | (x[3*i+1] & x[3*i+2]) | (x[3*i] & x[3*i+2]);
        end

        // L2: 8 x 2-bit Carry-In Adders (16 FAs, consumes x[48..55])
        for (i = 0; i < 8; i = i + 1) begin : gen_p7
            wire c0 = x[48 + i];
            wire c1 = (p3[2*i][0] & p3[2*i+1][0]) | (p3[2*i][0] & c0) | (p3[2*i+1][0] & c0);
            assign p7[i][0] = p3[2*i][0] ^ p3[2*i+1][0] ^ c0;
            assign p7[i][1] = p3[2*i][1] ^ p3[2*i+1][1] ^ c1;
            assign p7[i][2] = (p3[2*i][1] & p3[2*i+1][1]) | (p3[2*i][1] & c1) | (p3[2*i+1][1] & c1);
        end

        // L3: 4 x 3-bit Carry-In Adders (12 FAs, consumes x[56..59])
        for (i = 0; i < 4; i = i + 1) begin : gen_p15
            wire [3:0] c;
            assign c[0] = x[56 + i];
            for (k = 0; k < 3; k = k + 1) begin : bit_p15
                assign p15[i][k] = p7[2*i][k] ^ p7[2*i+1][k] ^ c[k];
                assign c[k+1]    = (p7[2*i][k] & p7[2*i+1][k]) | (p7[2*i][k] & c[k]) | (p7[2*i+1][k] & c[k]);
            end
            assign p15[i][3] = c[3];
        end

        // L4: 2 x 4-bit Carry-In Adders (8 FAs, consumes x[60..61])
        for (i = 0; i < 2; i = i + 1) begin : gen_p31
            wire [4:0] c;
            assign c[0] = x[60 + i];
            for (k = 0; k < 4; k = k + 1) begin : bit_p31
                assign p31[i][k] = p15[2*i][k] ^ p15[2*i+1][k] ^ c[k];
                assign c[k+1]    = (p15[2*i][k] & p15[2*i+1][k]) | (p15[2*i][k] & c[k]) | (p15[2*i+1][k] & c[k]);
            end
            assign p31[i][4] = c[4];
        end
    endgenerate

    // L5: 1 x 5-bit Carry-In Adder (5 FAs, consumes x[62])
    wire [5:0] c63;
    assign c63[0] = x[62];
    generate
        for (k = 0; k < 5; k = k + 1) begin : bit_p63
            assign p63[k]   = p31[0][k] ^ p31[1][k] ^ c63[k];
            assign c63[k+1] = (p31[0][k] & p31[1][k]) | (p31[0][k] & c63[k]) | (p31[1][k] & c63[k]);
        end
    endgenerate
    assign p63[5] = c63[5];

    // Final: 6 Half Adders (consumes x[63])
    wire [6:0] c64;
    assign c64[0] = x[63];
    generate
        for (k = 0; k < 6; k = k + 1) begin : bit_p64
            assign final_popcount[k] = p63[k] ^ c64[k];
            assign c64[k+1]          = p63[k] & c64[k];
        end
    endgenerate
    assign final_popcount[6] = c64[6];

    always @(posedge clk) begin
        if (rst) begin
            f_output <= 7'd0;
        end else if (valid_in) begin
            f_output <= final_popcount;
        end
    end
endmodule


module MLB_4_grouped (
    output signed [20:0] mlb_sum,
    output reg           done,
    input  [255:0]       axi_pos,
    input  [255:0]       axi_neg,
    input  [255:0]       awi,
    input                clk,
    input                rst,
    input                valid_in
);
    wire [6:0] pop_pos [0:3][0:3];
    wire [6:0] pop_neg [0:3][0:3];

    genvar i, j;
    generate
        for (i = 0; i < 4; i = i + 1) begin : rows
            for (j = 0; j < 4; j = j + 1) begin : cols
                and_popcount_64 u_pos (
                    .f_output(pop_pos[i][j]),
                    .a       (axi_pos[i*64 +: 64]),
                    .b       (awi[j*64 +: 64]),
                    .clk     (clk),
                    .rst     (rst),
                    .valid_in(valid_in)
                );
                and_popcount_64 u_neg (
                    .f_output(pop_neg[i][j]),
                    .a       (axi_neg[i*64 +: 64]),
                    .b       (awi[j*64 +: 64]),
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

    // Group unsigned positive popcounts by shift weight s = i + j in {0..6}
    wire [6:0] pos_g0 = pop_pos[0][0];
    wire [7:0] pos_g1 = pop_pos[0][1] + pop_pos[1][0];
    wire [8:0] pos_g2 = (pop_pos[0][2] + pop_pos[1][1]) + pop_pos[2][0];
    wire [8:0] pos_g3 = (pop_pos[0][3] + pop_pos[1][2]) + (pop_pos[2][1] + pop_pos[3][0]);
    wire [8:0] pos_g4 = (pop_pos[1][3] + pop_pos[2][2]) + pop_pos[3][1];
    wire [7:0] pos_g5 = pop_pos[2][3] + pop_pos[3][2];
    wire [6:0] pos_g6 = pop_pos[3][3];

    // Group unsigned negative popcounts by shift weight s = i + j in {0..6}
    wire [6:0] neg_g0 = pop_neg[0][0];
    wire [7:0] neg_g1 = pop_neg[0][1] + pop_neg[1][0];
    wire [8:0] neg_g2 = (pop_neg[0][2] + pop_neg[1][1]) + pop_neg[2][0];
    wire [8:0] neg_g3 = (pop_neg[0][3] + pop_neg[1][2]) + (pop_neg[2][1] + pop_neg[3][0]);
    wire [8:0] neg_g4 = (pop_neg[1][3] + pop_neg[2][2]) + pop_neg[3][1];
    wire [7:0] neg_g5 = pop_neg[2][3] + pop_neg[3][2];
    wire [6:0] neg_g6 = pop_neg[3][3];

    // Max unsigned total is 64 * 15 * 15 = 14,400 < 2^14 (fits in 14 bits!)
    wire [13:0] pos_total = {7'b0, pos_g0}
                          + {5'b0, pos_g1, 1'b0}
                          + {3'b0, pos_g2, 2'b0}
                          + {2'b0, pos_g3, 3'b0}
                          + {1'b0, pos_g4, 4'b0}
                          + {1'b0, pos_g5, 5'b0}
                          + {1'b0, pos_g6, 6'b0};

    wire [13:0] neg_total = {7'b0, neg_g0}
                          + {5'b0, neg_g1, 1'b0}
                          + {3'b0, neg_g2, 2'b0}
                          + {2'b0, neg_g3, 3'b0}
                          + {1'b0, neg_g4, 4'b0}
                          + {1'b0, neg_g5, 5'b0}
                          + {1'b0, neg_g6, 6'b0};

    // Single 14-bit subtraction -> 15-bit signed result in [-14400, +14400]
    wire signed [14:0] diff_15 = $signed({1'b0, pos_total}) - $signed({1'b0, neg_total});

    reg signed [14:0] mlb_sum_15;
    always @(posedge clk) begin
        if (rst) begin
            mlb_sum_15 <= 15'sd0;
            done       <= 1'b0;
        end else begin
            done <= 1'b0;
            if (pop_valid) begin
                mlb_sum_15 <= diff_15;
                done       <= 1'b1;
            end
        end
    end

    // Zero-gate wire sign-extension to 21-bit output bus
    assign mlb_sum = {{6{mlb_sum_15[14]}}, mlb_sum_15};

endmodule


module fp8_mlb_top_corrected (
    output signed [20:0] wide_integer_sum,
    output signed [8:0]  shared_exponent,
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,
    input [511:0]        fp8_activations,
    input [511:0]        fp8_weights
);

    wire [255:0] axi_planes;
    wire [255:0] awi_planes;
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
    wire [63:0] sign_k;
    generate
        for (i = 0; i < 64; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [63:0] sign_k_d1;
    reg        valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= 64'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    wire [63:0] pos_mask = ~sign_k_d1;
    wire [63:0] neg_mask =  sign_k_d1;

    wire [255:0] axi_planes_pos;
    wire [255:0] axi_planes_neg;

    assign axi_planes_pos[63:0]    = axi_planes[63:0]    & pos_mask;
    assign axi_planes_pos[127:64]  = axi_planes[127:64]  & pos_mask;
    assign axi_planes_pos[191:128] = axi_planes[191:128] & pos_mask;
    assign axi_planes_pos[255:192] = axi_planes[255:192] & pos_mask;

    assign axi_planes_neg[63:0]    = axi_planes[63:0]    & neg_mask;
    assign axi_planes_neg[127:64]  = axi_planes[127:64]  & neg_mask;
    assign axi_planes_neg[191:128] = axi_planes[191:128] & neg_mask;
    assign axi_planes_neg[255:192] = axi_planes[255:192] & neg_mask;

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

    wire signed [8:0] shared_exponent_comb = $signed({5'b0, max_exp_x})
                                           + $signed({5'b0, max_exp_w})
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
