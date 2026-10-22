// ===========================================================================
// FP8_corrected.v  --  Optimized 1-Cycle AND-Popcount MLB-MAC (N=512, M=4)
// ===========================================================================
// Architectural Optimizations Applied (Task 1):
//   1. Exact (2^k - 1) Carry-In Compressor Tree in and_popcount_512.
//   2. Eliminated redundant MLB_unit dynamic barrel shifters and duplicate
//      16-bit registers; replaced with zero-gate hardwired shifts << (i+j).
//   3. Unsigned Equal-Weight Shift Grouping (G0..G6) + Single 17-bit Subtractor.
//   4. Preserved complementary sign-masked AND-popcount cores (50% zero-masked).
// ===========================================================================

`timescale 1ns/1ps

module and_popcount_512 (
    output reg [9:0] f_output,
    input      [511:0] a,
    input      [511:0] b,
    input             clk,
    input             rst,
    input             valid_in
);
    wire [511:0] x = a & b;

    wire [1:0] p3 [0:127];
    wire [2:0] p7 [0:63];
    wire [3:0] p15 [0:31];
    wire [4:0] p31 [0:15];
    wire [5:0] p63 [0:7];
    wire [6:0] p127 [0:3];
    wire [7:0] p255 [0:1];
    wire [8:0] p511;
    wire [9:0] final_popcount;
    genvar i, k;
    generate
        for (i = 0; i < 128; i = i + 1) begin : gen_p3
            assign p3[i][0] = x[3*i] ^ x[3*i+1] ^ x[3*i+2];
            assign p3[i][1] = (x[3*i] & x[3*i+1]) | (x[3*i+1] & x[3*i+2]) | (x[3*i] & x[3*i+2]);
        end
        for (i = 0; i < 64; i = i + 1) begin : gen_p7
            wire [2:0] c;
            assign c[0] = x[384 + i];
            for (k = 0; k < 2; k = k + 1) begin : bit_p7
                assign p7[i][k] = p3[2*i][k] ^ p3[2*i+1][k] ^ c[k];
                assign c[k+1] = (p3[2*i][k] & p3[2*i+1][k]) | (p3[2*i][k] & c[k]) | (p3[2*i+1][k] & c[k]);
            end
            assign p7[i][2] = c[2];
        end
        for (i = 0; i < 32; i = i + 1) begin : gen_p15
            wire [3:0] c;
            assign c[0] = x[448 + i];
            for (k = 0; k < 3; k = k + 1) begin : bit_p15
                assign p15[i][k] = p7[2*i][k] ^ p7[2*i+1][k] ^ c[k];
                assign c[k+1] = (p7[2*i][k] & p7[2*i+1][k]) | (p7[2*i][k] & c[k]) | (p7[2*i+1][k] & c[k]);
            end
            assign p15[i][3] = c[3];
        end
        for (i = 0; i < 16; i = i + 1) begin : gen_p31
            wire [4:0] c;
            assign c[0] = x[480 + i];
            for (k = 0; k < 4; k = k + 1) begin : bit_p31
                assign p31[i][k] = p15[2*i][k] ^ p15[2*i+1][k] ^ c[k];
                assign c[k+1] = (p15[2*i][k] & p15[2*i+1][k]) | (p15[2*i][k] & c[k]) | (p15[2*i+1][k] & c[k]);
            end
            assign p31[i][4] = c[4];
        end
        for (i = 0; i < 8; i = i + 1) begin : gen_p63
            wire [5:0] c;
            assign c[0] = x[496 + i];
            for (k = 0; k < 5; k = k + 1) begin : bit_p63
                assign p63[i][k] = p31[2*i][k] ^ p31[2*i+1][k] ^ c[k];
                assign c[k+1] = (p31[2*i][k] & p31[2*i+1][k]) | (p31[2*i][k] & c[k]) | (p31[2*i+1][k] & c[k]);
            end
            assign p63[i][5] = c[5];
        end
        for (i = 0; i < 4; i = i + 1) begin : gen_p127
            wire [6:0] c;
            assign c[0] = x[504 + i];
            for (k = 0; k < 6; k = k + 1) begin : bit_p127
                assign p127[i][k] = p63[2*i][k] ^ p63[2*i+1][k] ^ c[k];
                assign c[k+1] = (p63[2*i][k] & p63[2*i+1][k]) | (p63[2*i][k] & c[k]) | (p63[2*i+1][k] & c[k]);
            end
            assign p127[i][6] = c[6];
        end
        for (i = 0; i < 2; i = i + 1) begin : gen_p255
            wire [7:0] c;
            assign c[0] = x[508 + i];
            for (k = 0; k < 7; k = k + 1) begin : bit_p255
                assign p255[i][k] = p127[2*i][k] ^ p127[2*i+1][k] ^ c[k];
                assign c[k+1] = (p127[2*i][k] & p127[2*i+1][k]) | (p127[2*i][k] & c[k]) | (p127[2*i+1][k] & c[k]);
            end
            assign p255[i][7] = c[7];
        end
    endgenerate
    wire [8:0] c511;
    assign c511[0] = x[510];
    generate
        for (k = 0; k < 8; k = k + 1) begin : bit_p511
            assign p511[k] = p255[0][k] ^ p255[1][k] ^ c511[k];
            assign c511[k+1] = (p255[0][k] & p255[1][k]) | (p255[0][k] & c511[k]) | (p255[1][k] & c511[k]);
        end
    endgenerate
    assign p511[8] = c511[8];
    wire [9:0] c512;
    assign c512[0] = x[511];
    generate
        for (k = 0; k < 9; k = k + 1) begin : bit_p512
            assign final_popcount[k] = p511[k] ^ c512[k];
            assign c512[k+1]          = p511[k] & c512[k];
        end
    endgenerate
    assign final_popcount[9] = c512[9];

    always @(posedge clk) begin
        if (rst) begin
            f_output <= 10'd0;
        end else if (valid_in) begin
            f_output <= final_popcount;
        end
    end
endmodule


module MLB_4_grouped (
    output signed [23:0] mlb_sum,
    output reg           done,
    input  [2047:0]       axi_pos,
    input  [2047:0]       axi_neg,
    input  [2047:0]       awi,
    input                clk,
    input                rst,
    input                valid_in
);
    wire [9:0] pop_pos [0:3][0:3];
    wire [9:0] pop_neg [0:3][0:3];

    genvar i, j;
    generate
        for (i = 0; i < 4; i = i + 1) begin : rows
            for (j = 0; j < 4; j = j + 1) begin : cols
                and_popcount_512 u_pos (
                    .f_output(pop_pos[i][j]),
                    .a       (axi_pos[i*512 +: 512]),
                    .b       (awi[j*512 +: 512]),
                    .clk     (clk),
                    .rst     (rst),
                    .valid_in(valid_in)
                );
                and_popcount_512 u_neg (
                    .f_output(pop_neg[i][j]),
                    .a       (axi_neg[i*512 +: 512]),
                    .b       (awi[j*512 +: 512]),
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
    wire [9:0] pos_g0 = pop_pos[0][0];
    wire [10:0] pos_g1 = pop_pos[0][1] + pop_pos[1][0];
    wire [11:0] pos_g2 = (pop_pos[0][2] + pop_pos[1][1]) + pop_pos[2][0];
    wire [11:0] pos_g3 = (pop_pos[0][3] + pop_pos[1][2]) + (pop_pos[2][1] + pop_pos[3][0]);
    wire [11:0] pos_g4 = (pop_pos[1][3] + pop_pos[2][2]) + pop_pos[3][1];
    wire [10:0] pos_g5 = pop_pos[2][3] + pop_pos[3][2];
    wire [9:0] pos_g6 = pop_pos[3][3];

    // Group unsigned negative popcounts by shift weight s = i + j in {0..6}
    wire [9:0] neg_g0 = pop_neg[0][0];
    wire [10:0] neg_g1 = pop_neg[0][1] + pop_neg[1][0];
    wire [11:0] neg_g2 = (pop_neg[0][2] + pop_neg[1][1]) + pop_neg[2][0];
    wire [11:0] neg_g3 = (pop_neg[0][3] + pop_neg[1][2]) + (pop_neg[2][1] + pop_neg[3][0]);
    wire [11:0] neg_g4 = (pop_neg[1][3] + pop_neg[2][2]) + pop_neg[3][1];
    wire [10:0] neg_g5 = pop_neg[2][3] + pop_neg[3][2];
    wire [9:0] neg_g6 = pop_neg[3][3];

    wire [16:0] pos_total = {7'b0, pos_g0}
                          + {5'b0, pos_g1, 1'b0}
                          + {3'b0, pos_g2, 2'b0}
                          + {2'b0, pos_g3, 3'b0}
                          + {1'b0, pos_g4, 4'b0}
                          + {1'b0, pos_g5, 5'b0}
                          + {1'b0, pos_g6, 6'b0};

    wire [16:0] neg_total = {7'b0, neg_g0}
                          + {5'b0, neg_g1, 1'b0}
                          + {3'b0, neg_g2, 2'b0}
                          + {2'b0, neg_g3, 3'b0}
                          + {1'b0, neg_g4, 4'b0}
                          + {1'b0, neg_g5, 5'b0}
                          + {1'b0, neg_g6, 6'b0};

    wire signed [17:0] diff_r = $signed({1'b0, pos_total}) - $signed({1'b0, neg_total});

    reg signed [17:0] mlb_sum_r;
    always @(posedge clk) begin
        if (rst) begin
            mlb_sum_r <= 18'sd0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;
            if (pop_valid) begin
                mlb_sum_r <= diff_r;
                done      <= 1'b1;
            end
        end
    end

    assign mlb_sum = {{6{mlb_sum_r[17]}}, mlb_sum_r};

endmodule


module fp8_mlb_top_corrected (
    output signed [23:0] wide_integer_sum,
    output signed [8:0]  shared_exponent,
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,
    input [4095:0]        fp8_activations,
    input [4095:0]        fp8_weights
);

    wire [2047:0] axi_planes;
    wire [2047:0] awi_planes;
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
    wire [511:0] sign_k;
    generate
        for (i = 0; i < 512; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [511:0] sign_k_d1;
    reg        valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= 512'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    wire [511:0] pos_mask = ~sign_k_d1;
    wire [511:0] neg_mask =  sign_k_d1;

    wire [2047:0] axi_planes_pos;
    wire [2047:0] axi_planes_neg;

    assign axi_planes_pos[511:0]         = axi_planes[511:0]         & pos_mask;
    assign axi_planes_pos[1023:512]     = axi_planes[1023:512]     & pos_mask;
    assign axi_planes_pos[1535:1024]   = axi_planes[1535:1024]   & pos_mask;
    assign axi_planes_pos[2047:1536]   = axi_planes[2047:1536]   & pos_mask;

    assign axi_planes_neg[511:0]         = axi_planes[511:0]         & neg_mask;
    assign axi_planes_neg[1023:512]     = axi_planes[1023:512]     & neg_mask;
    assign axi_planes_neg[1535:1024]   = axi_planes[1535:1024]   & neg_mask;
    assign axi_planes_neg[2047:1536]   = axi_planes[2047:1536]   & neg_mask;

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
