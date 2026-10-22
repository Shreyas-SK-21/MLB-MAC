// ===========================================================================
// FP8_corrected.v  --  Optimized 1-Cycle AND-Popcount MLB-MAC (N=9, M=4)
// ===========================================================================
// Architectural Optimizations Applied (Task 1):
//   1. Exact (2^k - 1) Carry-In Compressor Tree in and_popcount_9.
//   2. Eliminated redundant MLB_unit dynamic barrel shifters and duplicate
//      16-bit registers; replaced with zero-gate hardwired shifts << (i+j).
//   3. Unsigned Equal-Weight Shift Grouping (G0..G6) + Single 11-bit Subtractor.
//   4. Preserved complementary sign-masked AND-popcount cores (50% zero-masked).
// ===========================================================================

`timescale 1ns/1ps

module and_popcount_9 (
    output reg [3:0] f_output,
    input      [8:0] a,
    input      [8:0] b,
    input             clk,
    input             rst,
    input             valid_in
);
    wire [8:0] x = a & b;

    wire [1:0] p3 [0:1];
    wire [2:0] p7;
    wire [1:0] p2;
    wire [3:0] final_popcount;
    genvar i;
    generate
        for (i = 0; i < 2; i = i + 1) begin : gen_p3
            assign p3[i][0] = x[3*i] ^ x[3*i+1] ^ x[3*i+2];
            assign p3[i][1] = (x[3*i] & x[3*i+1]) | (x[3*i+1] & x[3*i+2]) | (x[3*i] & x[3*i+2]);
        end
    endgenerate
    wire c0 = x[6];
    wire c1 = (p3[0][0] & p3[1][0]) | (p3[0][0] & c0) | (p3[1][0] & c0);
    assign p7[0] = p3[0][0] ^ p3[1][0] ^ c0;
    assign p7[1] = p3[0][1] ^ p3[1][1] ^ c1;
    assign p7[2] = (p3[0][1] & p3[1][1]) | (p3[0][1] & c1) | (p3[1][1] & c1);
    assign p2[0] = x[7] ^ x[8];
    assign p2[1] = x[7] & x[8];
    assign final_popcount = {1'b0, p7} + {2'b0, p2};

    always @(posedge clk) begin
        if (rst) begin
            f_output <= 4'd0;
        end else if (valid_in) begin
            f_output <= final_popcount;
        end
    end
endmodule


module MLB_4_grouped (
    output signed [16:0] mlb_sum,
    output reg           done,
    input  [35:0]       axi_pos,
    input  [35:0]       axi_neg,
    input  [35:0]       awi,
    input                clk,
    input                rst,
    input                valid_in
);
    wire [3:0] pop_pos [0:3][0:3];
    wire [3:0] pop_neg [0:3][0:3];

    genvar i, j;
    generate
        for (i = 0; i < 4; i = i + 1) begin : rows
            for (j = 0; j < 4; j = j + 1) begin : cols
                and_popcount_9 u_pos (
                    .f_output(pop_pos[i][j]),
                    .a       (axi_pos[i*9 +: 9]),
                    .b       (awi[j*9 +: 9]),
                    .clk     (clk),
                    .rst     (rst),
                    .valid_in(valid_in)
                );
                and_popcount_9 u_neg (
                    .f_output(pop_neg[i][j]),
                    .a       (axi_neg[i*9 +: 9]),
                    .b       (awi[j*9 +: 9]),
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
    wire [3:0] pos_g0 = pop_pos[0][0];
    wire [4:0] pos_g1 = pop_pos[0][1] + pop_pos[1][0];
    wire [5:0] pos_g2 = (pop_pos[0][2] + pop_pos[1][1]) + pop_pos[2][0];
    wire [5:0] pos_g3 = (pop_pos[0][3] + pop_pos[1][2]) + (pop_pos[2][1] + pop_pos[3][0]);
    wire [5:0] pos_g4 = (pop_pos[1][3] + pop_pos[2][2]) + pop_pos[3][1];
    wire [4:0] pos_g5 = pop_pos[2][3] + pop_pos[3][2];
    wire [3:0] pos_g6 = pop_pos[3][3];

    // Group unsigned negative popcounts by shift weight s = i + j in {0..6}
    wire [3:0] neg_g0 = pop_neg[0][0];
    wire [4:0] neg_g1 = pop_neg[0][1] + pop_neg[1][0];
    wire [5:0] neg_g2 = (pop_neg[0][2] + pop_neg[1][1]) + pop_neg[2][0];
    wire [5:0] neg_g3 = (pop_neg[0][3] + pop_neg[1][2]) + (pop_neg[2][1] + pop_neg[3][0]);
    wire [5:0] neg_g4 = (pop_neg[1][3] + pop_neg[2][2]) + pop_neg[3][1];
    wire [4:0] neg_g5 = pop_neg[2][3] + pop_neg[3][2];
    wire [3:0] neg_g6 = pop_neg[3][3];

    wire [10:0] pos_total = {7'b0, pos_g0}
                          + {5'b0, pos_g1, 1'b0}
                          + {3'b0, pos_g2, 2'b0}
                          + {2'b0, pos_g3, 3'b0}
                          + {1'b0, pos_g4, 4'b0}
                          + {1'b0, pos_g5, 5'b0}
                          + {1'b0, pos_g6, 6'b0};

    wire [10:0] neg_total = {7'b0, neg_g0}
                          + {5'b0, neg_g1, 1'b0}
                          + {3'b0, neg_g2, 2'b0}
                          + {2'b0, neg_g3, 3'b0}
                          + {1'b0, neg_g4, 4'b0}
                          + {1'b0, neg_g5, 5'b0}
                          + {1'b0, neg_g6, 6'b0};

    wire signed [11:0] diff_r = $signed({1'b0, pos_total}) - $signed({1'b0, neg_total});

    reg signed [11:0] mlb_sum_r;
    always @(posedge clk) begin
        if (rst) begin
            mlb_sum_r <= 12'sd0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;
            if (pop_valid) begin
                mlb_sum_r <= diff_r;
                done      <= 1'b1;
            end
        end
    end

    assign mlb_sum = {{5{mlb_sum_r[11]}}, mlb_sum_r};

endmodule


module fp8_mlb_top_corrected (
    output signed [16:0] wide_integer_sum,
    output signed [8:0]  shared_exponent,
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,
    input [71:0]        fp8_activations,
    input [71:0]        fp8_weights
);

    wire [35:0] axi_planes;
    wire [35:0] awi_planes;
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
    wire [8:0] sign_k;
    generate
        for (i = 0; i < 9; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [8:0] sign_k_d1;
    reg        valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= 9'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    wire [8:0] pos_mask = ~sign_k_d1;
    wire [8:0] neg_mask =  sign_k_d1;

    wire [35:0] axi_planes_pos;
    wire [35:0] axi_planes_neg;

    assign axi_planes_pos[8:0]         = axi_planes[8:0]         & pos_mask;
    assign axi_planes_pos[17:9]     = axi_planes[17:9]     & pos_mask;
    assign axi_planes_pos[26:18]   = axi_planes[26:18]   & pos_mask;
    assign axi_planes_pos[35:27]   = axi_planes[35:27]   & pos_mask;

    assign axi_planes_neg[8:0]         = axi_planes[8:0]         & neg_mask;
    assign axi_planes_neg[17:9]     = axi_planes[17:9]     & neg_mask;
    assign axi_planes_neg[26:18]   = axi_planes[26:18]   & neg_mask;
    assign axi_planes_neg[35:27]   = axi_planes[35:27]   & neg_mask;

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
