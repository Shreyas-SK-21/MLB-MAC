// ===========================================================================
// FP8_corrected.v  --  fp8_mlb_top_corrected  (AND-Popcount, N=256, M=4)
// ===========================================================================
// Fully unrolled parallel dual MLB_4 architecture for sustained II = 1.
// Eliminates 2-cycle FSM time-multiplexing overhead.
// ===========================================================================

`timescale 1ns/1ps

module fp8_mlb_top_corrected (
    output signed [22:0] wide_integer_sum,
    output signed [8:0]  shared_exponent,
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,
    input [2047:0]        fp8_activations,
    input [2047:0]        fp8_weights
);

    // -----------------------------------------------------------------------
    // Stage 0 : BFP Alignment (1-cycle registered)
    // -----------------------------------------------------------------------
    wire [1023:0] axi_planes;
    wire [1023:0] awi_planes;
    wire [3:0]   max_exp_x;
    wire [3:0]   max_exp_w;

    bfp_aligner align_x (
        .clk          (clk),
        .fp8_vec      (fp8_activations),
        .aligned_planes(axi_planes),
        .max_exp      (max_exp_x)
    );

    bfp_aligner align_w (
        .clk          (clk),
        .fp8_vec      (fp8_weights),
        .aligned_planes(awi_planes),
        .max_exp      (max_exp_w)
    );

    // -----------------------------------------------------------------------
    // Stage 0 (parallel) : Sign XOR, registered to match aligned_planes
    // -----------------------------------------------------------------------
    genvar i;
    wire [255:0] sign_k;
    generate
        for (i = 0; i < 256; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [255:0] sign_k_d1;
    reg              valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= 256'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 1 : Sign Masking
    // -----------------------------------------------------------------------
    wire [255:0] pos_mask = ~sign_k_d1;
    wire [255:0] neg_mask =  sign_k_d1;

    wire [1023:0] axi_planes_pos;
    wire [1023:0] axi_planes_neg;

    assign axi_planes_pos[255:0] = axi_planes[255:0] & pos_mask;
    assign axi_planes_pos[511:256] = axi_planes[511:256] & pos_mask;
    assign axi_planes_pos[767:512] = axi_planes[767:512] & pos_mask;
    assign axi_planes_pos[1023:768] = axi_planes[1023:768] & pos_mask;

    assign axi_planes_neg[255:0] = axi_planes[255:0] & neg_mask;
    assign axi_planes_neg[511:256] = axi_planes[511:256] & neg_mask;
    assign axi_planes_neg[767:512] = axi_planes[767:512] & neg_mask;
    assign axi_planes_neg[1023:768] = axi_planes[1023:768] & neg_mask;

    // -----------------------------------------------------------------------
    // Stages 1-4 : TWO PARALLEL MLB_4 Arrays (II = 1)
    // -----------------------------------------------------------------------
    wire signed [21:0] pos_sum;
    wire               pos_done;

    wire signed [21:0] neg_sum;
    wire               neg_done;

    MLB_4 mac_pos (
        .mlb     (pos_sum),
        .done    (pos_done),
        .axi     (axi_planes_pos),
        .awi     (awi_planes),
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    MLB_4 mac_neg (
        .mlb     (neg_sum),
        .done    (neg_done),
        .axi     (axi_planes_neg),
        .awi     (awi_planes),
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    // -----------------------------------------------------------------------
    // Stage 5 : Output Combiner (1-cycle registered subtraction)
    // -----------------------------------------------------------------------
    reg signed [22:0] result_reg;
    reg                    done_reg;

    always @(posedge clk) begin
        if (rst) begin
            result_reg <= 23'sd0;
            done_reg   <= 1'b0;
        end else begin
            done_reg <= 1'b0;
            if (pos_done & neg_done) begin
                result_reg <= pos_sum - neg_sum;
                done_reg   <= 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Shared Exponent Pipeline (4 stages to match MLB latency)
    // -----------------------------------------------------------------------
    wire signed [8:0] shared_exponent_comb = $signed({5'b0, max_exp_x})
                                           + $signed({5'b0, max_exp_w})
                                           - 9'sd20;

    reg signed [8:0] exp_pipe_1, exp_pipe_2, exp_pipe_3, exp_pipe_4;

    always @(posedge clk) begin
        if (rst) begin
            exp_pipe_1 <= 9'sd0;
            exp_pipe_2 <= 9'sd0;
            exp_pipe_3 <= 9'sd0;
            exp_pipe_4 <= 9'sd0;
        end else begin
            exp_pipe_1 <= shared_exponent_comb;
            exp_pipe_2 <= exp_pipe_1;
            exp_pipe_3 <= exp_pipe_2;
            exp_pipe_4 <= exp_pipe_3;
        end
    end

    assign wide_integer_sum = result_reg;
    assign mac_done         = done_reg;
    assign shared_exponent  = exp_pipe_4;

endmodule
