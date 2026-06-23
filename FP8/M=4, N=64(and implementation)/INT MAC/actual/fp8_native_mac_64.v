`timescale 1ns/1ps

module FP8_NATIVE_MAC_64 (
    input clk,
    input rst,
    input valid_in,
    input [511:0] fp8_act,      // 64 parallel FP8 E4M3 activations
    input [511:0] fp8_wgt,      // 64 parallel FP8 E4M3 weights
    output reg done,
    output reg signed [42:0] final_acc // 43-bit exact sum (implicit scale 2^-18)
);

    // ====================================================================
    // STAGE 1: 64 Native FP8 Multipliers -> 37-bit Fixed-Point
    // ====================================================================
    reg signed [36:0] shifted_prod [0:63];
    reg stage1_valid;

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : fp8_mult_lane
            // Wires for extraction
            wire sign_a = fp8_act[i*8 + 7];
            wire [3:0] exp_a = fp8_act[i*8 + 3 +: 4];
            wire [2:0] man_a = fp8_act[i*8 +: 3];

            wire sign_w = fp8_wgt[i*8 + 7];
            wire [3:0] exp_w = fp8_wgt[i*8 + 3 +: 4];
            wire [2:0] man_w = fp8_wgt[i*8 +: 3];

            // 1. Add hidden bit for normal numbers (0 for subnormals)
            wire [3:0] true_man_a = (exp_a == 4'd0) ? {1'b0, man_a} : {1'b1, man_a};
            wire [3:0] true_man_w = (exp_w == 4'd0) ? {1'b0, man_w} : {1'b1, man_w};

            // 2. Adjust Exponents for Subnormals (If 0, act as 1 for math purposes)
            wire [3:0] true_exp_a = (exp_a == 4'd0) ? 4'd1 : exp_a;
            wire [3:0] true_exp_w = (exp_w == 4'd0) ? 4'd1 : exp_w;

            // 3. Multiplication Logic
            wire sign_out = sign_a ^ sign_w;
            wire [7:0] prod_mant = true_man_a * true_man_w; // 4b x 4b = 8b
            wire [4:0] exp_sum = true_exp_a + true_exp_w;   // Range: 2 to 30

            // 4. Shift into 37-bit Integer (LSB represents 2^-18)
            // Mathematical shift amount is exactly (exp_sum - 2)
            wire [4:0] shift_amt = exp_sum - 5'd2;
            wire [35:0] aligned_mag = {28'd0, prod_mant} << shift_amt;

            // 5. Apply Sign and Register
            always @(posedge clk) begin
                if (rst) begin
                    shifted_prod[i] <= 37'sd0;
                end else if (valid_in) begin
                    if (sign_out)
                        shifted_prod[i] <= -$signed({1'b0, aligned_mag});
                    else
                        shifted_prod[i] <=  $signed({1'b0, aligned_mag});
                end
            end
        end
    endgenerate

    // Stage 1 Control
    always @(posedge clk) begin
        if (rst) stage1_valid <= 1'b0;
        else     stage1_valid <= valid_in;
    end

    // ====================================================================
    // STAGE 2: 64-to-1 Integer Reduction Tree
    // ====================================================================
    // Level widths: 38, 39, 40, 41, 42, 43 (6 levels)
    wire signed [37:0] sum_L1 [0:31];
    wire signed [38:0] sum_L2 [0:15];
    wire signed [39:0] sum_L3 [0:7];
    wire signed [40:0] sum_L4 [0:3];
    wire signed [41:0] sum_L5 [0:1];
    wire signed [42:0] sum_L6;

    genvar j;
    generate
        for (j=0; j<32; j=j+1) assign sum_L1[j] = $signed(shifted_prod[2*j]) + $signed(shifted_prod[2*j+1]);
        for (j=0; j<16; j=j+1) assign sum_L2[j] = $signed(sum_L1[2*j]) + $signed(sum_L1[2*j+1]);
        for (j=0; j<8;  j=j+1) assign sum_L3[j] = $signed(sum_L2[2*j]) + $signed(sum_L2[2*j+1]);
        for (j=0; j<4;  j=j+1) assign sum_L4[j] = $signed(sum_L3[2*j]) + $signed(sum_L3[2*j+1]);
        for (j=0; j<2;  j=j+1) assign sum_L5[j] = $signed(sum_L4[2*j]) + $signed(sum_L4[2*j+1]);
    endgenerate

    assign sum_L6 = $signed(sum_L5[0]) + $signed(sum_L5[1]);

    // ====================================================================
    // STAGE 3: Final Output Register
    // ====================================================================
    always @(posedge clk) begin
        if (rst) begin
            final_acc <= 43'sd0;
            done      <= 1'b0;
        end else begin
            done <= stage1_valid;
            if (stage1_valid) begin
                final_acc <= sum_L6;
            end
        end
    end

endmodule