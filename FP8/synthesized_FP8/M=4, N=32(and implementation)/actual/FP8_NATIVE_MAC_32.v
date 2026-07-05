`timescale 1ns/1ps

module FP8_NATIVE_MAC_32 (
    input clk,
    input rst,
    input valid_in,
    input [255:0] fp8_act,      // 32 parallel FP8 E4M3 activations
    input [255:0] fp8_wgt,      // 32 parallel FP8 E4M3 weights
    output reg done,
    output reg signed [41:0] final_acc // 42-bit exact sum (implicit scale 2^-18)
);

    reg signed [36:0] shifted_prod [0:31];
    reg stage1_valid;

    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin : fp8_mult_lane
            // Wires for extraction
            wire sign_a = fp8_act[i*8 + 7];
            wire [3:0] exp_a = fp8_act[i*8 + 3 +: 4];
            wire [2:0] man_a = fp8_act[i*8 +: 3];

            wire sign_w = fp8_wgt[i*8 + 7];
            wire [3:0] exp_w = fp8_wgt[i*8 + 3 +: 4];
            wire [2:0] man_w = fp8_wgt[i*8 +: 3];
            wire [3:0] true_man_a = (exp_a == 4'd0) ? {1'b0, man_a} : {1'b1, man_a};
            wire [3:0] true_man_w = (exp_w == 4'd0) ? {1'b0, man_w} : {1'b1, man_w};
            wire [3:0] true_exp_a = (exp_a == 4'd0) ? 4'd1 : exp_a;
            wire [3:0] true_exp_w = (exp_w == 4'd0) ? 4'd1 : exp_w;

            wire sign_out = sign_a ^ sign_w;
            wire [7:0] prod_mant = true_man_a * true_man_w; // 4b x 4b = 8b
            wire [4:0] exp_sum = true_exp_a + true_exp_w;   // Range: 2 to 30

            wire [4:0] shift_amt = exp_sum - 5'd2;
            wire [35:0] aligned_mag = {28'd0, prod_mant} << shift_amt;
            
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


    // ---- Reduction level 1: 32 -> 16 (width 37 -> 38) ----
    wire signed [37:0] sum_L1 [0:15];
    genvar j_L1;
    generate
        for (j_L1=0; j_L1<16; j_L1=j_L1+1) begin : lvl1_pair
            assign sum_L1[j_L1] = $signed(shifted_prod[2*j_L1]) + $signed(shifted_prod[2*j_L1+1]);
        end
    endgenerate
    // ---- Reduction level 2: 16 -> 8 (width 38 -> 39) ----
    wire signed [38:0] sum_L2 [0:7];
    genvar j_L2;
    generate
        for (j_L2=0; j_L2<8; j_L2=j_L2+1) begin : lvl2_pair
            assign sum_L2[j_L2] = $signed(sum_L1[2*j_L2]) + $signed(sum_L1[2*j_L2+1]);
        end
    endgenerate
    // ---- Reduction level 3: 8 -> 4 (width 39 -> 40) ----
    wire signed [39:0] sum_L3 [0:3];
    genvar j_L3;
    generate
        for (j_L3=0; j_L3<4; j_L3=j_L3+1) begin : lvl3_pair
            assign sum_L3[j_L3] = $signed(sum_L2[2*j_L3]) + $signed(sum_L2[2*j_L3+1]);
        end
    endgenerate
    // ---- Reduction level 4: 4 -> 2 (width 40 -> 41) ----
    wire signed [40:0] sum_L4 [0:1];
    genvar j_L4;
    generate
        for (j_L4=0; j_L4<2; j_L4=j_L4+1) begin : lvl4_pair
            assign sum_L4[j_L4] = $signed(sum_L3[2*j_L4]) + $signed(sum_L3[2*j_L4+1]);
        end
    endgenerate
    // ---- Reduction level 5: 2 -> 1 (width 41 -> 42) ----
    wire signed [41:0] sum_L5 [0:0];
    genvar j_L5;
    generate
        for (j_L5=0; j_L5<1; j_L5=j_L5+1) begin : lvl5_pair
            assign sum_L5[j_L5] = $signed(sum_L4[2*j_L5]) + $signed(sum_L4[2*j_L5+1]);
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            final_acc <= 42'sd0;
            done      <= 1'b0;
        end else begin
            done <= stage1_valid;
            if (stage1_valid) begin
                final_acc <= sum_L5[0];
            end
        end
    end

endmodule
