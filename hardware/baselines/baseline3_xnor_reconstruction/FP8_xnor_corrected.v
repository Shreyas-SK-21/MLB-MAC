// =============================================================================
// FP8_xnor_corrected.v  --  fp8_mlb_top_xnor_corrected (XNOR-Popcount, N=64, M=4)
// =============================================================================
//
// ABLATION BASELINE (Variant C: Original MLB-MAC formulation adapted to FP8)
// -------------------------------------------------------------------------
// This module implements the original TCAS-I 2025 Multi-Level Binary MAC
// architecture using bipolar XNOR-popcount with the exact mathematical
// correction terms (C1, C2, C3):
//
//   MLB4 = SUM_k sign_k_val * (2*smx[k] - 15) * (2*smw[k] - 15)
//        = 4*ref - 30*C1 - 30*C2 + 225*C3
//
//   where:
//     C1 = SUM_k sign_k_val * smx[k]
//     C2 = SUM_k sign_k_val * smw[k]
//     C3 = SUM_k sign_k_val,   sign_k_val = sk ? -1 : +1
//
//   Exact integer reconstruction (always divisible by 4):
//     ref = (MLB4 + 30*C1 + 30*C2 - 225*C3) >> 2
//
// PIPELINED 1-CYCLE EXECUTION (II = 1)
// -------------------------------------
// In the original unpipelined implementation, C1, C2, C3, and exp_sum were
// latched into single registers on valid_in. Because MLB_4 has a 4-cycle
// pipeline delay, back-to-back inputs would overwrite C1/C2/C3 before
// the MLB_4 output arrived, corrupting continuous streaming throughput.
//
// This corrected version pipelines C1, C2, C3, and exp_sum through a 4-stage
// shift register to match the MLB_4 latency. When mlb4_done fires at stage 4,
// the matched correction values are subtracted in lockstep:
//
//   Total pipeline latency : 5 clock cycles
//   Initiation Interval    : 1 clock cycle (fully pipelined streaming)
// =============================================================================

`timescale 1ns/1ps

module fp8_mlb_top_xnor_corrected (
    output wire signed [20:0] wide_integer_sum,  // BFP-aligned integer dot product
    output wire signed [8:0]  shared_exponent,   // Block exponent: max_ex + max_ew - 14
    output wire               mac_done,
    input  wire               clk,
    input  wire               rst,
    input  wire               valid_in,          // Pulse high for >= 1 cycle per block
    input  wire [511:0]       fp8_activations,   // 64 x FP8 E4M3 activations
    input  wire [511:0]       fp8_weights        // 64 x FP8 E4M3 weights
);

    // =========================================================================
    // 1. COMBINATIONAL BFP ENCODER & CORRECTION ACCUMULATOR
    // =========================================================================
    reg [3:0]   max_ex, max_ew;
    reg [3:0]   smx [0:63];      // BFP-aligned activation magnitudes
    reg [3:0]   smw [0:63];      // BFP-aligned weight magnitudes
    reg         sk  [0:63];      // product-sign per element
    reg [255:0] axi_comb;
    reg [255:0] awi_comb;

    reg signed [10:0] c1_acc, c2_acc; // C1, C2 in [-960, 960] -> 11-bit signed
    reg signed  [7:0] c3_acc;         // C3 in [-64, 64] -> 8-bit signed

    integer k, b;
    reg [3:0] ex_k, ew_k, mx_k, mw_k;
    reg [3:0] shift_x, shift_w;

    always @(*) begin
        // Pass 1: find max exponents
        max_ex = 4'd0;
        max_ew = 4'd0;
        for (k = 0; k < 64; k = k + 1) begin
            ex_k = fp8_activations[k*8 + 6 -: 4];
            ew_k = fp8_weights    [k*8 + 6 -: 4];
            if (ex_k > max_ex) max_ex = ex_k;
            if (ew_k > max_ew) max_ew = ew_k;
        end

        // Pass 2: align, sign, pack, accumulate correction terms
        c1_acc = 11'sd0;
        c2_acc = 11'sd0;
        c3_acc = 8'sd0;
        for (k = 0; k < 64; k = k + 1) begin
            // Activations: normal numbers have hidden 1; subnormals (exp==0) flushed to 0
            ex_k    = fp8_activations[k*8 + 6 -: 4];
            mx_k    = (ex_k != 4'd0) ? {1'b1, fp8_activations[k*8 + 2 -: 3]} : 4'd0;
            shift_x = max_ex - ex_k;
            smx[k]  = (shift_x > 4'd3) ? 4'd0 : (mx_k >> shift_x);

            // Weights
            ew_k    = fp8_weights[k*8 + 6 -: 4];
            mw_k    = (ew_k != 4'd0) ? {1'b1, fp8_weights[k*8 + 2 -: 3]} : 4'd0;
            shift_w = max_ew - ew_k;
            smw[k]  = (shift_w > 4'd3) ? 4'd0 : (mw_k >> shift_w);

            // Product sign (1 = negative product, 0 = positive product)
            sk[k] = fp8_activations[k*8 + 7] ^ fp8_weights[k*8 + 7];

            // Bipolar plane packing: invert bits for negative lanes
            for (b = 0; b < 4; b = b + 1) begin
                axi_comb[b*64 + k] = sk[k] ? ~smx[k][b] : smx[k][b];
                awi_comb[b*64 + k] = smw[k][b];
            end

            // Correction accumulation
            if (sk[k]) begin
                c1_acc = c1_acc - {7'b0, smx[k]};
                c2_acc = c2_acc - {7'b0, smw[k]};
                c3_acc = c3_acc - 8'sd1;
            end else begin
                c1_acc = c1_acc + {7'b0, smx[k]};
                c2_acc = c2_acc + {7'b0, smw[k]};
                c3_acc = c3_acc + 8'sd1;
            end
        end
    end

    wire signed [8:0] exp_sum_comb = $signed({5'b0, max_ex}) + $signed({5'b0, max_ew}) - 9'sd14;

    // =========================================================================
    // 2. STAGE 0 REGISTERS & 5-STAGE CORRECTION PIPELINE
    //    Total pipeline latency from valid_in to mlb4_done is exactly 5 clock cycles:
    //      Posedge 1: axi_reg / awi_reg / mlb4_valid (input stage)
    //      Posedge 2: xnor_popcount registers f_output & xp_done (Stage 1)
    //      Posedge 3: basis_multiplier registers stage1_valid    (Stage 2)
    //      Posedge 4: basis_multiplier registers basis_mult & done (Stage 3)
    //      Posedge 5: MLB_4 registers mlb & mlb4_done            (Stage 4)
    //    We pipeline C1, C2, C3, and exp_sum through 5 registers (d1..d5) so
    //    d5 matches mlb4_out and mlb4_done cycle-for-cycle under continuous streaming.
    // =========================================================================
    reg [255:0] axi_reg, awi_reg;
    reg         mlb4_valid;

    reg signed [10:0] c1_d1, c1_d2, c1_d3, c1_d4, c1_d5;
    reg signed [10:0] c2_d1, c2_d2, c2_d3, c2_d4, c2_d5;
    reg signed  [7:0] c3_d1, c3_d2, c3_d3, c3_d4, c3_d5;
    reg signed  [8:0] exp_d1, exp_d2, exp_d3, exp_d4, exp_d5;

    always @(posedge clk) begin
        if (rst) begin
            axi_reg    <= 256'b0;
            awi_reg    <= 256'b0;
            mlb4_valid <= 1'b0;

            c1_d1 <= 11'sd0; c1_d2 <= 11'sd0; c1_d3 <= 11'sd0; c1_d4 <= 11'sd0; c1_d5 <= 11'sd0;
            c2_d1 <= 11'sd0; c2_d2 <= 11'sd0; c2_d3 <= 11'sd0; c2_d4 <= 11'sd0; c2_d5 <= 11'sd0;
            c3_d1 <=  8'sd0; c3_d2 <=  8'sd0; c3_d3 <=  8'sd0; c3_d4 <=  8'sd0; c3_d5 <=  8'sd0;
            exp_d1<=  9'sd0; exp_d2<=  9'sd0; exp_d3<=  9'sd0; exp_d4<=  9'sd0; exp_d5<=  9'sd0;
        end else begin
            mlb4_valid <= valid_in;
            axi_reg    <= axi_comb;
            awi_reg    <= awi_comb;

            // Shift correction pipeline by 1 step every clock cycle
            c1_d1 <= c1_acc;
            c1_d2 <= c1_d1;
            c1_d3 <= c1_d2;
            c1_d4 <= c1_d3;
            c1_d5 <= c1_d4;

            c2_d1 <= c2_acc;
            c2_d2 <= c2_d1;
            c2_d3 <= c2_d2;
            c2_d4 <= c2_d3;
            c2_d5 <= c2_d4;

            c3_d1 <= c3_acc;
            c3_d2 <= c3_d1;
            c3_d3 <= c3_d2;
            c3_d4 <= c3_d3;
            c3_d5 <= c3_d4;

            exp_d1 <= exp_sum_comb;
            exp_d2 <= exp_d1;
            exp_d3 <= exp_d2;
            exp_d4 <= exp_d3;
            exp_d5 <= exp_d4;
        end
    end

    // =========================================================================
    // 3. XNOR MLB_4 CORE (Original Bipolar Formulation)
    // =========================================================================
    wire signed [20:0] mlb4_out;
    wire               mlb4_done;

    MLB_4 mlb4_core (
        .mlb      (mlb4_out),
        .done     (mlb4_done),
        .alpha_x  (16'h8421),
        .alpha_w  (16'h8421),
        .axi      (axi_reg),
        .awi      (awi_reg),
        .clk      (clk),
        .rst      (rst),
        .valid_in (mlb4_valid)
    );

    // =========================================================================
    // 4. BIPOLAR CORRECTION
    //    MLB4 = 4*ref - 30*C1 - 30*C2 + 225*C3
    //    ref  = (MLB4 + 30*C1 + 30*C2 - 225*C3) >> 2
    // =========================================================================
    wire signed [10:0] C1_match = c1_d5;
    wire signed [10:0] C2_match = c2_d5;
    wire signed  [7:0] C3_match = c3_d5;
    wire signed  [8:0] EXP_match= exp_d5;

    // 30*C1 = (C1<<5) - (C1<<1)
    wire signed [19:0] c1_30  = ({{9{C1_match[10]}}, C1_match} <<< 5)
                               - ({{9{C1_match[10]}}, C1_match} <<< 1);

    // 30*C2 = (C2<<5) - (C2<<1)
    wire signed [19:0] c2_30  = ({{9{C2_match[10]}}, C2_match} <<< 5)
                               - ({{9{C2_match[10]}}, C2_match} <<< 1);

    // 225*C3 = (C3<<8) - (C3<<5) + C3
    wire signed [16:0] c3_225 = ({{9{C3_match[7]}}, C3_match} <<< 8)
                               - ({{9{C3_match[7]}}, C3_match} <<< 5)
                               + {{9{C3_match[7]}}, C3_match};

    wire signed [22:0] mlb_ext    = {{2{mlb4_out[20]}}, mlb4_out};
    wire signed [22:0] c1_30_ext  = {{3{c1_30 [19]}},   c1_30};
    wire signed [22:0] c2_30_ext  = {{3{c2_30 [19]}},   c2_30};
    wire signed [22:0] c3_225_ext = {{6{c3_225[16]}},   c3_225};

    wire signed [22:0] sum_corr = mlb_ext + c1_30_ext + c2_30_ext - c3_225_ext;

    // Outputs
    assign wide_integer_sum = sum_corr[22:2];
    assign mac_done         = mlb4_done;
    assign shared_exponent  = EXP_match;

endmodule
