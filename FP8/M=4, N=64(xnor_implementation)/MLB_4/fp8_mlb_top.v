`timescale 1ns/1ps
// =============================================================================
// fp8_mlb_top.v  --  FP8 E4M3 -> BFP -> MLB_4 Wrapper
// =============================================================================
//
// FP8 E4M3 format  [7]=sign  [6:3]=exp(bias=7)  [2:0]=mantissa_frac
//   Normal   : (-1)^s * 2^(exp-7) * 1.mant  (exp in 1..15)
//   Subnormal: (exp==0) flushed to zero
//
// BFP Encoding (one shared exponent per 64-element vector):
//   max_exp = MAX(exp[k]) for k=0..63
//   mx[k]   = (exp[k]!=0) ? {1'b1, mant[k]} : 4'd0   (4-bit w/ implicit 1)
//   smx[k]  = mx[k] >> (max_exp - exp[k])             (BFP-aligned magnitude)
//   sk[k]   = sx[k] ^ sw[k]                           (product-sign)
//
// Binary-plane packing for MLB_4 (M=4 planes, N=64 bits):
//   sk=0 (positive pair): axi[b*64+k] = smx[k][b]   (natural)
//   sk=1 (negative pair): axi[b*64+k] = ~smx[k][b]  (inverted -> negates bipolar)
//   Weight planes always natural: awi[b*64+k] = smw[k][b]
//   alpha_x = alpha_w = {4'd8, 4'd4, 4'd2, 4'd1}  (fixed binary-plane weights)
//
// MLB_4 computes (bipolar inner product):
//   MLB4 = SUM_k  sign_k_val * (2*smx[k]-15) * (2*smw[k]-15)
//        = 4*ref  -  30*C1  -  30*C2  +  225*C3
//   C1 = SUM_k sign_k_val*smx[k],  C2 = SUM_k sign_k_val*smw[k]
//   C3 = SUM_k sign_k_val,  sign_k_val = sk?-1:+1
//
// Correction (exact integer, always divisible by 4):
//   ref_result = (MLB4 + 30*C1 + 30*C2 - 225*C3) >> 2
//
// Pipeline: 4 posedge cycles from valid_in=1 to mac_done=1
//   Cycle 0: BFP encode (comb), register planes/correction/exp, assert mlb4_valid
//   Cycle 1: xnor_popcount fires (MLB_4 internal)
//   Cycle 2: basis_multiplier stage-1 fires
//   Cycle 3: &unit_done -> MLB_4 latches mlb, done=1
//   mac_done = MLB_4.done; wide_integer_sum = combinational correction
// =============================================================================

module fp8_mlb_top (
    input  wire         clk,
    input  wire         rst,
    input  wire         valid_in,
    input  wire [511:0] fp8_activations,      // 64 x FP8-E4M3, byte k at [k*8+:8]
    input  wire [511:0] fp8_weights,
    output wire signed [20:0] wide_integer_sum,  // BFP-aligned integer dot product
    output wire [8:0]   shared_exponent,         // max_ex + max_ew - 14
    output wire         mac_done
);

    // =========================================================================
    // 1. COMBINATIONAL BFP ENCODER
    //    All datapath variables use minimum required widths (not 32-bit integer)
    //    to reduce synthesized combinational area.
    // =========================================================================
    reg [3:0]   max_ex, max_ew;
    reg [3:0]   smx [0:63];      // BFP-aligned activation magnitudes
    reg [3:0]   smw [0:63];      // BFP-aligned weight magnitudes
    reg         sk  [0:63];      // product-sign per element
    reg [255:0] axi_comb;
    reg [255:0] awi_comb;

    // Narrow-width accumulators: C1,C2 in [-960,960] -> 11-bit signed
    //                            C3    in [-64, 64]  ->  8-bit signed
    reg signed [10:0] c1_acc, c2_acc;
    reg signed  [7:0] c3_acc;

    integer k, b;   // loop indices only — unrolled by synthesis, no datapath width
    reg [3:0] ex_k, ew_k, mx_k, mw_k;
    reg [3:0] shift_x, shift_w;   // 4-bit: max shift = 15, compare > 3 catches > 3

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

        // Pass 2: align, sign, pack, accumulate correction
        c1_acc = 11'sd0; c2_acc = 11'sd0; c3_acc = 8'sd0;
        for (k = 0; k < 64; k = k + 1) begin
            // Activations
            ex_k    = fp8_activations[k*8 + 6 -: 4];
            mx_k    = (ex_k != 4'd0) ? {1'b1, fp8_activations[k*8 + 2 -: 3]} : 4'd0;
            shift_x = max_ex - ex_k;                          // 4-bit subtraction
            smx[k]  = (shift_x > 4'd3) ? 4'd0 : (mx_k >> shift_x);

            // Weights
            ew_k    = fp8_weights[k*8 + 6 -: 4];
            mw_k    = (ew_k != 4'd0) ? {1'b1, fp8_weights[k*8 + 2 -: 3]} : 4'd0;
            shift_w = max_ew - ew_k;
            smw[k]  = (shift_w > 4'd3) ? 4'd0 : (mw_k >> shift_w);

            // Product-sign
            sk[k] = fp8_activations[k*8 + 7] ^ fp8_weights[k*8 + 7];

            // Binary plane packing
            for (b = 0; b < 4; b = b + 1) begin
                axi_comb[b*64 + k] = sk[k] ? ~smx[k][b] : smx[k][b];
                awi_comb[b*64 + k] = smw[k][b];
            end

            // Correction terms with narrow accumulators
            if (sk[k]) begin
                c1_acc = c1_acc - {7'b0, smx[k]};  // C1 in [-960,960]
                c2_acc = c2_acc - {7'b0, smw[k]};
                c3_acc = c3_acc - 8'sd1;            // C3 in [-64, 64]
            end else begin
                c1_acc = c1_acc + {7'b0, smx[k]};
                c2_acc = c2_acc + {7'b0, smw[k]};
                c3_acc = c3_acc + 8'sd1;
            end
        end
    end

    wire [8:0] exp_sum_comb;
    assign exp_sum_comb = ({5'b0, max_ex} + {5'b0, max_ew}) - 9'd14;

    // =========================================================================
    // 2. REGISTER ENCODER OUTPUTS ON valid_in
    //    Kept registered for future K-scalability (pipelining across more stages)
    // =========================================================================
    reg [255:0]       axi_reg, awi_reg;
    reg signed [10:0] C1_reg, C2_reg;
    reg signed  [7:0] C3_reg;
    reg         [8:0] exp_sum_reg;
    reg               mlb4_valid;

    always @(posedge clk) begin
        mlb4_valid <= 1'b0;
        if (rst) begin
            axi_reg     <= 256'b0;
            awi_reg     <= 256'b0;
            C1_reg      <= 11'sd0;
            C2_reg      <= 11'sd0;
            C3_reg      <=  8'sd0;
            exp_sum_reg <=  9'b0;
        end else if (valid_in) begin
            axi_reg     <= axi_comb;
            awi_reg     <= awi_comb;
            C1_reg      <= c1_acc;
            C2_reg      <= c2_acc;
            C3_reg      <= c3_acc;
            exp_sum_reg <= exp_sum_comb;
            mlb4_valid  <= 1'b1;
        end
    end

    // =========================================================================
    // 3. MLB_4 CORE
    //    16'h8421 = {4'd8, 4'd4, 4'd2, 4'd1} for planes [3:0]
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
    // 4. BIPOLAR CORRECTION  (shift-add replaces general multipliers)
    //    MLB4 = 4*ref - 30*C1 - 30*C2 + 225*C3
    //    ref  = (MLB4 + 30*C1 + 30*C2 - 225*C3) >> 2   (always exact)
    //
    //    30  = 32 - 2    =>  x*30  = (x<<5) - (x<<1)
    //    225 = 256-32+1  =>  x*225 = (x<<8) - (x<<5) + x
    //
    //    Bit widths:
    //      c1_30  : signed [19:0], |max|= 30*960 = 28800 < 2^15
    //      c2_30  : signed [19:0], |max|= 28800
    //      c3_225 : signed [16:0], |max|= 225*64 = 14400 < 2^14
    //      sum_corr: signed [22:0], |max| < 2^17
    // =========================================================================

    // 30*C1 = (C1<<5) - (C1<<1)  — no multiplier cell needed
    wire signed [19:0] c1_30  = ({{9{C1_reg[10]}}, C1_reg} <<< 5)
                               - ({{9{C1_reg[10]}}, C1_reg} <<< 1);

    // 30*C2 = (C2<<5) - (C2<<1)
    wire signed [19:0] c2_30  = ({{9{C2_reg[10]}}, C2_reg} <<< 5)
                               - ({{9{C2_reg[10]}}, C2_reg} <<< 1);

    // 225*C3 = (C3<<8) - (C3<<5) + C3
    wire signed [16:0] c3_225 = ({{9{C3_reg[7]}}, C3_reg} <<< 8)
                               - ({{9{C3_reg[7]}}, C3_reg} <<< 5)
                               + {{9{C3_reg[7]}}, C3_reg};

    wire signed [22:0] mlb_ext    = {{2{mlb4_out[20]}}, mlb4_out};
    wire signed [22:0] c1_30_ext  = {{3{c1_30 [19]}},   c1_30};
    wire signed [22:0] c2_30_ext  = {{3{c2_30 [19]}},   c2_30};
    wire signed [22:0] c3_225_ext = {{6{c3_225[16]}},   c3_225};

    wire signed [22:0] sum_corr   = mlb_ext + c1_30_ext + c2_30_ext - c3_225_ext;

    // Arithmetic right-shift by 2: bits [22:2] of the 23-bit signed sum
    assign wide_integer_sum = sum_corr[22:2];

    // =========================================================================
    // 5. OUTPUTS
    // =========================================================================
    assign mac_done        = mlb4_done;
    assign shared_exponent = exp_sum_reg;

endmodule
