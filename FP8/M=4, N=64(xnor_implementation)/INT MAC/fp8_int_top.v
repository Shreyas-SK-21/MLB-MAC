`timescale 1ns/1ps
// =============================================================================
// fp8_int_top.v  --  FP8 E4M3 -> BFP -> Signed Integer MAC  (fixed)
// =============================================================================
//
// BFP encoding is IDENTICAL to fp8_mlb_top so both designs are
// compared on the same reference dot product (no precision loss).
//
// Fixes vs previous version:
//   1. Removed the lossy >>1 right-shift; full 4-bit mantissa magnitude used.
//   2. Signed operands are 5-bit (range -15..+15) to hold full ±15 values.
//   3. Exponent formula corrected to (max_ex + max_ew - 14), same as MLB.
//   4. Added mac_done output (matches fp8_mlb_top interface).
//   5. int_mac_64 is now a 1-cycle parallel dot-product (no M=4 accumulation).
//
// Pipeline (2 posedge cycles from valid_in to mac_done):
//   Cycle 0  (valid_in=1) : BFP encode (comb), register 5-bit signed operands
//                           + exp_sum; assert mac_valid.
//   Cycle 1  (mac_valid=1): int_mac_64 computes 64-way dot product,
//                           registers result, asserts done.
//   Outputs  : mac_done = int_mac_64.done; wide_integer_sum registered inside
//              int_mac_64; shared_exponent from exp_sum_reg.
// =============================================================================

module fp8_int_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [511:0] fp8_activations,   // 64 x FP8-E4M3, byte k at [k*8+:8]
    input  wire [511:0] fp8_weights,
    output wire signed [20:0] wide_integer_sum,
    output wire [8:0]  shared_exponent,
    output wire        mac_done
);

    // =========================================================================
    // 1. COMBINATIONAL BFP ENCODER
    //    Identical algorithm to fp8_mlb_top — guarantees same reference output
    // =========================================================================
    reg [3:0] max_ex, max_ew;
    reg [3:0] smx [0:63];      // BFP-aligned unsigned magnitude (0..15)
    reg [3:0] smw [0:63];

    integer k;
    reg [3:0] ex_k, ew_k, mx_k, mw_k;
    integer   shift_x, shift_w;

    always @(*) begin
        // Pass 1: find per-vector maximum exponents
        max_ex = 4'd0;
        max_ew = 4'd0;
        for (k = 0; k < 64; k = k + 1) begin
            ex_k = fp8_activations[k*8 + 6 -: 4];
            ew_k = fp8_weights    [k*8 + 6 -: 4];
            if (ex_k > max_ex) max_ex = ex_k;
            if (ew_k > max_ew) max_ew = ew_k;
        end

        // Pass 2: align mantissas (subnormals flushed to zero)
        for (k = 0; k < 64; k = k + 1) begin
            ex_k  = fp8_activations[k*8 + 6 -: 4];
            mx_k  = (ex_k != 4'd0) ? {1'b1, fp8_activations[k*8 + 2 -: 3]} : 4'd0;
            shift_x = max_ex - ex_k;
            smx[k]  = (shift_x > 3) ? 4'd0 : (mx_k >> shift_x);

            ew_k  = fp8_weights[k*8 + 6 -: 4];
            mw_k  = (ew_k != 4'd0) ? {1'b1, fp8_weights[k*8 + 2 -: 3]} : 4'd0;
            shift_w = max_ew - ew_k;
            smw[k]  = (shift_w > 3) ? 4'd0 : (mw_k >> shift_w);
        end
    end

    wire [8:0] exp_sum_comb;
    assign exp_sum_comb = ({5'b0, max_ex} + {5'b0, max_ew}) - 9'd14;

    // =========================================================================
    // 2. SIGN-MAGNITUDE -> 5-BIT SIGNED CONVERSION
    //    signed_smx[k] = sx ? -smx[k] : +smx[k]  in range [-15, +15]
    //    Pack 64 x 5-bit signed values into 320-bit buses for int_mac_64
    // =========================================================================
    reg [319:0] a_flat_comb, b_flat_comb;
    integer     sv;           // temporary signed integer

    always @(*) begin
        for (k = 0; k < 64; k = k + 1) begin
            // Activation (5-bit signed two's complement)
            sv = fp8_activations[k*8 + 7] ? (-smx[k]) : smx[k];
            a_flat_comb[k*5 +: 5] = sv[4:0];

            // Weight
            sv = fp8_weights[k*8 + 7] ? (-smw[k]) : smw[k];
            b_flat_comb[k*5 +: 5] = sv[4:0];
        end
    end

    // =========================================================================
    // 3. REGISTER ON valid_in
    // =========================================================================
    reg [319:0] a_flat_reg, b_flat_reg;
    reg [8:0]   exp_sum_reg;
    reg         mac_valid;

    always @(posedge clk) begin
        mac_valid <= 1'b0;
        if (rst) begin
            a_flat_reg  <= 320'b0;
            b_flat_reg  <= 320'b0;
            exp_sum_reg <= 9'b0;
        end else if (valid_in) begin
            a_flat_reg  <= a_flat_comb;
            b_flat_reg  <= b_flat_comb;
            exp_sum_reg <= exp_sum_comb;
            mac_valid   <= 1'b1;
        end
    end

    // =========================================================================
    // 4. int_mac_64 INSTANTIATION
    //    Computes result = SUM_k a[k]*b[k] in one pipeline cycle
    // =========================================================================
    int_mac_64 mac (
        .clk      (clk),
        .rst      (rst),
        .valid_in (mac_valid),
        .a_flat   (a_flat_reg),
        .b_flat   (b_flat_reg),
        .result   (wide_integer_sum),
        .done     (mac_done)
    );

    // Exponent registered at cycle 0, stable when mac_done fires at cycle 1
    assign shared_exponent = exp_sum_reg;

endmodule
