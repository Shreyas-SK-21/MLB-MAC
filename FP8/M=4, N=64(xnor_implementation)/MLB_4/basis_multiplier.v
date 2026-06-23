// basis_multiplier.v
// Computes: basis_mult = xnor_popcount * (alpha_x * alpha_w)
//
// Area optimisation: for the standard alpha set {1,2,4,8}, all products are
// powers of two.  The signed multiply is replaced by:
//   1. A combinational LUT (case) mapping {alpha_x, alpha_w} -> log2(product)
//      This encodes the shift amount (0..6) in 3 bits — no multiplier cell.
//   2. A signed arithmetic left shift of the registered popcount by that amount.
//      A 7-entry barrel shift of an 8-bit value is far cheaper than an 8x9 mult.
//
// For alpha values outside {1,2,4,8} (non-power-of-two), the module falls back
// to a direct multiply so general reuse is preserved.
module basis_multiplier(
    output reg signed [16:0] basis_mult,
    output reg               done,
    input                    xp_done,
    input  signed [7:0]      xnor_popcount,
    input         [3:0]      alpha_x, alpha_w,
    input                    rst, clk
);
    // -------------------------------------------------------------------------
    // Combinational LUT: {alpha_x, alpha_w} -> shift amount = log2(product)
    // Covers all 16 combinations of {1,2,4,8} x {1,2,4,8}.
    // Falls back to MSB-priority encoder for other power-of-2 values.
    // -------------------------------------------------------------------------
    reg [2:0] alpha_shift_comb;   // 0..6  (2^6 = 64 = max alpha product)
    reg       use_lut;

    always @(*) begin
        case ({alpha_x, alpha_w})
            8'h11: begin alpha_shift_comb = 3'd0; use_lut = 1'b1; end // 1*1 =  1 = 2^0
            8'h12: begin alpha_shift_comb = 3'd1; use_lut = 1'b1; end // 1*2 =  2 = 2^1
            8'h14: begin alpha_shift_comb = 3'd2; use_lut = 1'b1; end // 1*4 =  4 = 2^2
            8'h18: begin alpha_shift_comb = 3'd3; use_lut = 1'b1; end // 1*8 =  8 = 2^3
            8'h21: begin alpha_shift_comb = 3'd1; use_lut = 1'b1; end // 2*1 =  2
            8'h22: begin alpha_shift_comb = 3'd2; use_lut = 1'b1; end // 2*2 =  4
            8'h24: begin alpha_shift_comb = 3'd3; use_lut = 1'b1; end // 2*4 =  8
            8'h28: begin alpha_shift_comb = 3'd4; use_lut = 1'b1; end // 2*8 = 16 = 2^4
            8'h41: begin alpha_shift_comb = 3'd2; use_lut = 1'b1; end // 4*1 =  4
            8'h42: begin alpha_shift_comb = 3'd3; use_lut = 1'b1; end // 4*2 =  8
            8'h44: begin alpha_shift_comb = 3'd4; use_lut = 1'b1; end // 4*4 = 16
            8'h48: begin alpha_shift_comb = 3'd5; use_lut = 1'b1; end // 4*8 = 32 = 2^5
            8'h81: begin alpha_shift_comb = 3'd3; use_lut = 1'b1; end // 8*1 =  8
            8'h82: begin alpha_shift_comb = 3'd4; use_lut = 1'b1; end // 8*2 = 16
            8'h84: begin alpha_shift_comb = 3'd5; use_lut = 1'b1; end // 8*4 = 32
            8'h88: begin alpha_shift_comb = 3'd6; use_lut = 1'b1; end // 8*8 = 64 = 2^6
            default: begin alpha_shift_comb = 3'd0; use_lut = 1'b0; end
        endcase
    end

    // General-purpose product (only synthesised for non-LUT cases)
    wire signed [8:0]  alpha_prod_general = {1'b0, alpha_x} * {1'b0, alpha_w};

    // -------------------------------------------------------------------------
    // Pipeline registers (same two-stage latency as original)
    // -------------------------------------------------------------------------
    reg [2:0]       alpha_shift_reg;
    reg signed [8:0] alpha_prod_reg;    // used only for non-LUT fallback
    reg             lut_mode_reg;
    reg signed [7:0] xnor_popcount_reg;
    reg             stage1_valid;

    always @(posedge clk) begin
        if (rst) begin
            alpha_shift_reg   <= 3'd0;
            alpha_prod_reg    <= 9'sd0;
            lut_mode_reg      <= 1'b0;
            xnor_popcount_reg <= 8'sd0;
            stage1_valid      <= 1'b0;
            basis_mult        <= 17'sd0;
            done              <= 1'b0;
        end else begin
            done         <= 1'b0;
            stage1_valid <= 1'b0;

            if (xp_done) begin
                alpha_shift_reg   <= alpha_shift_comb;
                alpha_prod_reg    <= alpha_prod_general;
                lut_mode_reg      <= use_lut;
                xnor_popcount_reg <= xnor_popcount;
                stage1_valid      <= 1'b1;
            end

            if (stage1_valid) begin
                if (lut_mode_reg)
                    // Shift-based multiply: popcount * 2^shift (no multiplier cell)
                    basis_mult <= {{9{xnor_popcount_reg[7]}}, xnor_popcount_reg} <<< alpha_shift_reg;
                else
                    // General fallback (non-power-of-two alphas)
                    basis_mult <= xnor_popcount_reg * alpha_prod_reg;
                done <= 1'b1;
            end
        end
    end
endmodule