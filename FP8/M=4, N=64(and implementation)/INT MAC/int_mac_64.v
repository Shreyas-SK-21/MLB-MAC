// ============================================================
// Full Integer MAC — M=4, N=K=64  (Sign-After-Multiply version)
//
// Key difference from original:
//   Mantissas are received as 4-bit UNSIGNED [0..15].
//   A separate sign_flat bus carries one XOR-sign bit per lane.
//   Multiplication is done unsigned, sign is applied AFTER
//   the product — exactly like the FP8 MLB pos/neg mask approach.
//   This eliminates the >>1 truncation that cost 1 bit of precision.
//
// Block 1 : 64 parallel unsigned MAC lanes
//           product_i = a_i * b_i          (4b×4b → 8b unsigned)
//           signed_product_i = sign_i ? -product_i : +product_i  (9b signed)
//           acc_i += signed_product_i       (11b signed, handles M=4)
//
// Block 2 : Reduction tree 64 → 1         (17-bit signed result)
//
// Block 3 : Scaling + offset
//           result = αx·αw·s6 + βxw
//
// Ports
//   clk      — clock
//   rst      — synchronous reset
//   load     — hold high for M cycles to accumulate M dot-product slices
//   a_flat   — 64 × 4-bit UNSIGNED activations (256-bit bus)
//   b_flat   — 64 × 4-bit UNSIGNED weights     (256-bit bus)
//   sign_flat— 64 × 1-bit product sign         (sign_x XOR sign_w)
//   alpha_x  — unsigned 4-bit scaling factor αx
//   alpha_w  — unsigned 4-bit scaling factor αw
//   beta_xw  — signed 8-bit offset βxw
//   result   — 21-bit signed dot(x^q, w^q)
// ============================================================

module int_mac_64 (
    input clk,
    input rst,
    input load,

    input [255:0] a_flat,    // 64 × 4-bit UNSIGNED activations
    input [255:0] b_flat,    // 64 × 4-bit UNSIGNED weights
    input [63:0]  sign_flat, // 64 × 1-bit product sign (sign_x XOR sign_w)

    input [3:0]        alpha_x,
    input [3:0]        alpha_w,
    input signed [7:0] beta_xw,

    output signed [24:0] result
);

// ============================================================
// BLOCK 1 — 64 parallel unsigned Multiply + sign + Accumulate
//
// product_i        : 4b × 4b unsigned → 8-bit unsigned  (max 225)
// signed_product_i : ±product_i       → 9-bit signed    (±225)
// acc_i            : Σ signed_products → 11-bit signed  (M=4: max ±900)
// ============================================================

wire [7:0]         product       [0:63]; // unsigned magnitude product
wire signed [8:0]  signed_product[0:63]; // sign applied after multiply
reg  signed [10:0] acc           [0:63]; // accumulator, 11-bit for M=4

genvar i;
generate
    for (i = 0; i < 64; i = i + 1) begin : gen_mac_lane

        // Unsigned 4-bit × 4-bit multiply (no precision loss)
        assign product[i] = a_flat[4*i +: 4] * b_flat[4*i +: 4];

        // Apply sign AFTER multiply: negative if sign_x XOR sign_w = 1
        assign signed_product[i] = sign_flat[i]
                                    ? -$signed({1'b0, product[i]})
                                    :  $signed({1'b0, product[i]});

        // Accumulate (sign-extend 9-bit → 11-bit before adding)
        always @(posedge clk) begin
            if (rst)
                acc[i] <= 11'sd0;
            else if (load)
                acc[i] <= acc[i] + {{2{signed_product[i][8]}}, signed_product[i]};
        end

    end
endgenerate

// ============================================================
// BLOCK 2 — Reduction tree 64 → 1
// acc = 11-bit signed, each level adds 1 guard bit
// Level widths: 12, 13, 14, 15, 16, 17  (6 levels)
// ============================================================

wire signed [11:0] s1 [0:31];
wire signed [12:0] s2 [0:15];
wire signed [13:0] s3 [0:7];
wire signed [14:0] s4 [0:3];
wire signed [15:0] s5 [0:1];
wire signed [16:0] s6;

genvar j;
generate
    for (j = 0; j < 32; j = j + 1) begin : gen_r1
        assign s1[j] = $signed(acc[2*j]) + $signed(acc[2*j+1]);
    end
    for (j = 0; j < 16; j = j + 1) begin : gen_r2
        assign s2[j] = $signed(s1[2*j]) + $signed(s1[2*j+1]);
    end
    for (j = 0; j < 8;  j = j + 1) begin : gen_r3
        assign s3[j] = $signed(s2[2*j]) + $signed(s2[2*j+1]);
    end
    for (j = 0; j < 4;  j = j + 1) begin : gen_r4
        assign s4[j] = $signed(s3[2*j]) + $signed(s3[2*j+1]);
    end
    for (j = 0; j < 2;  j = j + 1) begin : gen_r5
        assign s5[j] = $signed(s4[2*j]) + $signed(s4[2*j+1]);
    end
endgenerate

assign s6 = $signed(s5[0]) + $signed(s5[1]);

// ============================================================
// BLOCK 3 — Scaling αx·αw × sum + offset βxw
// αx, αw  : unsigned 4-bit → product 8-bit unsigned
// s6      : signed 17-bit
// scaled  : 17b × 9b = 26-bit (truncated to 21)
// ============================================================

wire [7:0]        alpha_prod;   // αx × αw (unsigned 8-bit)
wire signed [8:0] alpha_prod_s; // zero-extended to signed
wire signed [25:0] scaled;

assign alpha_prod   = alpha_x * alpha_w;
assign alpha_prod_s = {1'b0, alpha_prod};
assign scaled       = $signed(s6) * $signed(alpha_prod_s);//9+17=26

assign result = scaled[24:0] + {{17{beta_xw[7]}}, beta_xw};

endmodule