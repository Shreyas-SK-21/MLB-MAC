// ============================================================
// Full Integer MAC -- M=4, N=K=512  (Sign-After-Multiply version)
//
// Key difference from original:
//   Mantissas are received as 4-bit UNSIGNED [0..15].
//   A separate sign_flat bus carries one XOR-sign bit per lane.
//   Multiplication is done unsigned, sign is applied AFTER
//   the product -- exactly like the FP8 MLB pos/neg mask approach.
//   This eliminates the >>1 truncation that cost 1 bit of precision.
//
// Block 1 : 512 parallel unsigned MAC lanes
//           product_i = a_i * b_i          (4bx4b -> 8b unsigned)
//           signed_product_i = sign_i ? -product_i : +product_i  (9b signed)
//           acc_i += signed_product_i       (11b signed, handles M=4)
//
// Block 2 : Reduction tree 512 -> 1         (19-bit signed result)
//
// Block 3 : Scaling + offset
//           result = ax*aw*s_final + bxw
//
// Ports
//   clk      -- clock
//   rst      -- synchronous reset
//   load     -- hold high for M cycles to accumulate M dot-product slices
//   a_flat   -- 512 x 4-bit UNSIGNED activations (2048-bit bus)
//   b_flat   -- 512 x 4-bit UNSIGNED weights     (2048-bit bus)
//   sign_flat-- 512 x 1-bit product sign         (sign_x XOR sign_w)
//   alpha_x  -- unsigned 4-bit scaling factor ax
//   alpha_w  -- unsigned 4-bit scaling factor aw
//   beta_xw  -- signed 8-bit offset bxw
//   result   -- 21-bit signed dot(x^q, w^q)
// ============================================================

module int_mac_512 (
    input clk,
    input rst,
    input load,

    input [2047:0] a_flat,    // 512 x 4-bit UNSIGNED activations
    input [2047:0] b_flat,    // 512 x 4-bit UNSIGNED weights
    input [511:0]  sign_flat, // 512 x 1-bit product sign (sign_x XOR sign_w)

    input [3:0]        alpha_x,
    input [3:0]        alpha_w,
    input signed [7:0] beta_xw,

    output signed [27:0] result
);

// ============================================================
// BLOCK 1 -- 512 parallel unsigned Multiply + sign + Accumulate
//
// product_i        : 4b x 4b unsigned -> 8-bit unsigned  (max 225)
// signed_product_i : +/-product_i     -> 9-bit signed    (+/-225)
// acc_i            : sum signed_products -> 11-bit signed  (M=4: max +/-900)
// ============================================================

wire [7:0]         product       [0:511]; // unsigned magnitude product
wire signed [8:0]  signed_product[0:511]; // sign applied after multiply
reg  signed [10:0] acc           [0:511]; // accumulator, 11-bit for M=4

genvar i;
generate
    for (i = 0; i < 512; i = i + 1) begin : gen_mac_lane

        // Unsigned 4-bit x 4-bit multiply (no precision loss)
        assign product[i] = a_flat[4*i +: 4] * b_flat[4*i +: 4];

        // Apply sign AFTER multiply: negative if sign_x XOR sign_w = 1
        assign signed_product[i] = sign_flat[i]
                                    ? -$signed({1'b0, product[i]})
                                    :  $signed({1'b0, product[i]});

        // Accumulate (sign-extend 9-bit -> 11-bit before adding)
        always @(posedge clk) begin
            if (rst)
                acc[i] <= 11'sd0;
            else if (load)
                acc[i] <= acc[i] + {{2{signed_product[i][8]}}, signed_product[i]};
        end

    end
endgenerate

// ============================================================
// BLOCK 2 -- Reduction tree 512 -> 1
// acc = 11-bit signed, each level adds 1 guard bit
// ============================================================

    wire signed [11:0] s1 [0:255];
    wire signed [12:0] s2 [0:127];
    wire signed [13:0] s3 [0:63];
    wire signed [14:0] s4 [0:31];
    wire signed [15:0] s5 [0:15];
    wire signed [16:0] s6 [0:7];
    wire signed [17:0] s7 [0:3];
    wire signed [18:0] s8 [0:1];

    genvar j;
    generate
        for (j = 0; j < 256; j = j + 1) begin : gen_r1
            assign s1[j] = $signed(acc[2*j]) + $signed(acc[2*j+1]);
        end
        for (j = 0; j < 128; j = j + 1) begin : gen_r2
            assign s2[j] = $signed(s1[2*j]) + $signed(s1[2*j+1]);
        end
        for (j = 0; j < 64; j = j + 1) begin : gen_r3
            assign s3[j] = $signed(s2[2*j]) + $signed(s2[2*j+1]);
        end
        for (j = 0; j < 32; j = j + 1) begin : gen_r4
            assign s4[j] = $signed(s3[2*j]) + $signed(s3[2*j+1]);
        end
        for (j = 0; j < 16; j = j + 1) begin : gen_r5
            assign s5[j] = $signed(s4[2*j]) + $signed(s4[2*j+1]);
        end
        for (j = 0; j < 8; j = j + 1) begin : gen_r6
            assign s6[j] = $signed(s5[2*j]) + $signed(s5[2*j+1]);
        end
        for (j = 0; j < 4; j = j + 1) begin : gen_r7
            assign s7[j] = $signed(s6[2*j]) + $signed(s6[2*j+1]);
        end
        for (j = 0; j < 2; j = j + 1) begin : gen_r8
            assign s8[j] = $signed(s7[2*j]) + $signed(s7[2*j+1]);
        end
    endgenerate

wire signed [19:0] s_final;
assign s_final = $signed(s8[0]) + $signed(s8[1]);

// ============================================================
// BLOCK 3 -- Scaling ax*aw x sum + offset bxw
// ax, aw  : unsigned 4-bit -> product 8-bit unsigned
// s_final : signed 19-bit
// scaled  : 28-bit (truncated to 21)
// ============================================================

wire [7:0]        alpha_prod;   // ax x aw (unsigned 8-bit)
wire signed [8:0] alpha_prod_s; // zero-extended to signed
wire signed [28:0] scaled;

assign alpha_prod   = alpha_x * alpha_w;
assign alpha_prod_s = {1'b0, alpha_prod};
assign scaled       = $signed(s_final) * $signed(alpha_prod_s);//9+20

assign result = scaled[27:0] + {{20{beta_xw[7]}}, beta_xw};

endmodule
