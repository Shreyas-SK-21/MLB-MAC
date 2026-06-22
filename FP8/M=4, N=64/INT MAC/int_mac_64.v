// ============================================================
// Full Integer MAC — M=4, N=K=64
//
// Matches Fig. 4 exactly:
// Block 1 : 64 parallel MAC lanes, each with:
// - Multiplier : xd_i × wd_i (4b × 4b → 8b signed)
// - Adder : product + acc_prev
// - Acc register: holds running partial sum
// After M=4 load cycles:
// acc_i = Σ_{k=0}^{3} xd_i[k] · wd_i[k]
//
// Block 2 : Reduction tree (Tree Adder) 64 → 1 (14-bit sum)
// 6 levels: 64→32→16→8→4→2→1
//
// Block 3 : Scaling + offset
// dot(x^q, w^q) = αx · αw · Σ(xd·wd) + βxw
//
// Ports
// clk — clock
// rst — synchronous reset (clears all accumulators)
// load — hold high for exactly 4 cycles to feed one dot-product
// a_flat — 64 × 4-bit signed activations (256-bit bus)
// b_flat — 64 × 4-bit signed weights (256-bit bus)
// alpha_x — unsigned 4-bit scaling factor αx
// alpha_w — unsigned 4-bit scaling factor αw
// beta_xw — signed 8-bit offset βxw
// result — 21-bit signed dot(x^q, w^q)
// ============================================================

module int_mac_64 (
input clk,
input rst,
input load, // pulse high for 4 cycles

input signed [255:0] a_flat, // 64 × 4-bit signed activations
input signed [255:0] b_flat, // 64 × 4-bit signed weights

input [3:0] alpha_x, // unsigned 4-bit αx
input [3:0] alpha_w, // unsigned 4-bit αw
input signed [7:0] beta_xw, // signed 8-bit offset βxw

output signed [20:0] result // 21-bit signed dot(x^q, w^q)
);

// ============================================================
// BLOCK 1 — 64 parallel Multiply-Accumulate lanes
//
// product_i = xd_i × wd_i (4b × 4b = 8-bit signed)
// acc_i += product_i (feedback accumulator)
//
// Accumulator width:
// max |product| = 8 × 8 = 64 (needs 7 bits + sign = 8 bits)
// after 4 accumulations: max |acc| = 4 × 64 = 256 → 9 bits
// ============================================================

wire signed [7:0] product [0:63]; // 4b × 4b → 8-bit signed product
reg signed [8:0] acc [0:63]; // 9-bit accumulator per lane

genvar i;
generate
for (i = 0; i < 64; i = i + 1) begin : gen_mac_lane

// ── Multiplier ─────────────────────────────────────────
assign product[i] = $signed(a_flat[4*i +: 4]) // xd_i
* $signed(b_flat[4*i +: 4]); // wd_i

// ── Adder + Acc (feedback loop, Fig.4 Block 1) ────────
always @(posedge clk) begin
if (rst)
acc[i] <= 9'sd0;
else if (load)
acc[i] <= acc[i] + {{1{product[i][7]}}, product[i]};
// sign-extend 8b → 9b then accumulate
end

end
endgenerate

// ============================================================
// BLOCK 2 — Reduction tree (Tree Adder) 64 → 1
// acc = 9-bit, each level adds 1 guard bit
// Level widths: 9, 10, 11, 12, 13, 14 (6 levels)
// ============================================================

wire signed [9:0] s1 [0:31];
wire signed [10:0] s2 [0:15];
wire signed [11:0] s3 [0:7];
wire signed [12:0] s4 [0:3];
wire signed [13:0] s5 [0:1];
wire signed [14:0] s6; // 15-bit final sum

genvar j;
generate
for (j = 0; j < 32; j = j + 1) begin : gen_r1
assign s1[j] = $signed(acc[2*j]) + $signed(acc[2*j+1]);
end
for (j = 0; j < 16; j = j + 1) begin : gen_r2
assign s2[j] = $signed(s1[2*j]) + $signed(s1[2*j+1]);
end
for (j = 0; j < 8; j = j + 1) begin : gen_r3
assign s3[j] = $signed(s2[2*j]) + $signed(s2[2*j+1]);
end
for (j = 0; j < 4; j = j + 1) begin : gen_r4
assign s4[j] = $signed(s3[2*j]) + $signed(s3[2*j+1]);
end
for (j = 0; j < 2; j = j + 1) begin : gen_r5
assign s5[j] = $signed(s4[2*j]) + $signed(s4[2*j+1]);
end
endgenerate

assign s6 = $signed(s5[0]) + $signed(s5[1]);

// ============================================================
// BLOCK 3 — Scaling αx·αw × sum + offset βxw
// αx, αw : unsigned 4-bit → product unsigned 8-bit
// s6 : signed 15-bit
// scaled : 15 + 8 = 23-bit (use 24 for safety), truncate to 21
// result : scaled[20:0] + sign-extended beta_xw
// ============================================================

wire [7:0] alpha_prod; // αx × αw (unsigned 8-bit)
wire signed [8:0] alpha_prod_s; // zero-extended to signed
wire signed [23:0] scaled; // 15b × 9b = 24-bit product

assign alpha_prod = alpha_x * alpha_w;
assign alpha_prod_s = {1'b0, alpha_prod}; // non-negative signed

assign scaled = $signed(s6) * $signed(alpha_prod_s);

// Add βxw (sign-extend 8b → 21b) to lower 21 bits of scaled
assign result = scaled[20:0] + {{13{beta_xw[7]}}, beta_xw};

endmodule 
