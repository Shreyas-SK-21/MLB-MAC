module fp8_int_top (
    input  clk,
    input  rst,
    input  valid_in, // Connects to 'load' on the MAC array
    input  [511:0] fp8_activations,
    input  [511:0] fp8_weights,
    output signed [24:0] wide_integer_sum,
    output signed [8:0]   shared_exponent
);

    // ---------------- BFP alignment (Reused from MLB design) ----------------
    wire [255:0] axi_planes;
    wire [255:0] awi_planes;
    wire [3:0]   max_exp_x;
    wire [3:0]   max_exp_w;

    bfp_aligner align_x (
        .fp8_vec(fp8_activations),
        .aligned_planes(axi_planes),
        .max_exp(max_exp_x)
    );

    bfp_aligner align_w (
        .fp8_vec(fp8_weights),
        .aligned_planes(awi_planes),
        .max_exp(max_exp_w)
    );

    // ---------------- Format for INT MAC (Plane-to-Flat, Sign Separate) ----------------
    // Mantissas stay UNSIGNED [0..15]. Sign is handled AFTER multiplication inside int_mac_64,
    // exactly like the FP8 MLB design — avoids the >>1 truncation of the old approach.
    reg [3:0]  a_flat_reg  [0:63]; // 4-bit unsigned mantissa per lane
    reg [3:0]  b_flat_reg  [0:63]; // 4-bit unsigned mantissa per lane
    reg [63:0] sign_flat_reg;      // product sign per lane: sign_x XOR sign_w

    wire [255:0] a_flat_wire;
    wire [255:0] b_flat_wire;

    integer i;
    reg [3:0] mant_x, mant_w;

    always @(*) begin
        for (i = 0; i < 64; i = i + 1) begin
            // 1. Reconstruct full 4-bit unsigned mantissa from BFP bit-planes
            mant_x = {axi_planes[192+i], axi_planes[128+i], axi_planes[64+i], axi_planes[i]};
            mant_w = {awi_planes[192+i], awi_planes[128+i], awi_planes[64+i], awi_planes[i]};

            a_flat_reg[i] = mant_x;
            b_flat_reg[i] = mant_w;

            // 2. Compute product sign (sign-magnitude rule: sign = XOR of input signs)
            sign_flat_reg[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    end

    // Pack mantissa registers into flat buses
    genvar k;
    generate
        for (k = 0; k < 64; k = k + 1) begin : pack_flat
            assign a_flat_wire[4*k +: 4] = a_flat_reg[k];
            assign b_flat_wire[4*k +: 4] = b_flat_reg[k];
        end
    endgenerate

    // ---------------- INT MAC Instantiation ----------------
    int_mac_64 mac_array (
        .clk(clk),
        .rst(rst),
        .load(valid_in),
        .a_flat(a_flat_wire),
        .b_flat(b_flat_wire),
        .sign_flat(sign_flat_reg),   // per-lane product sign
        .alpha_x(4'd1),
        .alpha_w(4'd1),
        .beta_xw(8'sd0),
        .result(wide_integer_sum)
    );

    // ---------------- Final Exponent ----------------
    // No >>1 compensation needed anymore. Full 4-bit mantissas are used (hidden bit at
    // position 3, so each mantissa integer is 2^3 = 8x the true fraction).
    // Two mantissas multiplied → 2^6 total scale → same formula as FP8 MLB: bias*2 + 6 = 14+6 = 20
    assign shared_exponent = $signed({5'b0, max_exp_x}) + $signed({5'b0, max_exp_w}) - 9'sd20;

endmodule