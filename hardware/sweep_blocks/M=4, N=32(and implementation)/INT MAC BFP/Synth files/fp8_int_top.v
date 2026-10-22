module fp8_int_top (
    input  clk,
    input  rst,
    input  valid_in, // Connects to 'load' on the MAC array
    input  [255:0] fp8_activations,
    input  [255:0] fp8_weights,
    output signed [20:0] wide_integer_sum,
    output signed [8:0]   shared_exponent
);

    wire [127:0] axi_planes;
    wire [127:0] awi_planes;
    wire [3:0]   max_exp_x;
    wire [3:0]   max_exp_w;

    bfp_aligner align_x (
        .clk(clk),
        .fp8_vec(fp8_activations),
        .aligned_planes(axi_planes),
        .max_exp(max_exp_x)
    );

    bfp_aligner align_w (
        .clk(clk),
        .fp8_vec(fp8_weights),
        .aligned_planes(awi_planes),
        .max_exp(max_exp_w)
    );

    reg [255:0] fp8_activations_d1;
    reg [255:0] fp8_weights_d1;
    always @(posedge clk) begin
        fp8_activations_d1 <= fp8_activations;
        fp8_weights_d1     <= fp8_weights;
    end

    reg [3:0]  a_flat_reg  [0:31]; // 4-bit unsigned mantissa per lane
    reg [3:0]  b_flat_reg  [0:31]; // 4-bit unsigned mantissa per lane
    reg [31:0] sign_flat_reg;      // product sign per lane: sign_x XOR sign_w

    wire [127:0] a_flat_wire;
    wire [127:0] b_flat_wire;

    integer i;
    reg [3:0] mant_x, mant_w;

    always @(*) begin
        for (i = 0; i < 32; i = i + 1) begin
            // 1. Reconstruct full 4-bit unsigned mantissa from BFP bit-planes
            mant_x = {axi_planes[96+i], axi_planes[64+i], axi_planes[32+i], axi_planes[i]};
            mant_w = {awi_planes[96+i], awi_planes[64+i], awi_planes[32+i], awi_planes[i]};

            a_flat_reg[i] = mant_x;
            b_flat_reg[i] = mant_w;

            // 2. Compute product sign (sign-magnitude rule: sign = XOR of input signs)
            //    using the delayed (aligned-in-time) activation/weight copies.
            sign_flat_reg[i] = fp8_activations_d1[i*8 + 7] ^ fp8_weights_d1[i*8 + 7];
        end
    end

    // Pack mantissa registers into flat buses
    genvar k;
    generate
        for (k = 0; k < 32; k = k + 1) begin : pack_flat
            assign a_flat_wire[4*k +: 4] = a_flat_reg[k];
            assign b_flat_wire[4*k +: 4] = b_flat_reg[k];
        end
    endgenerate

    reg valid_in_d1;
    always @(posedge clk) begin
        if (rst) valid_in_d1 <= 1'b0;
        else     valid_in_d1 <= valid_in;
    end

    int_mac_32 mac_array (
        .clk(clk),
        .rst(rst),
        .load(valid_in_d1),
        .a_flat(a_flat_wire),
        .b_flat(b_flat_wire),
        .sign_flat(sign_flat_reg),   // per-lane product sign
        .alpha_x(4'd1),
        .alpha_w(4'd1),
        .beta_xw(8'sd0),
        .result(wide_integer_sum)
    );

    assign shared_exponent = $signed({5'b0, max_exp_x}) + $signed({5'b0, max_exp_w}) - 9'sd20;

endmodule
