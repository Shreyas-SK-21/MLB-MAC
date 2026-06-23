module fp8_int_top (
    input  clk,
    input  rst,
    input  valid_in, // Connects to 'load' on the MAC array
    input  [511:0] fp8_activations,
    input  [511:0] fp8_weights,
    output signed [20:0] wide_integer_sum,
    output [8:0]   shared_exponent
);

    // ---------------- BFP alignment (Reused from your MLB design) ----------------
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

    // ---------------- Format for INT MAC (Plane-to-Flat + Sign Application) ----------------
    reg signed [3:0] a_flat_reg [0:63];
    reg signed [3:0] b_flat_reg [0:63];
    wire [255:0] a_flat_wire;
    wire [255:0] b_flat_wire;

    integer i;
    // FIXED: Variables declared outside the always block to satisfy Verilog-2001
    reg [3:0] mant_x;
    reg [3:0] mant_w;
    reg sign_x;
    reg sign_w;
    reg [3:0] signed_x;
    reg [3:0] signed_w;

    always @(*) begin
        for (i = 0; i < 64; i = i + 1) begin
            // 1. Reconstruct 4-bit unsigned mantissa from the BFP bit-planes
            mant_x = {axi_planes[192+i], axi_planes[128+i], axi_planes[64+i], axi_planes[i]};
            mant_w = {awi_planes[192+i], awi_planes[128+i], awi_planes[64+i], awi_planes[i]};

            // 2. Extract signs from the original FP8 inputs
            sign_x = fp8_activations[i*8 + 7];
            sign_w = fp8_weights[i*8 + 7];

            // 3. Shift by 1 to fit in 4-bit signed (max val becomes 7), then apply Two's Complement sign
            signed_x = mant_x >> 1; 
            if (sign_x) a_flat_reg[i] = -signed_x;
            else        a_flat_reg[i] = signed_x;

            signed_w = mant_w >> 1;
            if (sign_w) b_flat_reg[i] = -signed_w;
            else        b_flat_reg[i] = signed_w;
        end
    end

    // Pack the registers into the flat 256-bit wires required by your int_mac_64 module
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
        .load(valid_in),         // valid_in acts as the load signal for the accumulators
        .a_flat(a_flat_wire),
        .b_flat(b_flat_wire),
        .alpha_x(4'd1),          // Default scale (can be mapped to actual quantization scales later)
        .alpha_w(4'd1),
        .beta_xw(8'sd0),         // Default offset
        .result(wide_integer_sum)
    );

    // ---------------- Final Exponent ----------------
    // E4M3 bias is 7. Because we shifted mantissas right by 1 (which acts as a divide by 2), 
    // we must compensate by adding 1 to the exponent for both x and w (total +2).
    // Original formula: exp_x + exp_w - 14. 
    // Adjusted formula: exp_x + exp_w - 12.
    assign shared_exponent = {5'b0, max_exp_x} + {5'b0, max_exp_w} - 9'd12;

endmodule