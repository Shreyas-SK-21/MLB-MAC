module MLB_8_332(
    output signed [38:0] mlb,
    output done,
    input [63:0] alpha_x, alpha_w,
    input [511:0] axi, awi,
    input [12:0] K,
    input clk, rst, valid_in
);

    // Outputs from the 9 sub-modules
    wire signed [36:0] M3_00, M3_01, M3_10, M3_11; // 3x3 symmetric blocks
    wire signed [35:0] M3x2_02, M3x2_12;           // 3x2 asymmetric blocks
    wire signed [35:0] M2x3_20, M2x3_21;           // 2x3 asymmetric blocks
    wire signed [34:0] M2_22;                      // 2x2 symmetric block

    wire d00, d01, d02, d10, d11, d12, d20, d21, d22;

    // --- ROW 0: A is Lower 3 bits ---
    MLB_3 M00 (.mlb(M3_00), .done(d00), .alpha_x(alpha_x[23:0]), .alpha_w(alpha_w[23:0]), .axi(axi[191:0]), .awi(awi[191:0]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));
    MLB_3 M01 (.mlb(M3_01), .done(d01), .alpha_x(alpha_x[23:0]), .alpha_w(alpha_w[47:24]), .axi(axi[191:0]), .awi(awi[383:192]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));
    MLB_3x2 M02 (.mlb(M3x2_02), .done(d02), .alpha_x(alpha_x[23:0]), .alpha_w(alpha_w[63:48]), .axi(axi[191:0]), .awi(awi[511:384]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));

    // --- ROW 1: A is Middle 3 bits ---
    MLB_3 M10 (.mlb(M3_10), .done(d10), .alpha_x(alpha_x[47:24]), .alpha_w(alpha_w[23:0]), .axi(axi[383:192]), .awi(awi[191:0]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));
    MLB_3 M11 (.mlb(M3_11), .done(d11), .alpha_x(alpha_x[47:24]), .alpha_w(alpha_w[47:24]), .axi(axi[383:192]), .awi(awi[383:192]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));
    MLB_3x2 M12 (.mlb(M3x2_12), .done(d12), .alpha_x(alpha_x[47:24]), .alpha_w(alpha_w[63:48]), .axi(axi[383:192]), .awi(awi[511:384]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));

    // --- ROW 2: A is Upper 2 bits ---
    MLB_2x3 M20 (.mlb(M2x3_20), .done(d20), .alpha_x(alpha_x[63:48]), .alpha_w(alpha_w[23:0]), .axi(axi[511:384]), .awi(awi[191:0]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));
    MLB_2x3 M21 (.mlb(M2x3_21), .done(d21), .alpha_x(alpha_x[63:48]), .alpha_w(alpha_w[47:24]), .axi(axi[511:384]), .awi(awi[383:192]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));
    MLB_2 M22 (.mlb(M2_22), .done(d22), .alpha_x(alpha_x[63:48]), .alpha_w(alpha_w[63:48]), .axi(axi[511:384]), .awi(awi[511:384]), .K(K), .clk(clk), .rst(rst), .valid_in(valid_in));

    // --- Final Reduction Tree ---
    wire signed [37:0] row0_sum, row1_sum, row2_sum;

    assign row0_sum = M3_00 + M3_01 + M3x2_02;
    assign row1_sum = M3_10 + M3_11 + M3x2_12;
    assign row2_sum = M2x3_20 + M2x3_21 + M2_22;

    assign mlb = row0_sum + row1_sum + row2_sum;

    // AND all unit done signals
    assign done = d00 & d01 & d02 & d10 & d11 & d12 & d20 & d21 & d22;

endmodule