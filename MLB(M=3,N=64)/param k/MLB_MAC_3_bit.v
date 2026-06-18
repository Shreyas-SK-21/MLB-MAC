module MLB_3 (
    output reg signed [27:0] mlb,    // was [20:0] wire; +7 bits for wider units
    output reg               done,
    input      [8:0]         alpha_x, alpha_w,   // 3 x 3-bit values packed
    input      [191:0]       axi, awi,            // 3 x 64-bit slices packed
    input      [12:0]        K,//must be a multiple of N 
    input                    clk, rst, valid_in
);

    genvar i, j;
    wire signed [21:0] out [0:2][0:2];   // was [14:0]
    wire               unit_done [0:2][0:2];

    generate
        for(i=0; i<3; i=i+1) begin : GI
            for(j=0; j<3; j=j+1) begin : GJ
                MLB_unit u_ij (
                    .out      (out[i][j]),
                    .done     (unit_done[i][j]),
                    .alpha_x  (alpha_x[i*3+2 : i*3]),
                    .alpha_w  (alpha_w[j*3+2 : j*3]),
                    .axi      (axi[i*64+63 : i*64]),
                    .awi      (awi[j*64+63 : j*64]),
                    .K        (K),
                    .clk      (clk),
                    .rst      (rst),
                    .valid_in (valid_in)
                );
            end
        end
    endgenerate

    // ----------------------------------------------------------
    // Reduction tree — same structure as original, wider wires
    // All 9 units finish the same cycle (same K, same valid_in)
    // ----------------------------------------------------------
    wire signed [22:0] s00, s01, s02, s03;   // was [15:0]
    wire signed [22:0] s04;
    wire signed [23:0] s10, s11, s12;         // was [16:0]
    wire signed [24:0] s20, s21;              // was [17:0]
    wire signed [25:0] s3;

    assign s00 = out[0][0] + out[0][1];
    assign s01 = out[0][2] + out[1][2];
    assign s02 = out[1][0] + out[1][1];
    assign s03 = out[2][0] + out[2][1];
    assign s04 = out[2][2];

    assign s10 = s00 + s01;
    assign s11 = s02 + s03;
    assign s12 = s04;

    assign s20 = s10 + s11;
    assign s21 = s12;

    assign s3  = s20 + s21;

    // All units are driven by the same valid_in and K,
    // so unit_done[0][0] firing means all 9 have fired
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            mlb  <= 28'sd0;
            done <= 1'b0;
        end else if (unit_done[0][0]) begin
            mlb  <= s3;
            done <= 1'b1;
        end
    end
endmodule