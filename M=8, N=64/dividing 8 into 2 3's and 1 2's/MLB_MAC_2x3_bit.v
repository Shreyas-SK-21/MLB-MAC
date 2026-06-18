module MLB_2x3(
    output reg signed [35:0] mlb,
    output reg done,
    input [15:0] alpha_x, 
    input [23:0] alpha_w,
    input [127:0] axi, 
    input [191:0] awi,
    input [12:0] K,
    input clk, rst, valid_in
);

    wire signed [32:0] out[0:5];
    wire [5:0] unit_done;
    genvar i, j;

    generate
        // i iterates over 2-bit A, j iterates over 3-bit W
        for(i=0; i<2; i=i+1) begin
            for(j=0; j<3; j=j+1) begin
                MLB_unit u_ij(
                    .out(out[i*3+j]),
                    .done(unit_done[i*3+j]),
                    .alpha_x(alpha_x[i*8+7:i*8]),
                    .alpha_w(alpha_w[j*8+7:j*8]),
                    .axi(axi[i*64+63:i*64]),
                    .awi(awi[j*64+63:j*64]),
                    .clk(clk), .rst(rst), .valid_in(valid_in), .K(K)
                );
            end
        end
    endgenerate

    // Reduction tree for 6 elements
    wire signed [33:0] s00, s01, s02;
    wire signed [34:0] s10;
    wire signed [35:0] s20;

    assign s00 = out[0] + out[1];
    assign s01 = out[2] + out[3];
    assign s02 = out[4] + out[5];

    assign s10 = s00 + s01;
    assign s20 = s10 + s02;

    always @(posedge clk) begin
        done <= 1'b0;
        if(rst) begin
            mlb <= 36'sd0;
            done <= 1'b0;
        end
        else if(&unit_done) begin
            mlb <= s20;
            done <= 1'b1;
        end
    end
endmodule