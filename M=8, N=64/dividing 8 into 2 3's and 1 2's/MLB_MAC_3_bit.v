module MLB_3(
    output reg signed [36:0] mlb,
    output reg done,
    input [23:0] alpha_x, alpha_w,
    input [191:0] axi, awi,
    input [12:0] K,
    input clk, rst, valid_in
);

    wire signed [32:0] out[0:8];
    wire [8:0] unit_done;
    genvar i, j;

    generate
        for(i=0; i<3; i=i+1) begin
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

    // Reduction tree for 9 elements
    wire signed [33:0] s00, s01, s02, s03;
    wire signed [34:0] s10, s11;
    wire signed [35:0] s20;
    wire signed [36:0] s30;

    assign s00 = out[0] + out[1];
    assign s01 = out[2] + out[3];
    assign s02 = out[4] + out[5];
    assign s03 = out[6] + out[7];

    assign s10 = s00 + s01;
    assign s11 = s02 + s03;

    assign s20 = s10 + s11;
    assign s30 = s20 + out[8]; // Adds the final 9th element

    always @(posedge clk) begin
        done <= 1'b0;
        if(rst) begin
            mlb <= 37'sd0;
            done <= 1'b0;
        end
        else if(&unit_done) begin
            mlb <= s30;
            done <= 1'b1;
        end
    end
endmodule