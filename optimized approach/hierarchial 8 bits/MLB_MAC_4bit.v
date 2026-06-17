module MLB_4_opt(
    output reg signed [36:0] mlb,
    output reg done,
    input [31:0] alpha_x, alpha_w,
    input [255:0] axi, awi,
    input [12:0] K,
    input clk, rst, valid_in
);
    genvar i, j;
    wire signed [32:0] out [0:15];
    wire [15:0] unit_done;

    generate
        for(i=0; i<4; i=i+1) begin : row
            for(j=0; j<4; j=j+1) begin : col
                MLB_unit_opt u_ij(
                    .out(out[i*4+j]),
                    .done(unit_done[i*4+j]),
                    .alpha_x(alpha_x[i*8+7:i*8]),
                    .alpha_w(alpha_w[j*8+7:j*8]),
                    .axi(axi[i*64+63:64*i]),
                    .awi(awi[j*64+63:64*j]),
                    .clk(clk), .rst(rst), .valid_in(valid_in), .K(K)
                );
            end
        end
    endgenerate

    // --------------------------------------------------------
    // Behavioral Reduction Tree (Optimal Synthesis)
    // --------------------------------------------------------
    reg signed [36:0] combinational_sum;
    integer k;
    
    always @(*) begin
        combinational_sum = 37'sd0;
        for(k=0; k<16; k=k+1) begin
            combinational_sum = combinational_sum + out[k];
        end
    end

    always @(posedge clk) begin
        if(rst) begin
            mlb <= 37'sd0;
            done <= 1'b0;
        end else begin
            // Update output only when all basis multipliers are finished
            if(&unit_done) begin
                mlb <= combinational_sum;
            end
            done <= &unit_done;
        end
    end
endmodule