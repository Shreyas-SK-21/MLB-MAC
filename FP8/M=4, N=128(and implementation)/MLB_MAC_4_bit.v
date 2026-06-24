module MLB_4(
    output reg signed [20:0] mlb,
    output reg done,
    input [511:0] axi, awi,
    input clk, rst, valid_in
);
    genvar i, j;
    wire signed [16:0] out[3:0][3:0];
    wire [15:0] unit_done;

    generate
        for(i=0; i<4; i=i+1) begin : rows
            for(j=0; j<4; j=j+1) begin : cols
                // Create a strict 3-bit wire to silence the port size warning
                wire [2:0] shift_val = i + j; 
                
                MLB_unit u_ij(
                    .out(out[i][j]),
                    .done(unit_done[4*i+j]),
                    .shift_amt(shift_val), 
                    .axi(axi[i*128+127 : 128*i]),
                    .awi(awi[j*128+127 : 128*j]),
                    .clk(clk),
                    .rst(rst),
                    .valid_in(valid_in)
                );
            end
        end
    endgenerate

    // Reduction tree (Unchanged)
    wire signed [17:0] s00,s01,s02,s03,s04,s05,s06,s07;
    wire signed [18:0] s10,s11,s12,s13;
    wire signed [19:0] s20,s21;

    assign s00=out[0][0]+out[0][1]; assign s01=out[0][2]+out[0][3];
    assign s02=out[1][0]+out[1][1]; assign s03=out[1][2]+out[1][3];   
    assign s04=out[2][0]+out[2][1]; assign s05=out[2][2]+out[2][3];
    assign s06=out[3][0]+out[3][1]; assign s07=out[3][2]+out[3][3];

    assign s10=s00+s01; assign s11=s02+s03;
    assign s12=s04+s05; assign s13=s06+s07;
    assign s20=s10+s11; assign s21=s12+s13;

    always @(posedge clk) begin
        if(rst) begin
            mlb  <= 21'sd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0; // Default state
            if(&unit_done) begin
                mlb  <= s20+s21;
                done <= 1'b1;
            end
        end
    end
endmodule
