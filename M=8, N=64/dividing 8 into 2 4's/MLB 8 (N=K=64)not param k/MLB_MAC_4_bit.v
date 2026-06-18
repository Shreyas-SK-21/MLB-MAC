module MLB_4(output signed [28:0] mlb,input [31:0] alpha_x,alpha_w,input [255:0] axi,awi);
    genvar i,j;
    wire signed [24:0] out[3:0][3:0];
    generate
        for(i=0; i<4; i=i+1) begin
            for(j=0; j<4; j=j+1) begin
                MLB_unit u_ij(.out(out[i][j]),.alpha_x(alpha_x[i*8+7:i*8]),.alpha_w(alpha_w[j*8+7:j*8]),.axi(axi[i*64+63:64*i]),.awi(awi[j*64+63:64*j]));
            end
        end
    endgenerate
    //reduction tree
    wire signed [25:0] s00,s01,s02,s03,s04,s05,s06,s07;
    wire signed [26:0] s10,s11,s12,s13;
    wire signed [27:0] s20,s21;

    assign s00=out[0][0]+out[0][1];
    assign s01=out[0][2]+out[0][3];
    assign s02=out[1][0]+out[1][1];
    assign s03=out[1][2]+out[1][3];   
    assign s04=out[2][0]+out[2][1];
    assign s05=out[2][2]+out[2][3];
    assign s06=out[3][0]+out[3][1];
    assign s07=out[3][2]+out[3][3];

    assign s10=s00+s01;
    assign s11=s02+s03;
    assign s12=s04+s05;
    assign s13=s06+s07;

    assign s20=s10+s11;
    assign s21=s12+s13;

    assign mlb=s20+s21;
endmodule