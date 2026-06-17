module MLB_4(output reg signed [28:0] mlb,output reg done,input [15:0] alpha_x,alpha_w,input [255:0] axi,awi,input [12:0]K,input clk,rst,valid_in);
    genvar i,j;
    wire signed [24:0] out[0:15];
    wire [15:0]unit_done;
    generate
        for(i=0;i<4;i=i+1) begin
            for(j=0;j<4;j=j+1) begin
                MLB_unit u_ij(.out(out[i*4+j]),.done(unit_done[i*4+j]),.alpha_x(alpha_x[i*4+3:i*4]),.alpha_w(alpha_w[j*4+3:j*4]),.axi(axi[i*64+63:64*i]),.awi(awi[j*64+63:64*j]),.clk(clk),.rst(rst),.valid_in(valid_in),.K(K));
            end
        end
    endgenerate
    // Reduction tree
    wire signed [25:0]s00,s01,s02,s03,s04,s05,s06,s07;
    wire signed [26:0]s10,s11,s12,s13;
    wire signed [27:0]s20,s21;
    wire signed [28:0]s30;

    assign s00=out[0]+out[1];
    assign s01=out[2]+out[3];
    assign s02=out[4]+out[5];
    assign s03=out[6]+out[7];   
    assign s04=out[8]+out[9];
    assign s05=out[10]+out[11];
    assign s06=out[12]+out[13];
    assign s07=out[14]+out[15];

    assign s10=s00+s01;
    assign s11=s02+s03;
    assign s12=s04+s05;
    assign s13=s06+s07;

    assign s20=s10+s11;
    assign s21=s12+s13;

    assign s30=s20+s21;

    always @(posedge clk) begin
        done<=1'b0;
        if(rst) begin
            mlb<=29'sd0;
            done<=1'b0;
        end
        else if(unit_done[0]) begin
            mlb<=s30;
            done<=1'b1;
        end
    end
endmodule