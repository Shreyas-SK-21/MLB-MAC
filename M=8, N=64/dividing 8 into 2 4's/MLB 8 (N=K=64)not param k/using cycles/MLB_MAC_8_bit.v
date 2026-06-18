module MLB_8(output signed [29:0] mlb,output done,input [63:0] alpha_x,alpha_w,input [511:0] axi,awi,input clk,rst,valid_in);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire signed [27:0] M4s[3:0];
    wire done0,done1,done2,done3;
    MLB_4 M1(.mlb(M4s[0]),.done(done0),.alpha_x(alpha_x[31:0]),.alpha_w(alpha_w[31:0]),.axi(axi[255:0]),.awi(awi[255:0]),.clk(clk),.rst(rst),.valid_in(valid_in));//lower A * lower B
    MLB_4 M2(.mlb(M4s[1]),.done(done1),.alpha_x(alpha_x[31:0]),.alpha_w(alpha_w[63:32]),.axi(axi[255:0]),.awi(awi[511:256]),.clk(clk),.rst(rst),.valid_in(valid_in));//lower A * higher B
    MLB_4 M3(.mlb(M4s[2]),.done(done2),.alpha_x(alpha_x[63:32]),.alpha_w(alpha_w[31:0]),.axi(axi[511:256]),.awi(awi[255:0]),.clk(clk),.rst(rst),.valid_in(valid_in));//higher A * lower B
    MLB_4 M4(.mlb(M4s[3]),.done(done3),.alpha_x(alpha_x[63:32]),.alpha_w(alpha_w[63:32]),.axi(axi[511:256]),.awi(awi[511:256]),.clk(clk),.rst(rst),.valid_in(valid_in));//higher A * higher B
    assign mlb=(M4s[0])+(M4s[1]+M4s[2])+(M4s[3]);
    assign done=done0&done1&done2&done3;
endmodule