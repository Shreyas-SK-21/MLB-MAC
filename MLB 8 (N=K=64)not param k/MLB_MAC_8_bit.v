module MLB_8(output signed [30:0] mlb,input [63:0] alpha_x,alpha_w,input [511:0] axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire signed [28:0] M4s[3:0];
    wire done0,done1,done2,done3;
    MLB_4 M1(.mlb(M4s[0]),.alpha_x(alpha_x[31:0]),.alpha_w(alpha_w[31:0]),.axi(axi[255:0]),.awi(awi[255:0]));//lower A * lower B
    MLB_4 M2(.mlb(M4s[1]),.alpha_x(alpha_x[31:0]),.alpha_w(alpha_w[63:32]),.axi(axi[255:0]),.awi(awi[511:256]));//lower A * higher B
    MLB_4 M3(.mlb(M4s[2]),.alpha_x(alpha_x[63:32]),.alpha_w(alpha_w[31:0]),.axi(axi[511:256]),.awi(awi[255:0]));//higher A * lower B
    MLB_4 M4(.mlb(M4s[3]),.alpha_x(alpha_x[63:32]),.alpha_w(alpha_w[63:32]),.axi(axi[511:256]),.awi(awi[511:256]));//higher A * higher B
    assign mlb=(M4s[0])+(M4s[1]+M4s[2])+(M4s[3]);
endmodule