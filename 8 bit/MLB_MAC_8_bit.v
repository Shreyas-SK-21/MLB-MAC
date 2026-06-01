module MLB_8(output [22:0] mlb,input [63:0] alpha_x,alpha_w,axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire [159:0] out;
    MLB_unit MLB_1(.out(out[19:0]),.alpha_x(alpha_x[7:0]),.alpha_w(alpha_w[7:0]),.axi(axi[7:0]),.awi(awi[7:0]));
    MLB_unit MLB_2(.out(out[39:20]),.alpha_x(alpha_x[15:8]),.alpha_w(alpha_w[15:8]),.axi(axi[15:8]),.awi(awi[15:8]));
    MLB_unit MLB_3(.out(out[59:40]),.alpha_x(alpha_x[23:16]),.alpha_w(alpha_w[23:16]),.axi(axi[23:16]),.awi(awi[23:16]));
    MLB_unit MLB_4(.out(out[79:60]),.alpha_x(alpha_x[31:24]),.alpha_w(alpha_w[31:24]),.axi(axi[31:24]),.awi(awi[31:24]));
    MLB_unit MLB_5(.out(out[99:80]),.alpha_x(alpha_x[39:32]),.alpha_w(alpha_w[39:32]),.axi(axi[39:32]),.awi(awi[39:32]));
    MLB_unit MLB_6(.out(out[119:100]),.alpha_x(alpha_x[47:40]),.alpha_w(alpha_w[47:40]),.axi(axi[47:40]),.awi(awi[47:40]));
    MLB_unit MLB_7(.out(out[139:120]),.alpha_x(alpha_x[55:48]),.alpha_w(alpha_w[55:48]),.axi(axi[55:48]),.awi(awi[55:48]));
    MLB_unit MLB_8(.out(out[159:140]),.alpha_x(alpha_x[63:56]),.alpha_w(alpha_w[63:56]),.axi(axi[63:56]),.awi(awi[63:56]));
    reduction_tree_8 red_tree(.sum(mlb),.a0(out[19:0]),.a1(out[39:20]),.a2(out[59:40]),.a3(out[79:60]),.a4(out[99:80]),.a5(out[119:100]),.a6(out[139:120]),.a7(out[159:140]));
endmodule