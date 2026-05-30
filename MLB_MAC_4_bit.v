module MLB_4(output [11:0] mlb,input [15:0] alpha_x,alpha_w,axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire [39:0] out;
    MLB_unit MLB_1(.out(out[9:0]),.alpha_x(alpha_x[3:0]),.alpha_w(alpha_w[3:0]),.axi(axi[3:0]),.awi(awi[3:0]));
    MLB_unit MLB_2(.out(out[19:10]),.alpha_x(alpha_x[7:4]),.alpha_w(alpha_w[7:4]),.axi(axi[7:4]),.awi(awi[7:4]));
    MLB_unit MLB_3(.out(out[29:20]),.alpha_x(alpha_x[11:8]),.alpha_w(alpha_w[11:8]),.axi(axi[11:8]),.awi(awi[11:8]));
    MLB_unit MLB_4(.out(out[39:30]),.alpha_x(alpha_x[15:12]),.alpha_w(alpha_w[15:12]),.axi(axi[15:12]),.awi(awi[15:12]));
    reduction_tree_4 red_tree(.sum(mlb),.a0(out[9:0]),.a1(out[19:10]),.a2(out[29:20]),.a3(out[39:30]));
endmodule