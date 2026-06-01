module MLB_16(output [40:0] mlb,input [255:0] alpha_x,alpha_w,axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire [591:0] out;
    
    MLB_unit MLB_1(.out(out[36:0]),.alpha_x(alpha_x[15:0]),.alpha_w(alpha_w[15:0]),.axi(axi[15:0]),.awi(awi[15:0]));
    MLB_unit MLB_2(.out(out[73:37]),.alpha_x(alpha_x[31:16]),.alpha_w(alpha_w[31:16]),.axi(axi[31:16]),.awi(awi[31:16]));
    MLB_unit MLB_3(.out(out[110:74]),.alpha_x(alpha_x[47:32]),.alpha_w(alpha_w[47:32]),.axi(axi[47:32]),.awi(awi[47:32]));
    MLB_unit MLB_4(.out(out[147:111]),.alpha_x(alpha_x[63:48]),.alpha_w(alpha_w[63:48]),.axi(axi[63:48]),.awi(awi[63:48]));
    MLB_unit MLB_5(.out(out[184:148]),.alpha_x(alpha_x[79:64]),.alpha_w(alpha_w[79:64]),.axi(axi[79:64]),.awi(awi[79:64]));
    MLB_unit MLB_6(.out(out[221:185]),.alpha_x(alpha_x[95:80]),.alpha_w(alpha_w[95:80]),.axi(axi[95:80]),.awi(awi[95:80]));
    MLB_unit MLB_7(.out(out[258:222]),.alpha_x(alpha_x[111:96]),.alpha_w(alpha_w[111:96]),.axi(axi[111:96]),.awi(awi[111:96]));
    MLB_unit MLB_8(.out(out[295:259]),.alpha_x(alpha_x[127:112]),.alpha_w(alpha_w[127:112]),.axi(axi[127:112]),.awi(awi[127:112]));
    MLB_unit MLB_9(.out(out[332:296]),.alpha_x(alpha_x[143:128]),.alpha_w(alpha_w[143:128]),.axi(axi[143:128]),.awi(awi[143:128]));
    MLB_unit MLB_10(.out(out[369:333]),.alpha_x(alpha_x[159:144]),.alpha_w(alpha_w[159:144]),.axi(axi[159:144]),.awi(awi[159:144]));
    MLB_unit MLB_11(.out(out[406:370]),.alpha_x(alpha_x[175:160]),.alpha_w(alpha_w[175:160]),.axi(axi[175:160]),.awi(awi[175:160]));
    MLB_unit MLB_12(.out(out[443:407]),.alpha_x(alpha_x[191:176]),.alpha_w(alpha_w[191:176]),.axi(axi[191:176]),.awi(awi[191:176]));
    MLB_unit MLB_13(.out(out[480:444]),.alpha_x(alpha_x[207:192]),.alpha_w(alpha_w[207:192]),.axi(axi[207:192]),.awi(awi[207:192]));
    MLB_unit MLB_14(.out(out[517:481]),.alpha_x(alpha_x[223:208]),.alpha_w(alpha_w[223:208]),.axi(axi[223:208]),.awi(awi[223:208]));
    MLB_unit MLB_15(.out(out[554:518]),.alpha_x(alpha_x[239:224]),.alpha_w(alpha_w[239:224]),.axi(axi[239:224]),.awi(awi[239:224]));
    MLB_unit MLB_16(.out(out[591:555]),.alpha_x(alpha_x[255:240]),.alpha_w(alpha_w[255:240]),.axi(axi[255:240]),.awi(awi[255:240]));
    
    reduction_tree_16 red_tree(.sum(mlb),.a0(out[36:0]),.a1(out[73:37]),.a2(out[110:74]),.a3(out[147:111]),.a4(out[184:148]),.a5(out[221:185]),.a6(out[258:222]),.a7(out[295:259]),.a8(out[332:296]),.a9(out[369:333]),.a10(out[406:370]),.a11(out[443:407]),.a12(out[480:444]),.a13(out[517:481]),.a14(out[554:518]),.a15(out[591:555]));
endmodule