module basis_multiplier(output [9:0] basis_mult,input [2:0]xnor_popcount,input [3:0]alpha_x,alpha_w);
    wire [7:0] inter;//intermediates
    multiplier_4 mult_0(.o(inter),.a(alpha_x),.b(alpha_w));//multipliers
    multiplier_8_3 mult_1(.o(basis_mult),.a(inter),.b(xnor_popcount));
endmodule