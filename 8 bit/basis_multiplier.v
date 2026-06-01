module basis_multiplier(output [19:0] basis_mult,input [3:0]xnor_popcount,input [7:0]alpha_x,alpha_w);
    wire [15:0] inter;//intermediates
    multiplier_8 mult_0(.o(inter),.a(alpha_x),.b(alpha_w));//multipliers
    multiplier_16_4 mult_1(.o(basis_mult),.a(inter),.b(xnor_popcount));
endmodule