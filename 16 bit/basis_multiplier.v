module basis_multiplier(output [36:0] basis_mult,input [4:0]xnor_popcount,input [15:0]alpha_x,alpha_w);
    wire [31:0] inter;//intermediates
    multiplier_16 mult_0(.o(inter),.a(alpha_x),.b(alpha_w));//multipliers
    multiplier_32_5 mult_1(.o(basis_mult),.a(inter),.b(xnor_popcount));
endmodule