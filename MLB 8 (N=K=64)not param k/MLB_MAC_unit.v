module MLB_unit(output signed [24:0] out,input [7:0] alpha_x,alpha_w,input [63:0] axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire signed [7:0] inter;//intermediates
    xnor_popcount_4_bit xp(.signed_output(inter),.a(axi),.b(awi));
    basis_multiplier bm(.basis_mult(out),.xnor_popcount(inter),.alpha_x(alpha_x),.alpha_w(alpha_w));
endmodule