module MLB_unit(output [19:0] out,input [7:0] alpha_x,alpha_w,axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants

    wire [3:0] inter;//intermediates
    xnor_popcount_8_bit xp(.xnorpop(inter),.a(axi),.b(awi));
    basis_multiplier bm(.basis_mult(out),.xnor_popcount(inter),.alpha_x(alpha_x),.alpha_w(alpha_w));
endmodule