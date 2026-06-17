module MLB_unit(output signed [24:0] out,output done,input [3:0] alpha_x,alpha_w,input [63:0] axi,awi,input[12:0]K,input clk,rst,valid_in);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    wire signed [15:0] inter;//intermediates
    wire xp_done;
    xnor_popcount_4_bit xp(.signed_output(inter),.done(xp_done),.a(axi),.b(awi),.K(K),.clk(clk),.rst(rst),.valid_in(valid_in));
    basis_multiplier bm(.basis_mult(out),.done(done),.xp_done(xp_done),.rst(rst),.clk(clk),.xnor_popcount(inter),.alpha_x(alpha_x),.alpha_w(alpha_w));
endmodule