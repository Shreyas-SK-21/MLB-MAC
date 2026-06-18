module basis_multiplier(output signed [24:0] basis_mult,input signed [7:0]xnor_popcount,input [7:0]alpha_x,alpha_w);
    wire [15:0] alpha_product;
    assign alpha_product=alpha_x*alpha_w;
    wire signed [16:0] alpha_product_signed;
    assign alpha_product_signed={1'b0,alpha_product};
    assign basis_mult=xnor_popcount*alpha_product_signed;//17+8=25
endmodule