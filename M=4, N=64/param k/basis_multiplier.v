module basis_multiplier(output reg signed [24:0] basis_mult,output reg done,input xp_done,rst,clk,input signed [15:0]xnor_popcount,input [3:0]alpha_x,alpha_w);
    wire [7:0] alpha_product;
    assign alpha_product=alpha_x*alpha_w;
    wire signed [8:0] alpha_product_signed;
    assign alpha_product_signed={1'b0,alpha_product};//9 bits 9+16 = 25
    always @(posedge clk) begin
        done<=1'b0;
        if(rst) begin
            basis_mult<=25'sd0;
            done<=1'b0;
        end 
        else if(xp_done) begin
            basis_mult<=xnor_popcount*alpha_product_signed;
            done<=1'b1;
        end
    end
endmodule