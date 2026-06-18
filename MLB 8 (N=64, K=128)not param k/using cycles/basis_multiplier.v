module basis_multiplier(output reg signed [25:0] basis_mult,output reg done,input xp_done,rst,clk,input signed [8:0] xnor_popcount,input [7:0] alpha_x, alpha_w);
    reg signed [16:0] alpha_product_signed_reg;
    reg signed [8:0] xnor_popcount_reg; 
    reg stage1_valid;
    always @(posedge clk) begin
        if (rst) begin
            alpha_product_signed_reg<=17'sd0;
            xnor_popcount_reg<=9'sd0;
            stage1_valid<=1'b0;
            basis_mult<=26'sd0;
            done<=1'b0;
        end 
        else begin
            done<=1'b0;
            stage1_valid<=1'b0;
            if (xp_done) begin
                alpha_product_signed_reg<=alpha_x*alpha_w;
                xnor_popcount_reg<=xnor_popcount;
                stage1_valid<=1'b1;
            end
            if (stage1_valid) begin
                basis_mult<=xnor_popcount_reg*alpha_product_signed_reg;
                done<=1'b1;
            end
        end
    end
endmodule