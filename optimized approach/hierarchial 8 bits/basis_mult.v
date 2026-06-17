module basis_multiplier_opt(
    output reg signed [32:0] basis_mult,
    output reg done_out, // Pipelined done signal
    input xp_done, rst, clk,
    input signed [15:0] xnor_popcount,
    input [7:0] alpha_x, alpha_w
);
    wire [15:0] alpha_product = alpha_x * alpha_w;
    wire signed [16:0] alpha_product_signed = {1'b0, alpha_product};

    always @(posedge clk) begin
        if (rst) begin
            basis_mult <= 33'sd0;
            done_out <= 1'b0;
        end else begin
            // POWER WIN: Only clock the multiplier logic when valid data arrives.
            if (xp_done) begin
                basis_mult <= xnor_popcount * alpha_product_signed;
            end
            
            // Pass the done signal forward to the next stage
            done_out <= xp_done; 
        end
    end
endmodule