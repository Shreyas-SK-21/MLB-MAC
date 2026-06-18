module basis_multiplier (
    output reg signed [23:0] basis_mult,
    output reg done,
    input xp_done, rst, clk,
    input signed [7:0] xnor_popcount,
    input [7:0] alpha_x, alpha_w
);

    // Pipeline Registers (Intermediate storage)
    reg signed [16:0] alpha_product_signed_reg;
    reg signed [7:0]  xnor_popcount_reg;
    reg               stage1_valid; // Tells Stage 2 that data is ready

    always @(posedge clk) begin
        if (rst) begin
            // Reset all registers
            alpha_product_signed_reg <= 17'sd0;
            xnor_popcount_reg <= 8'sd0;
            stage1_valid <= 1'b0;
            basis_mult <= 24'sd0;
            done <= 1'b0;
        end 
        else begin
            // Default signals (act as auto-clear)
            done <= 1'b0;
            stage1_valid <= 1'b0;

            // ----------------------------------------------------
            // PIPELINE STAGE 1: First Multiplication & Capture
            // ----------------------------------------------------
// ----------------------------------------------------
            // PIPELINE STAGE 1: First Multiplication & Capture
            // ----------------------------------------------------
            if (xp_done) begin
                // Context-determined width: Verilog will zero-extend the 8-bit 
                // unsigned inputs to 17 bits before multiplying, preserving the full 65025.
                alpha_product_signed_reg <= alpha_x * alpha_w;

                // CRITICAL: Save xnor_popcount so it aligns with the delayed product
                xnor_popcount_reg <= xnor_popcount;

                // Tell the next stage to compute on the next clock edge
                stage1_valid <= 1'b1;
            end

            // ----------------------------------------------------
            // PIPELINE STAGE 2: Final Multiplication
            // ----------------------------------------------------
            if (stage1_valid) begin
                // Multiply the stored values
                basis_mult <= xnor_popcount_reg * alpha_product_signed_reg;
                
                // Assert final done signal
                done <= 1'b1;
            end
        end
    end
endmodule