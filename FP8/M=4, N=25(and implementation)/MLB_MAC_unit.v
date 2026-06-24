module MLB_unit(
    output signed [11:0] out,
    output done,
    input [24:0] axi, awi,
    input [2:0] shift_amt, // Replaces alpha_x and alpha_w
    input clk, rst, valid_in
);
    wire [4:0] inter;
    wire xp_done;
    
    // The core Popcount engine
    and_popcount_4_bit xp(
        .f_output(inter),
        .done(xp_done),
        .a(axi),
        .b(awi),
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in)
    );

    // ZERO-GATE MULTIPLIER: Just shift the popcount!
    reg signed [11:0] shifted_out;
    reg done_reg;

    always @(posedge clk) begin
        if (rst) begin
            shifted_out <= 12'sd0;
            done_reg <= 1'b0;
        end else begin
            done_reg <= xp_done;
            if (xp_done) begin
                // Cast to 12-bit signed, then shift by (i+j)
                shifted_out <= $signed({1'b0, inter}) << shift_amt;
            end
        end
    end

    assign out = shifted_out;
    assign done = done_reg;
endmodule
