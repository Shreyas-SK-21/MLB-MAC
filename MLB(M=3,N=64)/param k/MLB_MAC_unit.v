module MLB_unit (
    output reg signed [21:0] out,    // was [14:0]; wider: 16-bit xnorpop * 6-bit alpha
    output reg               done,
    input      [2:0]         alpha_x, alpha_w,
    input      [63:0]        axi, awi,
    input      [12:0]        K,
    input                    clk, rst, valid_in
);

    wire signed [15:0] inter;        // was [7:0]
    wire               xp_done;

    xnor_popcount_3_bit xp (
        .signed_output (inter),
        .done          (xp_done),
        .a             (axi),w
        .b             (awi),
        .K             (K),
        .clk           (clk),
        .rst           (rst),
        .valid_in      (valid_in)
    );

    // basis_multiplier is unchanged but we replicate its logic here
    // so we can register the result on the done edge
    wire [5:0]        alpha_product;
    wire signed [6:0] alpha_product_s;
    assign alpha_product   = alpha_x * alpha_w;
    assign alpha_product_s = {1'b0, alpha_product};

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            out  <= 22'sd0;
            done <= 1'b0;
        end else if (xp_done) begin
            out  <= inter * alpha_product_s;
            done <= 1'b1;
        end
    end

endmodule