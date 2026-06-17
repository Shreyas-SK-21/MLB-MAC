module xnor_popcount_4_bit_opt(
    output reg signed [15:0] signed_output,
    output reg done,
    input [63:0] a, b,
    input [12:0] K,
    input clk, rst, valid_in
);
    wire [63:0] xnn = ~(a ^ b);
    reg [6:0] xnorpop;
    integer i;

    // Combinational popcount
    always @(*) begin
        xnorpop = 7'd0;
        for(i=0; i<64; i=i+1) begin
            xnorpop = xnorpop + xnn[i];
        end
    end

    wire signed [8:0] contri = {1'b0, xnorpop, 1'b0} - 9'sd64; // 2P - N
    reg signed [15:0] acc;
    reg [6:0] cycle_count;
    wire [6:0] cycles_needed = K[12:6];

    always @(posedge clk) begin
        if (rst) begin
            acc <= 16'sd0;
            cycle_count <= 7'd0;
            signed_output <= 16'sd0;
            done <= 1'b0;
        end else if (valid_in) begin
            if (cycle_count == cycles_needed - 1) begin
                // Register the final output
                signed_output <= acc + contri; 
                done <= 1'b1;
                
                // Reset accumulators for next batch
                acc <= 16'sd0;
                cycle_count <= 7'd0;
            end else begin
                acc <= acc + contri;
                cycle_count <= cycle_count + 1;
                done <= 1'b0;
            end
        end else begin
            done <= 1'b0; // Ensure done pulses only for one cycle
        end
    end
endmodule