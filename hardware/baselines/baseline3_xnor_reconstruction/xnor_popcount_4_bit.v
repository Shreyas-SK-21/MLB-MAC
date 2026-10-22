// xnor_popcount_4_bit.v
// Computes bipolar inner product contribution: 2*popcount(XNOR(a,b)) - 64
// Replaces integer-loop accumulation with a 6-level balanced binary adder tree,
// which synthesizes as a shallow, area-efficient structure instead of a
// ripple-accumulation chain.
module xnor_popcount_4_bit(
    output reg signed [7:0] f_output,
    output reg              done,
    input      [63:0]       a, b,
    input                   clk, rst, valid_in
);
    wire [63:0] xnn = ~(a ^ b);

    // -------------------------------------------------------------------------
    // 6-level binary adder tree for 64-bit popcount
    // Level widths: 1b -> 2b -> 3b -> 4b -> 5b -> 6b -> 7b
    // -------------------------------------------------------------------------
    wire [1:0] l1 [0:31];
    wire [2:0] l2 [0:15];
    wire [3:0] l3 [0:7];
    wire [4:0] l4 [0:3];
    wire [5:0] l5 [0:1];
    wire [6:0] xnorpop;

    genvar ii;
    generate
        // Level 1: 32 pairs of single bits
        for (ii = 0; ii < 32; ii = ii + 1) begin : L1
            assign l1[ii] = {1'b0, xnn[2*ii]} + {1'b0, xnn[2*ii+1]};
        end
        // Level 2: 16 pairs of 2-bit sums
        for (ii = 0; ii < 16; ii = ii + 1) begin : L2
            assign l2[ii] = {1'b0, l1[2*ii]} + {1'b0, l1[2*ii+1]};
        end
        // Level 3: 8 pairs of 3-bit sums
        for (ii = 0; ii < 8; ii = ii + 1) begin : L3
            assign l3[ii] = {1'b0, l2[2*ii]} + {1'b0, l2[2*ii+1]};
        end
        // Level 4: 4 pairs of 4-bit sums
        for (ii = 0; ii < 4; ii = ii + 1) begin : L4
            assign l4[ii] = {1'b0, l3[2*ii]} + {1'b0, l3[2*ii+1]};
        end
        // Level 5: 2 pairs of 5-bit sums
        for (ii = 0; ii < 2; ii = ii + 1) begin : L5
            assign l5[ii] = {1'b0, l4[2*ii]} + {1'b0, l4[2*ii+1]};
        end
    endgenerate

    // Level 6: final 7-bit popcount (0..64)
    assign xnorpop = {1'b0, l5[0]} + {1'b0, l5[1]};

    // Bipolar contribution: 2P - N, N=64  =>  range [-64, 64] -> 8-bit signed
    wire signed [7:0] contri = {1'b0, xnorpop, 1'b0} - 8'd64;

    // Register output on valid_in pulse (one-cycle latency)
    always @(posedge clk) begin
        if (rst) begin
            f_output <= 8'sd0;
            done     <= 1'b0;
        end else if (valid_in) begin
            f_output <= contri;
            done     <= 1'b1;
        end else begin
            done <= 1'b0;
        end
    end
endmodule