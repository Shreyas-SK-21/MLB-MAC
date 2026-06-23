`timescale 1ns/1ps
// =============================================================================
// int_mac_64.v  --  64-lane Signed Integer Dot-Product Unit
// =============================================================================
//
// Computes: result = SUM_{k=0}^{63} a[k] * b[k]
//   where a[k], b[k] are 5-bit SIGNED integers (range -15 .. +15)
//   packed LSB-first into 320-bit buses (element k at bits [k*5 +: 5]).
//
// Datapath:
//   - 64 parallel 5b x 5b signed multipliers   -> 10-bit products
//   - 6-level combinational binary adder tree   -> 16-bit sum
//   - Registered output on valid_in             -> done 1 cycle later
//
// Bit-width per tree level (signed, +1 guard bit per level):
//   products : 10-bit  (max |val| = 15*15 = 225)
//   level 1  : 11-bit  (32 nodes)
//   level 2  : 12-bit  (16 nodes)
//   level 3  : 13-bit  ( 8 nodes)
//   level 4  : 14-bit  ( 4 nodes)
//   level 5  : 15-bit  ( 2 nodes)
//   level 6  : 16-bit  ( 1 node)  (max |sum| = 64*225 = 14400 < 2^14)
//
// Pipeline latency: 1 cycle (valid_in -> done)
// =============================================================================

module int_mac_64 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire [319:0] a_flat,   // 64 x 5-bit signed activations
    input  wire [319:0] b_flat,   // 64 x 5-bit signed weights
    output reg  signed [20:0] result,
    output reg         done
);

    // =========================================================================
    // 64 signed multipliers  (5b x 5b -> 10b)
    // =========================================================================
    wire signed [9:0] prod [0:63];
    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : gen_prod
            assign prod[i] = $signed(a_flat[i*5 +: 5]) * $signed(b_flat[i*5 +: 5]);
        end
    endgenerate

    // =========================================================================
    // 6-level combinational adder tree  64 -> 1
    // =========================================================================
    wire signed [10:0] s1 [0:31];
    wire signed [11:0] s2 [0:15];
    wire signed [12:0] s3 [0:7];
    wire signed [13:0] s4 [0:3];
    wire signed [14:0] s5 [0:1];
    wire signed [15:0] s6;

    genvar j;
    generate
        for (j = 0; j < 32; j = j + 1) begin : r1
            assign s1[j] = $signed(prod[2*j]) + $signed(prod[2*j+1]);
        end
        for (j = 0; j < 16; j = j + 1) begin : r2
            assign s2[j] = $signed(s1[2*j]) + $signed(s1[2*j+1]);
        end
        for (j = 0; j < 8; j = j + 1) begin : r3
            assign s3[j] = $signed(s2[2*j]) + $signed(s2[2*j+1]);
        end
        for (j = 0; j < 4; j = j + 1) begin : r4
            assign s4[j] = $signed(s3[2*j]) + $signed(s3[2*j+1]);
        end
        for (j = 0; j < 2; j = j + 1) begin : r5
            assign s5[j] = $signed(s4[2*j]) + $signed(s4[2*j+1]);
        end
    endgenerate
    assign s6 = $signed(s5[0]) + $signed(s5[1]);

    // =========================================================================
    // Register result on valid_in  (latency = 1 cycle)
    // =========================================================================
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            result <= 21'sd0;
        end else if (valid_in) begin
            result <= {{5{s6[15]}}, s6};   // sign-extend 16b -> 21b
            done   <= 1'b1;
        end
    end

endmodule
