module xnor_popcount_3_bit (
    output reg signed [15:0] signed_output,  // range -K..+K
    output reg               done,
    input      [63:0]        a, b,
    input      [12:0]        K,              // dot product size,vmultiple of 64
    input                    clk,
    input                    rst,
    input                    valid_in
);

    // ----------------------------------------------------------
    // Stage 1: XNOR
    // ----------------------------------------------------------
    wire [63:0] xnn;
    assign xnn = ~(a ^ b);

    // ----------------------------------------------------------
    // Stage 2: Structural binary reduction tree (6 levels)
    // N=64 is a design-time constant → Yosys infers narrow adders
    // P_cycle range: 0..64, fits in 7 bits
    // ----------------------------------------------------------
    wire [1:0] l0 [0:31];
    wire [2:0] l1 [0:15];
    wire [3:0] l2 [0:7];
    wire [4:0] l3 [0:3];
    wire [5:0] l4 [0:1];
    wire [6:0] P_cycle;

    genvar k;
    generate
        for(k=0;k<32;k=k+1) assign l0[k] = {1'b0,xnn[2*k]}  + {1'b0,xnn[2*k+1]};
        for(k=0;k<16;k=k+1) assign l1[k] = {1'b0,l0[2*k]}   + {1'b0,l0[2*k+1]};
        for(k=0;k<8; k=k+1) assign l2[k] = {1'b0,l1[2*k]}   + {1'b0,l1[2*k+1]};
        for(k=0;k<4; k=k+1) assign l3[k] = {1'b0,l2[2*k]}   + {1'b0,l2[2*k+1]};
        for(k=0;k<2; k=k+1) assign l4[k] = {1'b0,l3[2*k]}   + {1'b0,l3[2*k+1]};
    endgenerate
    assign P_cycle = {1'b0,l4[0]} + {1'b0,l4[1]};
    wire signed [7:0] contrib;
    assign contrib = {1'b0, P_cycle, 1'b0} - 8'd64;   // 2P - N

    reg signed [15:0] acc;
    reg        [6:0]  cycle_count;          // max K/64 = 8192/64 = 128

    wire [6:0] cycles_needed;
    assign cycles_needed = K[12:6];         // K >> 6  (K/64)

    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            acc         <= 16'sd0;
            cycle_count <= 7'd0;
            signed_output <= 16'sd0;
        end 
        else if (valid_in) begin
            if (cycle_count == cycles_needed - 1) begin
                signed_output <= acc + contrib;
                done          <= 1'b1;
                acc           <= 16'sd0;
                cycle_count   <= 7'd0;
            end else begin
                acc         <= acc + contrib;
                cycle_count <= cycle_count + 1;
            end
        end
    end

endmodule