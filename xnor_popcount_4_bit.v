// module xnor_popcount_4_bit(output [2:0] xnorpop,input [3:0]a,b);
//     wire [3:0] xnn;
//     wire s0,s1,carry;
//     wire [1:0] c,s2;

//     xnor xn0(xnn[0],a[0],b[0]);//xnors output
//     xnor xn1(xnn[1],a[1],b[1]);
//     xnor xn2(xnn[2],a[2],b[2]);
//     xnor xn3(xnn[3],a[3],b[3]);
//     full_adder adder1(.sum(s0),.carry(c[0]),.cin(1'b0),.a(xnn[0]),.b(xnn[1]));//summing using 1 bit adders
//     full_adder adder2(.sum(s1),.carry(c[1]),.cin(1'b0),.a(xnn[2]),.b(xnn[3]));

//     full_adder_2 adder3(.sum(s2),.carry(carry),.a({c[0],s0}),.b({c[1],s1}),.cin(1'b0));//summing using the 2 bit adders
//     assign xnorpop = {carry,s2};
// endmodule

module xnor_popcount #(parameter N = 4)(
    output [$clog2(N):0] count,
    input  [N-1:0] a, b
);
    localparam WIDTH  = $clog2(N) + 1;
    localparam LEVELS = $clog2(N) + 1;

    // Flat array: level*N + slot  →  WIDTH bits
    wire [WIDTH-1:0] partial [0:(LEVELS+1)*N - 1];

    // Helper: index into flat array
    // partial_idx(lvl, slot) = lvl*N + slot
    // (used inline below)

    // --- Stage 0: XNOR + zero-extend ---
    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin : xnor_load
            wire xnor_bit;
            xnor xn (xnor_bit, a[k], b[k]);
            assign partial[0*N + k] = {{(WIDTH-1){1'b0}}, xnor_bit};
        end
    endgenerate

    // --- Tree levels ---
    genvar lvl, j;
    generate
        for (lvl = 0; lvl < LEVELS; lvl = lvl + 1) begin : tree_level
            for (j = 0; j < N; j = j + 1) begin : tree_node
                // Number of valid entries at this level = ceil(N / 2^lvl)
                if (2*j+1 < ((N + (1<<lvl) - 1) >> lvl)) begin : add_pair
                    assign partial[(lvl+1)*N + j] =
                        partial[lvl*N + 2*j] + partial[lvl*N + 2*j+1];
                end else if (2*j < ((N + (1<<lvl) - 1) >> lvl)) begin : passthrough
                    assign partial[(lvl+1)*N + j] = partial[lvl*N + 2*j];
                end else begin : zero_fill
                    assign partial[(lvl+1)*N + j] = {WIDTH{1'b0}};
                end
            end
        end
    endgenerate

    assign count = partial[LEVELS*N][$clog2(N):0];

endmodule

    
