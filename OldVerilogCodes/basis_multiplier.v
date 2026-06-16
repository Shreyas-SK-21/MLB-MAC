// module basis_multiplier(output [9:0] basis_mult,input [2:0]xnor_popcount,input [3:0]alpha_x,alpha_w);
//     wire [7:0] inter;//intermediates
//     multiplier_P #(.N(4)) mult_0(.o(inter),.a(alpha_x),.b(alpha_w));//multipliers
//     multiplier2_P #(.A_WIDTH(8),.B_WIDTH(3)) mult_1(.o(basis_mult),.a(inter),.b(xnor_popcount));
// endmodule

module basis_multiplier #(
    parameter N = 4
)(
    output [(2*N)+$clog2(N+1)-1:0] basis_mult,
    input  [$clog2(N+1)-1:0] xnor_popcount,
    input  [N-1:0] alpha_x,
    input  [N-1:0] alpha_w
);

    localparam INTER_WIDTH = 2*N;
    localparam POP_WIDTH   = $clog2(N+1);
    localparam OUT_WIDTH   = INTER_WIDTH + POP_WIDTH;

    wire [INTER_WIDTH-1:0] inter;

    // alpha_x × alpha_w
    multiplier_P #(
        .N(N)
    ) mult_0 (
        .o(inter),
        .a(alpha_x),
        .b(alpha_w)
    );

    // (alpha_x × alpha_w) × popcount
    multiplier2_P #(
        .A_WIDTH(INTER_WIDTH),
        .B_WIDTH(POP_WIDTH)
    ) mult_1 (
        .o(basis_mult),
        .a(inter),
        .b(xnor_popcount)
    );

endmodule