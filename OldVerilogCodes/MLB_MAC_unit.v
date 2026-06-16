// module MLB_unit(output [9:0] out,input [3:0] alpha_x,alpha_w,axi,awi);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants

//     wire [2:0] inter;//intermediates
//     xnor_popcount #(.N(4)) xp(.xnorpop(inter),.a(axi),.b(awi));
//     basis_multiplier bm(.basis_mult(out),.xnor_popcount(inter),.alpha_x(alpha_x),.alpha_w(alpha_w));
// endmodule

module MLB_unit #(parameter N = 4)(
    output [(2*N)+$clog2(N+1)-1:0] out,
    input  [N-1:0] alpha_x,
    input  [N-1:0] alpha_w,
    input  [N-1:0] axi,
    input  [N-1:0] awi
);

    wire [$clog2(N+1)-1:0] inter;

    xnor_popcount #(
        .N(N)
    ) xp (
        .count(inter),
        .a(axi),
        .b(awi)
    );

    basis_multiplier #(
        .N(N)
    ) bm (
        .basis_mult(out),
        .xnor_popcount(inter),
        .alpha_x(alpha_x),
        .alpha_w(alpha_w)
    );

endmodule