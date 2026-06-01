// module multiplier_8_3(output [9:0] o,input [7:0]a,input [2:0]b);

//     wire [9:0]m0,m1,m2;

//     assign m0 = (b[0])? {2'b00,a}:10'b0;
//     assign m1 = (b[1])? {2'b00,a}<<1:10'b0;
//     assign m2 = (b[2])? {2'b00,a}<<2:10'b0;

//     wire [9:0] inter1;//intermediates
//     wire c1,c2;//carry_inter
//     ripple_carry_adder #(.N(10)) adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));//10 bit adders
//     ripple_carry_adder #(.N(10)) adder_2(.sum(o),.carry(c2),.a(m2),.b(inter1),.cin(1'b0));
// endmodule

module multiplier #(
    parameter A_WIDTH = 4,
    parameter B_WIDTH = 4
)(
    output [A_WIDTH+B_WIDTH-1:0] o,
    input  [A_WIDTH-1:0] a,
    input  [B_WIDTH-1:0] b
);

    localparam OUT_WIDTH = A_WIDTH + B_WIDTH;

    // Partial products
    wire [OUT_WIDTH-1:0] partial [0:B_WIDTH-1];

    genvar i;
    generate
        for(i = 0; i < B_WIDTH; i = i + 1) begin : gen_partial
            assign partial[i] =
                b[i] ? ({{B_WIDTH{1'b0}},a} << i)
                     : {OUT_WIDTH{1'b0}};
        end
    endgenerate

    // Running sums
    wire [OUT_WIDTH-1:0] sum_stage [0:B_WIDTH-1];
    wire [B_WIDTH-1:0] carry_stage;

    assign sum_stage[0] = partial[0];
    assign carry_stage[0] = 1'b0;

    generate
        for(i = 1; i < B_WIDTH; i = i + 1) begin : gen_adders
            ripple_carry_adder #(.N(OUT_WIDTH)) rca (
                .sum(sum_stage[i]),
                .carry(carry_stage[i]),
                .a(sum_stage[i-1]),
                .b(partial[i]),
                .cin(1'b0)
            );
        end
    endgenerate

    assign o = sum_stage[B_WIDTH-1];

endmodule