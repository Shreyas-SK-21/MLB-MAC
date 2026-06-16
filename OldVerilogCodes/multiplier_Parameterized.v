module multiplier_P #parameter(N=4) (
    output [2*N-1:0]  o,
    input  [N-1:0]    a,
    input  [N-1:0]    b
    );

    localparam OUT_WIDTH = 2*N;

    wire [OUT_WIDTH-1:0] partial [0:N-1];

    genvar i;
    generate
        for(i = 0; i < N; i = i + 1) begin : gen_partial
            assign partial[i] =
                b[i] ? ({{N{1'b0}}, a} << i)
                     : {OUT_WIDTH{1'b0}};
        end
    endgenerate

    // Running sums
    
    wire [OUT_WIDTH-1:0] sum_stage [0:N-1];
    wire [N-1:0] carry_stage;

    assign sum_stage[0] = partial[0];
    assign carry_stage[0] = 1'b0;

    generate
        for(i = 1; i < N; i = i + 1) begin : gen_adders
            ripple_carry_adder #(.N(OUT_WIDTH)) rca (
                .sum(sum_stage[i]),
                .carry_out(carry_stage[i]),
                .a(sum_stage[i-1]),
                .b(partial[i]),
                .cin(1'b0)
            );
        end
    endgenerate

    assign o = sum_stage[N-1];

endmodule