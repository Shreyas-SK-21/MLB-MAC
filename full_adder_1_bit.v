// module full_adder(output sum,carry, input a,b,cin);
//     xor x1(sum,a,b,cin);
//     assign carry = (a&b)|(a&cin)|(b&cin);
// endmodule

module ripple_carry_adder #(parameter N = 4) (
    output [N-1:0] sum,
    output         carry_out,
    input  [N-1:0] a, b,
    input          cin
);
    wire [N:0] carry;
    assign carry[0] = cin;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : fa_stage
            full_adder fa (
                .sum   (sum[i]),
                .carry (carry[i+1]),
                .a     (a[i]),
                .b     (b[i]),
                .cin   (carry[i])
            );
        end
    endgenerate

    assign carry_out = carry[N];
endmodule


module full_adder (
    output sum, carry,
    input  a, b, cin 
    );
    xor x1 (sum, a, b, cin);
    assign carry = (a & b) | (a & cin) | (b & cin);
endmodule