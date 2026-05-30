module xnor_popcount_4_bit(output [2:0] xnorpop,input [3:0]a,b);
    wire [3:0] xnn;
    wire s0,s1,carry;
    wire [1:0] c,s2;

    xnor xn0(xnn[0],a[0],b[0]);//xnors output
    xnor xn1(xnn[1],a[1],b[1]);
    xnor xn2(xnn[2],a[2],b[2]);
    xnor xn3(xnn[3],a[3],b[3]);
    full_adder adder1(.sum(s0),.carry(c[0]),.cin(1'b0),.a(xnn[0]),.b(xnn[1]));//summing using 1 bit adders
    full_adder adder2(.sum(s1),.carry(c[1]),.cin(1'b0),.a(xnn[2]),.b(xnn[3]));

    full_adder_2 adder3(.sum(s2),.carry(carry),.a({c[0],s0}),.b({c[1],s1}),.cin(1'b0));//summing using the 2 bit adders
    assign xnorpop = {carry,s2};
endmodule

module xnor_popcount #(parameter N = 4) (
    output [$clog2(N+1)-1:0] count,
    input  [N-1:0] a, b
);
    wire [N-1:0] xnn;
    wire [log(N):0] s;
    wire [log(N):0] c;

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : xnor_stage
            xnor xn (xnn[i], a[i], b[i]);
        end
    endgenerate

    ripple_carry_adder #(.N(4)) adder1

    
