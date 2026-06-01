module xnor_popcount_16_bit(output [4:0] xnorpop,input [15:0]a,b);
    wire [3:0] inter1,inter2;//intermediates
    xnor_popcount_8_bit popcount_1(.xnorpop(inter1),.a(a[7:0]),.b(b[7:0]));
    xnor_popcount_8_bit popcount_2(.xnorpop(inter2),.a(a[15:8]),.b(b[15:8]));

    full_adder_4 adder(.sum(xnorpop[3:0]),.carry(xnorpop[4]),.a(inter1),.b(inter2),.cin(1'b0));
endmodule