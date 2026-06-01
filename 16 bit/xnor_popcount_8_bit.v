module xnor_popcount_8_bit(output [3:0] xnorpop,input [7:0]a,b);
    wire [2:0] inter1,inter2;//intermediates
    xnor_popcount_4_bit popcount_1(.xnorpop(inter1),.a(a[3:0]),.b(b[3:0]));
    xnor_popcount_4_bit popcount_2(.xnorpop(inter2),.a(a[7:4]),.b(b[7:4]));
    
    full_adder_3 adder(.sum(xnorpop[2:0]),.carry(xnorpop[3]),.a(inter1),.b(inter2),.cin(1'b0));
endmodule