module reduction_tree_4(output [11:0] sum,input [9:0]a0,a1,a2,a3);
    wire c1;//inter carry
    wire [10:0] s1,s2;//inter sum

    ripple_carry_adder #(.N(10)) adder1(.sum(s1[9:0]),.carry(s1[10]),.a(a0),.b(a1),.cin(1'b0));//10 bit adders
    ripple_carry_adder #(.N(10)) adder2(.sum(s2[9:0]),.carry(s2[10]),.a(a2),.b(a3),.cin(1'b0));
    ripple_carry_adder #(.N(10)) adder3(.sum(sum[9:0]),.carry(c1),.a(s1[9:0]),.b(s2[9:0]),.cin(1'b0));
    ripple_carry_adder #(.N(10)) adder4(.sum(sum[10]),.carry(sum[11]),.a(s1[10]),.b(s2[10]),.cin(c1));
endmodule

// yet to parameterize this part