module reduction_tree_8(output [22:0] sum,input [19:0] a0,a1,a2,a3,a4,a5,a6,a7);
    wire [20:0] L1_0,L1_1,L1_2,L1_3;//inter sum of 21
    wire [21:0] L2_0,L2_1;//inter sum of 22
    wire c2_0,c2_1,c3;//inter_carry

    full_adder_20 adder_L1_0(.sum(L1_0[19:0]),.carry(L1_0[20]),.a(a0),.b(a1),.cin(1'b0));
    full_adder_20 adder_L1_1(.sum(L1_1[19:0]),.carry(L1_1[20]),.a(a2),.b(a3),.cin(1'b0));
    full_adder_20 adder_L1_2(.sum(L1_2[19:0]),.carry(L1_2[20]),.a(a4),.b(a5),.cin(1'b0));
    full_adder_20 adder_L1_3(.sum(L1_3[19:0]),.carry(L1_3[20]),.a(a6),.b(a7),.cin(1'b0));

    full_adder_20 adder_L2_0_0(.sum(L2_0[19:0]),.carry(c2_0),.a(L1_0[19:0]),.b(L1_1[19:0]),.cin(1'b0));
    full_adder adder_L2_0_1(.sum(L2_0[20]),.carry(L2_0[21]),.a(L1_0[20]),.b(L1_1[20]),.cin(c2_0));

    full_adder_20 adder_L2_1_0(.sum(L2_1[19:0]),.carry(c2_1),.a(L1_2[19:0]),.b(L1_3[19:0]),.cin(1'b0));
    full_adder adder_L2_1_1(.sum(L2_1[20]),.carry(L2_1[21]),.a(L1_2[20]),.b(L1_3[20]),.cin(c2_1));

    full_adder_20 adder_L3_0(.sum(sum[19:0]),.carry(c3),.a(L2_0[19:0]),.b(L2_1[19:0]),.cin(1'b0));
    full_adder_2 adder_L3_1(.sum(sum[21:20]),.carry(sum[22]),.a(L2_0[21:20]),.b(L2_1[21:20]),.cin(c3));
endmodule