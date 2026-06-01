module reduction_tree_16(output [40:0] sum,input [36:0] a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15);
    wire [37:0] L1_0,L1_1,L1_2,L1_3,L1_4,L1_5,L1_6,L1_7;
    wire [38:0] L2_0,L2_1,L2_2,L2_3;
    wire [39:0] L3_0,L3_1;
    wire c2_0,c2_1,c2_2,c2_3,c3_0,c3_1,c4;

    full_adder_37 adder_L1_0(.sum(L1_0[36:0]),.carry(L1_0[37]),.a(a0),.b(a1),.cin(1'b0));
    full_adder_37 adder_L1_1(.sum(L1_1[36:0]),.carry(L1_1[37]),.a(a2),.b(a3),.cin(1'b0));
    full_adder_37 adder_L1_2(.sum(L1_2[36:0]),.carry(L1_2[37]),.a(a4),.b(a5),.cin(1'b0));
    full_adder_37 adder_L1_3(.sum(L1_3[36:0]),.carry(L1_3[37]),.a(a6),.b(a7),.cin(1'b0));
    full_adder_37 adder_L1_4(.sum(L1_4[36:0]),.carry(L1_4[37]),.a(a8),.b(a9),.cin(1'b0));
    full_adder_37 adder_L1_5(.sum(L1_5[36:0]),.carry(L1_5[37]),.a(a10),.b(a11),.cin(1'b0));
    full_adder_37 adder_L1_6(.sum(L1_6[36:0]),.carry(L1_6[37]),.a(a12),.b(a13),.cin(1'b0));
    full_adder_37 adder_L1_7(.sum(L1_7[36:0]),.carry(L1_7[37]),.a(a14),.b(a15),.cin(1'b0));

    full_adder_37 adder_L2_0_0(.sum(L2_0[36:0]),.carry(c2_0),.a(L1_0[36:0]),.b(L1_1[36:0]),.cin(1'b0));
    full_adder adder_L2_0_1(.sum(L2_0[37]),.carry(L2_0[38]),.a(L1_0[37]),.b(L1_1[37]),.cin(c2_0));

    full_adder_37 adder_L2_1_0(.sum(L2_1[36:0]),.carry(c2_1),.a(L1_2[36:0]),.b(L1_3[36:0]),.cin(1'b0));
    full_adder adder_L2_1_1(.sum(L2_1[37]),.carry(L2_1[38]),.a(L1_2[37]),.b(L1_3[37]),.cin(c2_1));

    full_adder_37 adder_L2_2_0(.sum(L2_2[36:0]),.carry(c2_2),.a(L1_4[36:0]),.b(L1_5[36:0]),.cin(1'b0));
    full_adder adder_L2_2_1(.sum(L2_2[37]),.carry(L2_2[38]),.a(L1_4[37]),.b(L1_5[37]),.cin(c2_2));

    full_adder_37 adder_L2_3_0(.sum(L2_3[36:0]),.carry(c2_3),.a(L1_6[36:0]),.b(L1_7[36:0]),.cin(1'b0));
    full_adder adder_L2_3_1(.sum(L2_3[37]),.carry(L2_3[38]),.a(L1_6[37]),.b(L1_7[37]),.cin(c2_3));

    full_adder_37 adder_L3_0_0(.sum(L3_0[36:0]),.carry(c3_0),.a(L2_0[36:0]),.b(L2_1[36:0]),.cin(1'b0));
    full_adder_2 adder_L3_0_1(.sum(L3_0[38:37]),.carry(L3_0[39]),.a(L2_0[38:37]),.b(L2_1[38:37]),.cin(c3_0));

    full_adder_37 adder_L3_1_0(.sum(L3_1[36:0]),.carry(c3_1),.a(L2_2[36:0]),.b(L2_3[36:0]),.cin(1'b0));
    full_adder_2 adder_L3_1_1(.sum(L3_1[38:37]),.carry(L3_1[39]),.a(L2_2[38:37]),.b(L2_3[38:37]),.cin(c3_1));

    full_adder_37 adder_L4_0(.sum(sum[36:0]),.carry(c4),.a(L3_0[36:0]),.b(L3_1[36:0]),.cin(1'b0));
    full_adder_3 adder_L4_1(.sum(sum[39:37]),.carry(sum[40]),.a(L3_0[39:37]),.b(L3_1[39:37]),.cin(c4));
endmodule