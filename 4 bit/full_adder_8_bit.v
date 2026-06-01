module full_adder_8(output [7:0] sum,output carry, input [7:0] a,b, input cin);
    wire inter1,inter2,inter3,inter4,inter5,inter6,inter7;//intermediates
    full_adder adder_1(.sum(sum[0]),.carry(inter1),.a(a[0]),.b(b[0]),.cin(cin));
    full_adder adder_2(.sum(sum[1]),.carry(inter2),.a(a[1]),.b(b[1]),.cin(inter1));
    full_adder adder_3(.sum(sum[2]),.carry(inter3),.a(a[2]),.b(b[2]),.cin(inter2));
    full_adder adder_4(.sum(sum[3]),.carry(inter4),.a(a[3]),.b(b[3]),.cin(inter3));
    full_adder adder_5(.sum(sum[4]),.carry(inter5),.a(a[4]),.b(b[4]),.cin(inter4));
    full_adder adder_6(.sum(sum[5]),.carry(inter6),.a(a[5]),.b(b[5]),.cin(inter5));
    full_adder adder_7(.sum(sum[6]),.carry(inter7),.a(a[6]),.b(b[6]),.cin(inter6));
    full_adder adder_8(.sum(sum[7]),.carry(carry),.a(a[7]),.b(b[7]),.cin(inter7));
endmodule