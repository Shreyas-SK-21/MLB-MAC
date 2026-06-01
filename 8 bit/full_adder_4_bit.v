module full_adder_4(output [3:0] sum,output carry, input [3:0] a,b, input cin);
    wire inter1,inter2,inter3;//intermediates
    full_adder adder_1(.sum(sum[0]),.carry(inter1),.a(a[0]),.b(b[0]),.cin(cin));
    full_adder adder_2(.sum(sum[1]),.carry(inter2),.a(a[1]),.b(b[1]),.cin(inter1));
    full_adder adder_3(.sum(sum[2]),.carry(inter3),.a(a[2]),.b(b[2]),.cin(inter2));
    full_adder adder_4(.sum(sum[3]),.carry(carry),.a(a[3]),.b(b[3]),.cin(inter3));
endmodule