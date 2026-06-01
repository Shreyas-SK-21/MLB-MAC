module full_adder_3(output [2:0] sum,output carry, input [2:0] a,b, input cin);
    wire inter1,inter2;//intermediates
    full_adder adder_1(.sum(sum[0]),.carry(inter1),.a(a[0]),.b(b[0]),.cin(cin));
    full_adder adder_2(.sum(sum[1]),.carry(inter2),.a(a[1]),.b(b[1]),.cin(inter1));
    full_adder adder_3(.sum(sum[2]),.carry(carry),.a(a[2]),.b(b[2]),.cin(inter2));
endmodule