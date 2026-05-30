module full_adder_2(output [1:0] sum,output carry, input [1:0] a,b, input cin);
    wire inter;
    full_adder adder_1(.sum(sum[0]),.carry(inter),.a(a[0]),.b(b[0]),.cin(cin));
    full_adder adder_2(.sum(sum[1]),.carry(carry),.a(a[1]),.b(b[1]),.cin(inter));
endmodule