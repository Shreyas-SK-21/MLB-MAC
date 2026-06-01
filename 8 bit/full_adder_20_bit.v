module full_adder_20(output [19:0] sum,output carry, input [19:0] a,b, input cin);
    wire inter1,inter2,inter3,inter4,inter5,inter6,inter7,inter8,inter9,inter10,inter11,inter12,inter13,inter14,inter15,inter16,inter17,inter18,inter19;//intermediates
    full_adder adder_1(.sum(sum[0]),.carry(inter1),.a(a[0]),.b(b[0]),.cin(cin));
    full_adder adder_2(.sum(sum[1]),.carry(inter2),.a(a[1]),.b(b[1]),.cin(inter1));
    full_adder adder_3(.sum(sum[2]),.carry(inter3),.a(a[2]),.b(b[2]),.cin(inter2));
    full_adder adder_4(.sum(sum[3]),.carry(inter4),.a(a[3]),.b(b[3]),.cin(inter3));
    full_adder adder_5(.sum(sum[4]),.carry(inter5),.a(a[4]),.b(b[4]),.cin(inter4));
    full_adder adder_6(.sum(sum[5]),.carry(inter6),.a(a[5]),.b(b[5]),.cin(inter5));
    full_adder adder_7(.sum(sum[6]),.carry(inter7),.a(a[6]),.b(b[6]),.cin(inter6));
    full_adder adder_8(.sum(sum[7]),.carry(inter8),.a(a[7]),.b(b[7]),.cin(inter7));
    full_adder adder_9(.sum(sum[8]),.carry(inter9),.a(a[8]),.b(b[8]),.cin(inter8));
    full_adder adder_10(.sum(sum[9]),.carry(inter10),.a(a[9]),.b(b[9]),.cin(inter9));
    full_adder adder_11(.sum(sum[10]),.carry(inter11),.a(a[10]),.b(b[10]),.cin(inter10));
    full_adder adder_12(.sum(sum[11]),.carry(inter12),.a(a[11]),.b(b[11]),.cin(inter11));
    full_adder adder_13(.sum(sum[12]),.carry(inter13),.a(a[12]),.b(b[12]),.cin(inter12));
    full_adder adder_14(.sum(sum[13]),.carry(inter14),.a(a[13]),.b(b[13]),.cin(inter13));
    full_adder adder_15(.sum(sum[14]),.carry(inter15),.a(a[14]),.b(b[14]),.cin(inter14));
    full_adder adder_16(.sum(sum[15]),.carry(inter16),.a(a[15]),.b(b[15]),.cin(inter15));
    full_adder adder_17(.sum(sum[16]),.carry(inter17),.a(a[16]),.b(b[16]),.cin(inter16));
    full_adder adder_18(.sum(sum[17]),.carry(inter18),.a(a[17]),.b(b[17]),.cin(inter17));
    full_adder adder_19(.sum(sum[18]),.carry(inter19),.a(a[18]),.b(b[18]),.cin(inter18));
    full_adder adder_20(.sum(sum[19]),.carry(carry),.a(a[19]),.b(b[19]),.cin(inter19));
endmodule