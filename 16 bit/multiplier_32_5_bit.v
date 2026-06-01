module multiplier_32_5(output [36:0] o, input [31:0]a, input [4:0]b);

    wire [36:0]m0,m1,m2,m3,m4;

    assign m0 = (b[0]) ? {5'b0,a}:37'b0;
    assign m1 = (b[1]) ? {5'b0,a}<<1:37'b0;
    assign m2 = (b[2]) ? {5'b0,a}<<2:37'b0;
    assign m3 = (b[3]) ? {5'b0,a}<<3:37'b0;
    assign m4 = (b[4]) ? {5'b0,a}<<4:37'b0;

    wire [36:0] inter1,inter2,inter3;//intermediates
    wire c1,c2,c3,c4;//carry_inter
    full_adder_37 adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));//37 bit adders
    full_adder_37 adder_2(.sum(inter2),.carry(c2),.a(m2),.b(inter1),.cin(1'b0));
    full_adder_37 adder_3(.sum(inter3),.carry(c3),.a(m3),.b(inter2),.cin(1'b0));
    full_adder_37 adder_4(.sum(o),.carry(c4),.a(m4),.b(inter3),.cin(1'b0));

endmodule