module multiplier_16_4(output [19:0] o,input [15:0]a,input [3:0]b);

    wire [19:0]m0,m1,m2,m3;

    assign m0 = (b[0])? {4'b0,a}:20'b0;
    assign m1 = (b[1])? {4'b0,a}<<1:20'b0;
    assign m2 = (b[2])? {4'b0,a}<<2:20'b0;
    assign m3 = (b[3])? {4'b0,a}<<3:20'b0;

    wire [19:0] inter1,inter2;//intermediates
    wire c1,c2,c3;//carry_inter
    full_adder_20 adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));//20 bit adders
    full_adder_20 adder_2(.sum(inter2),.carry(c2),.a(m2),.b(inter1),.cin(1'b0));
    full_adder_20 adder_3(.sum(o),.carry(c3),.a(m3),.b(inter2),.cin(1'b0));
endmodule