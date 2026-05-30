module multiplier_8_3(output [9:0] o,input [7:0]a,input [2:0]b);

    wire [9:0]m0,m1,m2;

    assign m0 = (b[0])? {2'b00,a}:10'b0;
    assign m1 = (b[1])? {2'b00,a}<<1:10'b0;
    assign m2 = (b[2])? {2'b00,a}<<2:10'b0;

    wire [9:0] inter1;//intermediates
    wire c1,c2;//carry_inter
    full_adder_10 adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));//10 bit adders
    full_adder_10 adder_2(.sum(o),.carry(c2),.a(m2),.b(inter1),.cin(1'b0));
endmodule