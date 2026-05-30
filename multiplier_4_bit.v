module multiplier_4(output [7:0] o,input [3:0]a,b);

    wire [7:0]m0,m1,m2,m3;

    assign m0 = (b[0])? {4'b0000,a}:8'b0;
    assign m1 = (b[1])? {4'b0000,a}<<1:8'b0;
    assign m2 = (b[2])? {4'b0000,a}<<2:8'b0;
    assign m3 = (b[3])? {4'b0000,a}<<3:8'b0;

    wire [7:0] inter1,inter2;//intermediates
    wire c1,c2,c3;//carry_inter
    ripple_carry_adder #(.N(8)) adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));//8 bit adders
    ripple_carry_adder #(.N(8)) adder_2(.sum(inter2),.carry(c2),.a(inter1),.b(m2),.cin(1'b0));
    ripple_carry_adder #(.N(8)) adder_3(.sum(o),.carry(c3),.a(inter2),.b(m3),.cin(1'b0));
endmodule