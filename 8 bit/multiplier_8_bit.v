module multiplier_8(output [15:0] o, input [7:0]a,b);

    wire [15:0]m0,m1,m2,m3,m4,m5,m6,m7;

    assign m0 = (b[0]) ? {8'b0,a}:16'b0;
    assign m1 = (b[1]) ? {8'b0,a}<<1:16'b0;
    assign m2 = (b[2]) ? {8'b0,a}<<2:16'b0;
    assign m3 = (b[3]) ? {8'b0,a}<<3:16'b0;
    assign m4 = (b[4]) ? {8'b0,a}<<4:16'b0;
    assign m5 = (b[5]) ? {8'b0,a}<<5:16'b0;
    assign m6 = (b[6]) ? {8'b0,a}<<6:16'b0;
    assign m7 = (b[7]) ? {8'b0,a}<<7:16'b0;

    wire [15:0] inter1,inter2,inter3,inter4,inter5,inter6;//intermediates
    wire c1,c2,c3,c4,c5,c6,c7; //carry_inter
    // Addition chain using 16-bit adders
    full_adder_16 adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));
    full_adder_16 adder_2(.sum(inter2),.carry(c2),.a(inter1),.b(m2),.cin(1'b0));
    full_adder_16 adder_3(.sum(inter3),.carry(c3),.a(inter2),.b(m3),.cin(1'b0));
    full_adder_16 adder_4(.sum(inter4),.carry(c4),.a(inter3),.b(m4),.cin(1'b0));
    full_adder_16 adder_5(.sum(inter5),.carry(c5),.a(inter4),.b(m5),.cin(1'b0));
    full_adder_16 adder_6(.sum(inter6),.carry(c6),.a(inter5),.b(m6),.cin(1'b0));
    full_adder_16 adder_7(.sum(o),.carry(c7),.a(inter6),.b(m7),.cin(1'b0));
endmodule