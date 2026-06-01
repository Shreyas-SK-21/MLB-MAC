module multiplier_16(output [31:0] o, input [15:0]a,b);

    wire [31:0]m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15;

    assign m0 = (b[0]) ? {16'b0,a}:32'b0;
    assign m1 = (b[1]) ? {16'b0,a}<<1:32'b0;
    assign m2 = (b[2]) ? {16'b0,a}<<2:32'b0;
    assign m3 = (b[3]) ? {16'b0,a}<<3:32'b0;
    assign m4 = (b[4]) ? {16'b0,a}<<4:32'b0;
    assign m5 = (b[5]) ? {16'b0,a}<<5:32'b0;
    assign m6 = (b[6]) ? {16'b0,a}<<6:32'b0;
    assign m7 = (b[7]) ? {16'b0,a}<<7:32'b0;
    assign m8 = (b[8]) ? {16'b0,a}<<8:32'b0;
    assign m9 = (b[9]) ? {16'b0,a}<<9:32'b0;
    assign m10 = (b[10]) ? {16'b0,a}<<10:32'b0;
    assign m11 = (b[11]) ? {16'b0,a}<<11:32'b0;
    assign m12 = (b[12]) ? {16'b0,a}<<12:32'b0;
    assign m13 = (b[13]) ? {16'b0,a}<<13:32'b0;
    assign m14 = (b[14]) ? {16'b0,a}<<14:32'b0;
    assign m15 = (b[15]) ? {16'b0,a}<<15:32'b0;

    wire [31:0] inter1,inter2,inter3,inter4,inter5,inter6,inter7,inter8,inter9,inter10,inter11,inter12,inter13,inter14;//intermediates
    wire c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15; //carry_inter
    
    full_adder_32 adder_1(.sum(inter1),.carry(c1),.a(m0),.b(m1),.cin(1'b0));//adders
    full_adder_32 adder_2(.sum(inter2),.carry(c2),.a(inter1),.b(m2),.cin(1'b0));
    full_adder_32 adder_3(.sum(inter3),.carry(c3),.a(inter2),.b(m3),.cin(1'b0));
    full_adder_32 adder_4(.sum(inter4),.carry(c4),.a(inter3),.b(m4),.cin(1'b0));
    full_adder_32 adder_5(.sum(inter5),.carry(c5),.a(inter4),.b(m5),.cin(1'b0));
    full_adder_32 adder_6(.sum(inter6),.carry(c6),.a(inter5),.b(m6),.cin(1'b0));
    full_adder_32 adder_7(.sum(inter7),.carry(c7),.a(inter6),.b(m7),.cin(1'b0));
    full_adder_32 adder_8(.sum(inter8),.carry(c8),.a(inter7),.b(m8),.cin(1'b0));
    full_adder_32 adder_9(.sum(inter9),.carry(c9),.a(inter8),.b(m9),.cin(1'b0));
    full_adder_32 adder_10(.sum(inter10),.carry(c10),.a(inter9),.b(m10),.cin(1'b0));
    full_adder_32 adder_11(.sum(inter11),.carry(c11),.a(inter10),.b(m11),.cin(1'b0));
    full_adder_32 adder_12(.sum(inter12),.carry(c12),.a(inter11),.b(m12),.cin(1'b0));
    full_adder_32 adder_13(.sum(inter13),.carry(c13),.a(inter12),.b(m13),.cin(1'b0));
    full_adder_32 adder_14(.sum(inter14),.carry(c14),.a(inter13),.b(m14),.cin(1'b0));
    full_adder_32 adder_15(.sum(o),.carry(c15),.a(inter14),.b(m15),.cin(1'b0));

endmodule