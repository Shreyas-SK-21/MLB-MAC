module xnor_popcount_4_bit(output signed [7:0] signed_output,input [63:0]a,b);
    wire [63:0]xnn;
    reg [7:0]xnorpop;
    assign xnn=~(a^b);
    integer i;
    always @(*) begin
    xnorpop=8'b0;
    for(i=0;i<64;i=i+1) begin
        xnorpop=xnorpop+xnn[i];
        end
    end
    assign signed_output=(xnorpop<<1)-8'd64;//2P-N; N=64
endmodule