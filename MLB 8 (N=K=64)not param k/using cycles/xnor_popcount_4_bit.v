module xnor_popcount_4_bit(output reg signed [7:0] signed_output,output reg done,input [63:0]a,b,input clk,rst,valid_in);
    wire [63:0] xnn;
    assign xnn=~(a^b);
//adding the 64 bit output 
    reg [6:0] xnorpop;
    integer i;
    always @(*) begin
        xnorpop=7'b0;
        for(i=0;i<64;i=i+1) begin
            xnorpop=xnorpop+xnn[i];
        end
    end
    wire signed [7:0] contri;
    assign contri={1'b0,xnorpop,1'b0}-8'd64;//2P-N, N=64
    always @(posedge clk) begin
        done<=1'b0;
        if(rst) begin
            signed_output <= 8'sd0;
        end 
        else if(valid_in) begin//prints if loaded
                signed_output <= contri;
                done <= 1'b1;
        end
    end
endmodule