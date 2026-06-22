module xnor_popcount_4_bit(output reg signed [7:0] signed_output,output reg done,input [63:0]a,b,input clk,rst,valid_in);
    wire [63:0]xnn;
    assign xnn=~(a^b);
    reg [6:0]xnorpop;
    integer i;
    always @(*)begin
        xnorpop=7'b0;
        for(i=0;i<64;i=i+1) begin
            xnorpop=xnorpop+xnn[i];
        end
    end
    wire signed [8:0]contri;
    assign contri={1'b0,xnorpop,1'b0}-9'd64;//2P-N; N=64
    reg cycle_cnt;
    reg signed [8:0]acc;
    always @(posedge clk)begin
        if(rst)begin
            signed_output<=9'sd0;
            done<=1'b0;
            cycle_cnt<=1'b0;
            acc<=9'sd0;
        end 
        else if(valid_in)begin
            if(cycle_cnt==1'b0) begin
                signed_output<=contri;
                cycle_cnt<=1'b0;
                done<=1'b1;//Popcount complete for K=64
            end
        end 
        else begin
            done<=1'b0;
        end
    end
endmodule