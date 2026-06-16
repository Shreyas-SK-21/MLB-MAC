module xnor_popcount_4_bit(output reg signed [15:0] signed_output,output reg done,input [63:0]a,b,input [12:0]K,input clk,rst,valid_in);
    wire [63:0] xnn;
    assign xnn=~(a^b);
//adding the 64 bit output 
    reg [6:0] xnorpop;
    integer i;
    always @(*) begin
        xnorpop=6'b0;
        for(i=0;i<64;i=i+1) begin
            xnorpop=xnorpop+xnn[i];
        end
    end
    wire signed [8:0] contri;
    assign contri={1'b0,xnorpop,1'b0}-8'd64;//2P-N, N=64
    reg signed [15:0] acc;
    reg [6:0] cycle_count;
    wire [6:0] cycles_needed;
    assign cycles_needed=K[12:6];//K>>6,(K/N),N=64

    always @(posedge clk) begin
        done<=1'b0;
        if(rst) begin
            acc <= 16'sd0;
            cycle_count <= 7'd0;
            signed_output <= 16'sd0;
        end 
        else if(valid_in) begin//prints if loaded
            if(cycle_count==cycles_needed-1) begin//completed
                signed_output <= acc + contri;
                done <= 1'b1;
                acc <= 16'sd0;
                cycle_count <= 7'd0;
            end 
            else begin//accumulating 
                acc <= acc + contri;
                cycle_count <= cycle_count + 1;
            end
        end
    end
endmodule