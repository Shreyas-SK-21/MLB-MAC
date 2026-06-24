module and_popcount_4_bit(
    output reg [3:0] f_output,
    output reg done,
    input [8:0] a, b,
    input clk, rst, valid_in
);
    wire [8:0] xnn = a & b;

    // Explicit Binary Reduction Tree for Area & Timing optimization
    wire [1:0] sum_L1 [0:3];
    wire [2:0] sum_L2 [0:1];
    wire [3:0] final_popcount;

    genvar i;
    generate
        for(i=0; i<4; i=i+1) assign sum_L1[i] = xnn[2*i] + xnn[2*i+1];
        for(i=0; i<2; i=i+1) assign sum_L2[i] = sum_L1[2*i] + sum_L1[2*i+1];
    endgenerate

    assign final_popcount = {1'b0, sum_L2[0]} + {1'b0, sum_L2[1]} + {3'b0, xnn[8]};

    // Pipeline Logic
    reg cycle_cnt;
    always @(posedge clk) begin
        if(rst) begin
            f_output  <= 4'sd0;
            done      <= 1'b0;
            cycle_cnt <= 1'b0;
        end else if(valid_in) begin
            if(cycle_cnt == 1'b0) begin
                f_output  <= final_popcount;
                cycle_cnt <= 1'b0;
                done      <= 1'b1;
            end
        end else begin
            done <= 1'b0;
        end
    end
endmodule
