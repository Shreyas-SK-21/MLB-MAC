module MLB_4(output reg signed [36:0] mlb,output reg done,input [31:0] alpha_x,alpha_w,input [255:0] axi,awi,input [12:0]K,input clk,rst,valid_in);//xnor_popcount of axi and awi and alpha_x and alpha_w is the matrix constants
    
    // Time-multiplexing control registers
    reg [1:0] state;
    wire [7:0] multiplexed_alpha_x = alpha_x[state*8 +: 8];
    wire [63:0] multiplexed_axi     = axi[state*64 +: 64];

    genvar j;
    wire signed [32:0] unit_out[0:3];
    wire [3:0] sub_unit_done;
    generate
        for(j=0;j<4;j=j+1) begin
            MLB_unit u_j(.out(unit_out[j]),.done(sub_unit_done[j]),.alpha_x(multiplexed_alpha_x),.alpha_w(alpha_w[j*8+7:j*8]),.axi(multiplexed_axi),.awi(awi[j*64+63:64*j]),.clk(clk),.rst(rst),.valid_in(valid_in),.K(K));
        end
    endgenerate

    // Maintaining original structure using reg arrays to accumulate across cycles
    reg signed [32:0] out[0:15];
    reg [15:0]unit_done;
    integer login;

    always @(posedge clk) begin
        if(rst) begin
            state <= 2'd0;
            unit_done <= 16'b0;
            for(login=0; login<16; login=login+1) out[login] <= 33'sd0;
        end
        else if(&sub_unit_done) begin
            // Capture row output chunks into their original positions
            out[state*4]   <= unit_out[0];
            out[state*4+1] <= unit_out[1];
            out[state*4+2] <= unit_out[2];
            out[state*4+3] <= unit_out[3];
            
            unit_done[state*4 +: 4] <= sub_unit_done;
            state <= state + 1'b1;
        end
        else if(&unit_done) begin
            unit_done <= 16'b0; // Reset tracking register after calculation cycle completes
        end
    end

    // Reduction tree
    wire signed [33:0]s00,s01,s02,s03,s04,s05,s06,s07;
    wire signed [34:0]s10,s11,s12,s13;
    wire signed [35:0]s20,s21;
    wire signed [36:0]s30;

    assign s00=out[0]+out[1];
    assign s01=out[2]+out[3];
    assign s02=out[4]+out[5];
    assign s03=out[6]+out[7];   
    assign s04=out[8]+out[9];
    assign s05=out[10]+out[11];
    assign s06=out[12]+out[13];
    assign s07=out[14]+out[15];

    assign s10=s00+s01;
    assign s11=s02+s03;
    assign s12=s04+s05;
    assign s13=s06+s07;

    assign s20=s10+s11;
    assign s21=s12+s13;

    assign s30=s20+s21;
    always @(posedge clk) begin
        done<=1'b0;
        if(rst) begin
            mlb<=37'sd0;
            done<=1'b0;
        end
        else if(&unit_done) begin
            mlb<=s30;
            done<=1'b1;
        end
    end
endmodule