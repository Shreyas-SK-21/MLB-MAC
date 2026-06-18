module int_mac_M3_N64 (
    input                        clk,
    input                        rst,
    input                        load,     

    input  signed [191:0]        a_flat,   
    input  signed [191:0]        b_flat,   

    input         [2:0]          alpha_x,  
    input         [2:0]          alpha_w,  
    input  signed [5:0]          beta_xw,  

    output signed [20:0]         result    
);
wire signed [5:0] product [0:63];
reg  signed [7:0] acc [0:63];

genvar i;
generate
    for (i = 0; i < 64; i = i + 1) begin : gen_mac_lane

        assign product[i] = $signed(a_flat[3*i +: 3])   // xd_i
                           * $signed(b_flat[3*i +: 3]);  // wd_i

        always @(posedge clk) begin
            if (rst)
                acc[i] <= 8'sd0;
            else if (load)
                acc[i] <= acc[i] + {{2{product[i][5]}}, product[i]};  // sign-extend 6→8 then add
        end

    end
endgenerate

wire signed [8:0]  s1 [0:31];
wire signed [9:0]  s2 [0:15];
wire signed [10:0] s3 [0:7];
wire signed [11:0] s4 [0:3];
wire signed [12:0] s5 [0:1];
wire signed [13:0] s6;

genvar j;
generate
    for (j = 0; j < 32; j = j + 1) begin : gen_r1
        assign s1[j] = $signed(acc[2*j])   + $signed(acc[2*j+1]);
    end
    for (j = 0; j < 16; j = j + 1) begin : gen_r2
        assign s2[j] = $signed(s1[2*j])    + $signed(s1[2*j+1]);
    end
    for (j = 0; j < 8;  j = j + 1) begin : gen_r3
        assign s3[j] = $signed(s2[2*j])    + $signed(s2[2*j+1]);
    end
    for (j = 0; j < 4;  j = j + 1) begin : gen_r4
        assign s4[j] = $signed(s3[2*j])    + $signed(s3[2*j+1]);
    end
    for (j = 0; j < 2;  j = j + 1) begin : gen_r5
        assign s5[j] = $signed(s4[2*j])    + $signed(s4[2*j+1]);
    end
endgenerate

assign s6 = $signed(s5[0]) + $signed(s5[1]);


wire [5:0]          alpha_prod;    
wire signed [6:0]   alpha_prod_s;  
wire signed [20:0]  scaled;        

assign alpha_prod   = alpha_x * alpha_w;
assign alpha_prod_s = {1'b0, alpha_prod};

assign scaled = $signed(s6) * $signed(alpha_prod_s);


assign result = scaled + {{15{beta_xw[5]}}, beta_xw};

endmodule
