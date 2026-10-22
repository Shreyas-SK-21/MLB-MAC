
module int_mac_9 (
    input clk,
    input rst,
    input load,

    input [35:0] a_flat,    // 9 x 4-bit UNSIGNED activations
    input [35:0] b_flat,    // 9 x 4-bit UNSIGNED weights
    input [8:0]  sign_flat, // 9 x 1-bit product sign (sign_x XOR sign_w)

    input [3:0]        alpha_x,
    input [3:0]        alpha_w,
    input signed [7:0] beta_xw,

    output signed [20:0] result
);

localparam GROUP = 16;
localparam NGRP  = (9 + GROUP - 1) / GROUP;

wire rst_buf  [0:NGRP-1];
wire load_buf [0:NGRP-1];

genvar g;
generate
    for (g = 0; g < NGRP; g = g + 1) begin : gen_ctrl_buf
        assign rst_buf[g]  = rst;
        assign load_buf[g] = load;
    end
endgenerate

wire [7:0]         product       [0:8]; // unsigned magnitude product
wire signed [8:0]  signed_product[0:8]; // sign applied after multiply
reg  signed [10:0] acc           [0:8]; // accumulator, 11-bit for M=4

genvar i;
generate
    for (i = 0; i < 9; i = i + 1) begin : gen_mac_lane


        assign product[i] = a_flat[4*i +: 4] * b_flat[4*i +: 4];
        assign signed_product[i] = sign_flat[i]
                                    ? -$signed({1'b0, product[i]})
                                    :  $signed({1'b0, product[i]});
        always @(posedge clk) begin
            if (rst_buf[i/GROUP])
                acc[i] <= 11'sd0;
            else if (load_buf[i/GROUP])
                acc[i] <= acc[i] + {{2{signed_product[i][8]}}, signed_product[i]};
        end

    end
endgenerate


    wire signed [11:0] s1 [0:3];
    wire signed [12:0] s2 [0:1];

    genvar j;
    generate
        for (j = 0; j < 4; j = j + 1) begin : gen_r1
            assign s1[j] = $signed(acc[2*j]) + $signed(acc[2*j+1]);
        end
        for (j = 0; j < 2; j = j + 1) begin : gen_r2
            assign s2[j] = $signed(s1[2*j]) + $signed(s1[2*j+1]);
        end
    endgenerate

wire signed [13:0] s_final;
assign s_final = $signed(s2[0]) + $signed(s2[1]) + $signed({{{3{{acc[8][10]}}}}, acc[8]});


wire [7:0]        alpha_prod;   // ax x aw (unsigned 8-bit)
wire signed [8:0] alpha_prod_s; // zero-extended to signed
wire signed [22:0] scaled;

assign alpha_prod   = alpha_x * alpha_w;
assign alpha_prod_s = {1'b0, alpha_prod};
assign scaled       = $signed(s_final) * $signed(alpha_prod_s);

assign result = scaled[20:0] + {{13{beta_xw[7]}}, beta_xw};

endmodule
