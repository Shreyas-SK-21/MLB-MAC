module bfp_aligner (
    input  clk,
    input  [511:0] fp8_vec,
    output reg [255:0] aligned_planes, // Reorganized for MLB_4 axi/awi ports
    output reg [3:0]   max_exp
);
    genvar g;

    wire [3:0] exps_w [0:63];
    generate
        for (g=0; g<64; g=g+1) begin : extract_exp
            assign exps_w[g] = fp8_vec[(g*8)+3 +: 4];
        end
    endgenerate

    wire [3:0] maxL1 [0:31];
    wire [3:0] maxL2 [0:15];
    wire [3:0] maxL3 [0:7];
    wire [3:0] maxL4 [0:3];
    wire [3:0] maxL5 [0:1];

    genvar m;
    generate
        for(m=0; m<32; m=m+1) assign maxL1[m] = (exps_w[2*m] > exps_w[2*m+1]) ? exps_w[2*m] : exps_w[2*m+1];
        for(m=0; m<16; m=m+1) assign maxL2[m] = (maxL1[2*m] > maxL1[2*m+1]) ? maxL1[2*m] : maxL1[2*m+1];
        for(m=0; m<8; m=m+1) assign maxL3[m] = (maxL2[2*m] > maxL2[2*m+1]) ? maxL2[2*m] : maxL2[2*m+1];
        for(m=0; m<4; m=m+1) assign maxL4[m] = (maxL3[2*m] > maxL3[2*m+1]) ? maxL3[2*m] : maxL3[2*m+1];
        for(m=0; m<2; m=m+1) assign maxL5[m] = (maxL4[2*m] > maxL4[2*m+1]) ? maxL4[2*m] : maxL4[2*m+1];
    endgenerate

    wire [3:0] max_exp_c = (((maxL5[0]) > (maxL5[1])) ? (maxL5[0]) : (maxL5[1]));

    wire [3:0] mants_w        [0:63];
    wire [3:0] shifted_mant_w [0:63];
    generate
        for (g=0; g<64; g=g+1) begin : align_lane
            assign mants_w[g] = (exps_w[g] != 4'd0) ? {1'b1, fp8_vec[(g*8) +: 3]} : 4'd0;
            assign shifted_mant_w[g] = mants_w[g] >> (max_exp_c - exps_w[g]);
        end
    endgenerate

    wire [255:0] aligned_planes_c;
    generate
        for (g=0; g<64; g=g+1) begin : corner_turn
            assign aligned_planes_c[g]         = shifted_mant_w[g][0]; // Plane 0 (axi[63:0])
            assign aligned_planes_c[64 + g]  = shifted_mant_w[g][1]; // Plane 1 (axi[127:64])
            assign aligned_planes_c[128 + g] = shifted_mant_w[g][2]; // Plane 2 (axi[191:128])
            assign aligned_planes_c[192 + g] = shifted_mant_w[g][3]; // Plane 3 (axi[255:192])
        end
    endgenerate

    always @(posedge clk) begin
        aligned_planes <= aligned_planes_c;
        max_exp        <= max_exp_c;
    end
endmodule
