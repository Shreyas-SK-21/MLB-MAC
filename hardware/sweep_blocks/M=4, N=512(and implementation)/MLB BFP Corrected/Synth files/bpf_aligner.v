module bfp_aligner (
    input  clk,
    input  [4095:0] fp8_vec,
    output reg [2047:0] aligned_planes, // Reorganized for MLB_4 axi/awi ports
    output reg [3:0]   max_exp
);
    genvar g;

    wire [3:0] exps_w [0:511];
    generate
        for (g=0; g<512; g=g+1) begin : extract_exp
            assign exps_w[g] = fp8_vec[(g*8)+3 +: 4];
        end
    endgenerate

    wire [3:0] maxL1 [0:255];
    wire [3:0] maxL2 [0:127];
    wire [3:0] maxL3 [0:63];
    wire [3:0] maxL4 [0:31];
    wire [3:0] maxL5 [0:15];
    wire [3:0] maxL6 [0:7];
    wire [3:0] maxL7 [0:3];
    wire [3:0] maxL8 [0:1];

    genvar m;
    generate
        for(m=0; m<256; m=m+1) assign maxL1[m] = (exps_w[2*m] > exps_w[2*m+1]) ? exps_w[2*m] : exps_w[2*m+1];
        for(m=0; m<128; m=m+1) assign maxL2[m] = (maxL1[2*m] > maxL1[2*m+1]) ? maxL1[2*m] : maxL1[2*m+1];
        for(m=0; m<64; m=m+1) assign maxL3[m] = (maxL2[2*m] > maxL2[2*m+1]) ? maxL2[2*m] : maxL2[2*m+1];
        for(m=0; m<32; m=m+1) assign maxL4[m] = (maxL3[2*m] > maxL3[2*m+1]) ? maxL3[2*m] : maxL3[2*m+1];
        for(m=0; m<16; m=m+1) assign maxL5[m] = (maxL4[2*m] > maxL4[2*m+1]) ? maxL4[2*m] : maxL4[2*m+1];
        for(m=0; m<8; m=m+1) assign maxL6[m] = (maxL5[2*m] > maxL5[2*m+1]) ? maxL5[2*m] : maxL5[2*m+1];
        for(m=0; m<4; m=m+1) assign maxL7[m] = (maxL6[2*m] > maxL6[2*m+1]) ? maxL6[2*m] : maxL6[2*m+1];
        for(m=0; m<2; m=m+1) assign maxL8[m] = (maxL7[2*m] > maxL7[2*m+1]) ? maxL7[2*m] : maxL7[2*m+1];
    endgenerate

    wire [3:0] max_exp_c = (((maxL8[0]) > (maxL8[1])) ? (maxL8[0]) : (maxL8[1]));

    wire [3:0] mants_w        [0:511];
    wire [3:0] shifted_mant_w [0:511];
    generate
        for (g=0; g<512; g=g+1) begin : align_lane
            assign mants_w[g] = (exps_w[g] != 4'd0) ? {1'b1, fp8_vec[(g*8) +: 3]} : 4'd0;
            assign shifted_mant_w[g] = mants_w[g] >> (max_exp_c - exps_w[g]);
        end
    endgenerate

    wire [2047:0] aligned_planes_c;
    generate
        for (g=0; g<512; g=g+1) begin : corner_turn
            assign aligned_planes_c[g]         = shifted_mant_w[g][0]; // Plane 0 (axi[511:0])
            assign aligned_planes_c[512 + g]  = shifted_mant_w[g][1]; // Plane 1 (axi[1023:512])
            assign aligned_planes_c[1024 + g] = shifted_mant_w[g][2]; // Plane 2 (axi[1535:1024])
            assign aligned_planes_c[1536 + g] = shifted_mant_w[g][3]; // Plane 3 (axi[2047:1536])
        end
    endgenerate

    always @(posedge clk) begin
        aligned_planes <= aligned_planes_c;
        max_exp        <= max_exp_c;
    end
endmodule
