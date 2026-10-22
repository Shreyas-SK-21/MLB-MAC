module bfp_aligner (
    input  clk,
    input  [71:0] fp8_vec,
    output reg [35:0] aligned_planes, // Reorganized for MLB_4 axi/awi ports
    output reg [3:0]   max_exp
);
    genvar g;

    wire [3:0] exps_w [0:8];
    generate
        for (g=0; g<9; g=g+1) begin : extract_exp
            assign exps_w[g] = fp8_vec[(g*8)+3 +: 4];
        end
    endgenerate

    wire [3:0] maxL1 [0:3];
    wire [3:0] maxL2 [0:1];

    genvar m;
    generate
        for(m=0; m<4; m=m+1) assign maxL1[m] = (exps_w[2*m] > exps_w[2*m+1]) ? exps_w[2*m] : exps_w[2*m+1];
        for(m=0; m<2; m=m+1) assign maxL2[m] = (maxL1[2*m] > maxL1[2*m+1]) ? maxL1[2*m] : maxL1[2*m+1];
    endgenerate

    wire [3:0] max_exp_c = ((((((maxL2[0]) > (maxL2[1])) ? (maxL2[0]) : (maxL2[1]))) > (exps_w[8])) ? ((((maxL2[0]) > (maxL2[1])) ? (maxL2[0]) : (maxL2[1]))) : (exps_w[8]));

    wire [3:0] mants_w        [0:8];
    wire [3:0] shifted_mant_w [0:8];
    generate
        for (g=0; g<9; g=g+1) begin : align_lane
            assign mants_w[g] = (exps_w[g] != 4'd0) ? {1'b1, fp8_vec[(g*8) +: 3]} : 4'd0;
            assign shifted_mant_w[g] = mants_w[g] >> (max_exp_c - exps_w[g]);
        end
    endgenerate

    wire [35:0] aligned_planes_c;
    generate
        for (g=0; g<9; g=g+1) begin : corner_turn
            assign aligned_planes_c[g]         = shifted_mant_w[g][0]; // Plane 0 (axi[8:0])
            assign aligned_planes_c[9 + g]  = shifted_mant_w[g][1]; // Plane 1 (axi[17:9])
            assign aligned_planes_c[18 + g] = shifted_mant_w[g][2]; // Plane 2 (axi[26:18])
            assign aligned_planes_c[27 + g] = shifted_mant_w[g][3]; // Plane 3 (axi[35:27])
        end
    endgenerate

    always @(posedge clk) begin
        aligned_planes <= aligned_planes_c;
        max_exp        <= max_exp_c;
    end
endmodule
