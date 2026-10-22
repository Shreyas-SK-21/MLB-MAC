module bfp_aligner (
    input  clk,
    input  [199:0] fp8_vec,
    output reg [99:0] aligned_planes, // Reorganized for MLB_4 axi/awi ports
    output reg [3:0]   max_exp
);
    // ------------------------------------------------------------------
    // PIPELINE + TREE FIX:
    //  1) The exponent max-search below is a LOG-DEPTH binary max-tree
    //     (not a sequential running-max loop), so comparator depth is
    //     O(log2 N) instead of O(N). This was the actual bottleneck at
    //     large N even after output-registering the aligner.
    //  2) Per-lane mantissa/shift/corner-turn logic is fully parallel
    //     'generate' assigns -- each lane depends only on the final
    //     max_exp_c value, never on another lane -- so it adds only
    //     one small (4-bit) shifter's worth of depth, not N of them.
    //  3) The whole result is registered on clk, giving a clean
    //     pipeline boundary before the MAC array.
    // ------------------------------------------------------------------
    genvar g;

    // ---- Phase 1a: parallel exponent extraction ----
    wire [3:0] exps_w [0:24];
    generate
        for (g=0; g<25; g=g+1) begin : extract_exp
            assign exps_w[g] = fp8_vec[(g*8)+3 +: 4];
        end
    endgenerate

    // ---- Phase 1b: log-depth max-reduction tree ----
    wire [3:0] maxL1 [0:11];
    wire [3:0] maxL2 [0:5];
    wire [3:0] maxL3 [0:2];
    wire [3:0] maxL4 [0:0];

    genvar m;
    generate
        for(m=0; m<12; m=m+1) assign maxL1[m] = (exps_w[2*m] > exps_w[2*m+1]) ? exps_w[2*m] : exps_w[2*m+1];
        for(m=0; m<6; m=m+1) assign maxL2[m] = (maxL1[2*m] > maxL1[2*m+1]) ? maxL1[2*m] : maxL1[2*m+1];
        for(m=0; m<3; m=m+1) assign maxL3[m] = (maxL2[2*m] > maxL2[2*m+1]) ? maxL2[2*m] : maxL2[2*m+1];
        for(m=0; m<1; m=m+1) assign maxL4[m] = (maxL3[2*m] > maxL3[2*m+1]) ? maxL3[2*m] : maxL3[2*m+1];
    endgenerate

    wire [3:0] max_exp_c = ((((((maxL4[0]) > (exps_w[24])) ? (maxL4[0]) : (exps_w[24]))) > (maxL3[2])) ? ((((maxL4[0]) > (exps_w[24])) ? (maxL4[0]) : (exps_w[24]))) : (maxL3[2]));

    // ---- Phase 2: parallel per-lane mantissa extraction, shift, corner-turn ----
    wire [3:0] mants_w        [0:24];
    wire [3:0] shifted_mant_w [0:24];
    generate
        for (g=0; g<25; g=g+1) begin : align_lane
            // Extract mantissa and add hidden '1' (flush subnormals to zero)
            assign mants_w[g] = (exps_w[g] != 4'd0) ? {1'b1, fp8_vec[(g*8) +: 3]} : 4'd0;
            // Right shift to align to the block max exponent
            assign shifted_mant_w[g] = mants_w[g] >> (max_exp_c - exps_w[g]);
        end
    endgenerate

    wire [99:0] aligned_planes_c;
    generate
        for (g=0; g<25; g=g+1) begin : corner_turn
            // CORNER TURN: map the 4-bit integer into the 4 planes for MLB_4
            assign aligned_planes_c[g]         = shifted_mant_w[g][0]; // Plane 0 (axi[24:0])
            assign aligned_planes_c[25 + g]  = shifted_mant_w[g][1]; // Plane 1 (axi[49:25])
            assign aligned_planes_c[50 + g] = shifted_mant_w[g][2]; // Plane 2 (axi[74:50])
            assign aligned_planes_c[75 + g] = shifted_mant_w[g][3]; // Plane 3 (axi[99:75])
        end
    endgenerate

    // Register the aligner's result -- one clean pipeline boundary.
    always @(posedge clk) begin
        aligned_planes <= aligned_planes_c;
        max_exp        <= max_exp_c;
    end
endmodule
