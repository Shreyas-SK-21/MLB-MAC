module bfp_aligner (
    input  [71:0] fp8_vec,
    output reg [35:0] aligned_planes, // Reorganized for MLB_4 axi/awi ports
    output reg [3:0]   max_exp
);
    reg [3:0] exps [0:8];
    reg [3:0] mants [0:8];
    reg [3:0] shift_amt;
    reg [3:0] shifted_mant;
    integer i;

    always @(*) begin
        max_exp = 4'd0;
        
        // Phase 1: Extract Exponents and Find Max
        for (i=0; i<9; i=i+1) begin
            exps[i] = fp8_vec[(i*8)+3 +: 4];
            if (exps[i] > max_exp) begin
                max_exp = exps[i];
            end
        end

        // Phase 2: Align Mantissas and "Corner Turn" into Bit Planes
        for (i=0; i<9; i=i+1) begin
            // Extract mantissa and add hidden '1'
            if (exps[i] != 4'd0) begin
                mants[i] = {1'b1, fp8_vec[(i*8) +: 3]};
            end else begin
                mants[i] = 4'd0; // Flush subnormals to zero
            end

            // Right shift to align to the block max exponent
            shift_amt = max_exp - exps[i];
            shifted_mant = mants[i] >> shift_amt;

            // CORNER TURN: Map the 4-bit integer into the 4 planes for MLB_4
            aligned_planes[i]         = shifted_mant[0]; // Plane 0 (axi[8:0])
            aligned_planes[9 + i]  = shifted_mant[1]; // Plane 1 (axi[17:9])
            aligned_planes[18 + i] = shifted_mant[2]; // Plane 2 (axi[26:18])
            aligned_planes[27 + i] = shifted_mant[3]; // Plane 3 (axi[35:27])
        end
    end
endmodule
