`timescale 1ns / 1ps

module tb_MLB_8_advanced;

    // -------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------
    reg  [63:0]  alpha_x, alpha_w;
    reg  [511:0] axi, awi;
    wire signed [30:0] mlb;

    // Instantiate the Unit Under Test (DUT)
    MLB_8 dut (
        .mlb     (mlb),
        .alpha_x (alpha_x),
        .alpha_w (alpha_w),
        .axi     (axi),
        .awi     (awi)
    );

    integer errors;

    // -------------------------------------------------------
    // Task: Apply combinational delay and check result
    // -------------------------------------------------------
    task run_and_check;
        input signed [30:0] expected;
        input integer       test_id;
        begin
            // Wait for combinational logic to settle
            #10;

            if($signed(mlb) === expected) begin
                $display("PASS test %02d : mlb = %0d", test_id, $signed(mlb));
            end else begin
                $display("FAIL test %02d : expected %0d, got %0d",
                          test_id, expected, $signed(mlb));
                errors = errors + 1;
            end
        end
    endtask

    // -------------------------------------------------------
    // Helper: Pack identical 8-bit alpha value into all 8 slots (64 bits)
    // -------------------------------------------------------
    function [63:0] pack_alpha;
        input [7:0] v;
        begin
            pack_alpha = {8{v}};
        end
    endfunction

    // -------------------------------------------------------
    // Helper: Pack 64-bit pattern into all 8 bit-slices (512 bits)
    // -------------------------------------------------------
    function [511:0] pack_bits;
        input [63:0] pat;
        begin
            pack_bits = {8{pat}};
        end
    endfunction

    // -------------------------------------------------------
    // Reference function for a single xnor_popcount result
    // popcount of 64-bit XNOR, then signed = 2P-64
    // -------------------------------------------------------
    function integer ref_xnorpop;
        input [63:0] pa, pb;
        integer p, c;
        reg [63:0] xnn;
        begin
            xnn = ~(pa ^ pb);
            p   = 0;
            for(c=0; c<64; c=c+1) begin
                if(xnn[c]) p = p + 1;
            end
            ref_xnorpop = (2 * p) - 64;
        end
    endfunction

    // -------------------------------------------------------
    // Reference: full MLB_8 result
    // mlb = Σ_i Σ_j alpha_x[i]*alpha_w[j]*xnorpop(bx_i, bw_j)
    // -------------------------------------------------------
    function integer ref_mlb;
        input [63:0]  ax, aw;
        input [511:0] bx, bw;
        integer ii, jj, acc;
        integer axv, awv, pop;
        reg [63:0] bxi, bwj;
        begin
            acc = 0;
            for(ii=0; ii<8; ii=ii+1) begin
                for(jj=0; jj<8; jj=jj+1) begin
                    // Extract the 8-bit alphas (treated as unsigned positive integers)
                    axv = ax[ii*8 +: 8];
                    awv = aw[jj*8 +: 8];
                    
                    // Extract the 64-bit bitmasks
                    bxi = bx[ii*64 +: 64];
                    bwj = bw[jj*64 +: 64];
                    
                    pop = ref_xnorpop(bxi, bwj);
                    
                    // Accumulate: xnorpop * (alpha_x * alpha_w)
                    acc = acc + (axv * awv * pop);
                end
            end
            ref_mlb = acc;
        end
    endfunction

    // -------------------------------------------------------
    // Tests
    // -------------------------------------------------------
    initial begin
        $dumpfile("tb_MLB_8_advanced.vcd");
        $dumpvars(0, tb_MLB_8_advanced);
        
        errors   = 0;
        alpha_x  = 0;
        alpha_w  = 0;
        axi      = 0;
        awi      = 0;
        
        #10;

        $display("==========================================================");
        $display("   STARTING MLB_8 TESTS");
        $display("==========================================================");

        // ==============================================================
        // GROUP A — Hardcoded Expected Values
        // ==============================================================

        // Test 1: All bits match, all alphas = 1
        // XNOR all 1s → P=64, contrib=64
        // 8x8 matrix = 64 pairs. Each pair: 1*1*64 = 64. Total = 64 * 64 = 4096.
        alpha_x = pack_alpha(8'd1);
        alpha_w = pack_alpha(8'd1);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(31'sd4096, 1);

        // Test 2: All bits mismatch (a=1s, b=0s), all alphas = 1
        // P=0, contrib=-64. 64 pairs -> 64 * -64 = -4096.
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-31'sd4096, 2);

        // Test 3: All zeros in both → all match, same as test 1
        axi = pack_bits(64'h0000000000000000);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(31'sd4096, 3);

        // Test 4: Half match / half mismatch → P=32, contrib=0
        // Result must be 0 regardless of alpha
        axi = pack_bits(64'hAAAAAAAAAAAAAAAA);
        awi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(31'sd0, 4);

        // Test 5: alpha_x=2, alpha_w=3, all match
        // Σ_i Σ_j ax[i]*aw[j] = (8*2)*(8*3) = 16 * 24 = 384. 
        // mlb = 384 * 64 = 24576
        alpha_x = pack_alpha(8'd2);
        alpha_w = pack_alpha(8'd3);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(31'sd24576, 5);

        // Test 6: alpha_x=10, alpha_w=10, all mismatch
        // Sum = (8*10)*(8*10) = 80*80 = 6400. 
        // mlb = 6400 * (-64) = -409600
        alpha_x = pack_alpha(8'd10);
        alpha_w = pack_alpha(8'd10);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'h0000000000000000);
        run_and_check(-31'sd409600, 6);

        // Test 7: Zeroed alpha array -> Result must be 0
        alpha_x = pack_alpha(8'd0);
        alpha_w = pack_alpha(8'd7);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(31'sd0, 7);

        // ==============================================================
        // GROUP B — Dynamic Expected Values via Reference Function
        // ==============================================================

        // Test 8: Sequential Alphas (1 through 8), all match
        alpha_x = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        alpha_w = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi), 8);

        // Test 9: Complex Alphas, Random alternating bit patterns
        alpha_x = {8'd1, 8'd25, 8'd3, 8'd17, 8'd5, 8'd12, 8'd7, 8'd2};
        alpha_w = {8'd4, 8'd8, 8'd15, 8'd16, 8'd23, 8'd42, 8'd10, 8'd9};
        axi     = pack_bits(64'hF0F0F0F0F0F0F0F0);
        awi     = pack_bits(64'h0F0F0F0F0F0F0F0F); // Completely orthogonal
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi), 9);

        // Test 10: Highly Mixed AXI/AWI masks and Alphas
        alpha_x = {8'd11, 8'd2, 8'd33, 8'd4, 8'd55, 8'd6, 8'd77, 8'd8};
        alpha_w = {8'd88, 8'd7, 8'd66, 8'd5, 8'd44, 8'd3, 8'd22, 8'd1};
        axi     = {
                    64'hAAAAAAAAAAAAAAAA, 64'hFFFFFFFFFFFFFFFF, 
                    64'h0000000000000000, 64'h123456789ABCDEF0,
                    64'h0FEDCBA987654321, 64'h5555555555555555,
                    64'hFFFFFFFFFFFFFFFF, 64'hAAAAAAAAAAAAAAAA
                  };
        awi     = {
                    64'h5555555555555555, 64'hFFFFFFFFFFFFFFFF, 
                    64'h1111111111111111, 64'h123456789ABCDEF0,
                    64'h0FEDCBA987654321, 64'hAAAAAAAAAAAAAAAA,
                    64'h0000000000000000, 64'hFFFFFFFFFFFFFFFF
                  };
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi), 10);

        // ==============================================================
        // Summary
        // ==============================================================
        $display("==========================================================");
        if(errors == 0)
            $display("   SUCCESS: ALL 10 TESTS PASSED");
        else
            $display("   FAILED: %0d TEST(S) FAILED", errors);
        $display("==========================================================");

        $finish;
    end

endmodule