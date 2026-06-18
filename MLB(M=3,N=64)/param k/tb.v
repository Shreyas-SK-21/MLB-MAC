module tb_MLB_MAC_3_bit;

    // -------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------
    reg  [8:0]   alpha_x, alpha_w;
    reg  [191:0] axi, awi;
    reg  [12:0]  K;
    reg          clk, rst, valid_in;
    wire signed [27:0] mlb;
    wire               done;

    MLB_3 dut (
        .mlb      (mlb),
        .done     (done),
        .alpha_x  (alpha_x),
        .alpha_w  (alpha_w),
        .axi      (axi),
        .awi      (awi),
        .K        (K),
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in)
    );

    // -------------------------------------------------------
    // Clock: 10 ns period
    // -------------------------------------------------------
    initial clk = 0;
    always  #5 clk = ~clk;

    integer errors;

    // -------------------------------------------------------
    // Task: drive valid_in for exactly K/64 cycles, then wait
    // for done and check result
    // -------------------------------------------------------
    task run_and_check;
        input signed [27:0] expected;
        input integer        test_id;
        input integer        cycles;   // K/64
        integer c;
        begin
            // Reset accumulator state
            @(posedge clk); #1;
            rst      = 1;
            valid_in = 0;
            @(posedge clk); #1;
            rst = 0;
            @(posedge clk); #1;

            // Drive valid_in for 'cycles' clock cycles
            for(c = 0; c < cycles; c = c+1) begin
                valid_in = 1;
                @(posedge clk); #1;
            end
            valid_in = 0;

            // Wait for done
            wait(done);
            #1;

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
    // Helper: pack identical alpha value into all 3 slots
    // -------------------------------------------------------
    function [8:0] pack_alpha;
        input [2:0] v;
        begin
            pack_alpha = {v, v, v};
        end
    endfunction

    // -------------------------------------------------------
    // Helper: fill all 3 bit-slices with the same 64-bit pattern
    // -------------------------------------------------------
    function [191:0] pack_bits;
        input [63:0] pat;
        begin
            pack_bits = {pat, pat, pat};
        end
    endfunction

    integer i;

    // -------------------------------------------------------
    // Reference function for a single xnor_popcount result
    // (all K bits same pattern repeated K/64 times)
    // popcount of 64-bit XNOR, then signed = 2P-64 per cycle,
    // accumulated over K/64 cycles
    // -------------------------------------------------------
    function integer ref_xnorpop;
        input [63:0] pa, pb;
        input [12:0] kk;
        integer p, c, cycle_contrib, total;
        reg [63:0] xnn;
        begin
            xnn = ~(pa ^ pb);
            p   = 0;
            for(c=0; c<64; c=c+1) if(xnn[c]) p = p+1;
            cycle_contrib = 2*p - 64;
            total = cycle_contrib * (kk >> 6);   // accumulate K/64 cycles
            ref_xnorpop = total;
        end
    endfunction

    // -------------------------------------------------------
    // Reference: full MLB result
    // mlb = Σ_i Σ_j alpha_x[i]*alpha_w[j]*xnorpop(bx_i, bw_j)
    // -------------------------------------------------------
    function integer ref_mlb;
        input [8:0]   ax, aw;
        input [191:0] bx, bw;
        input [12:0]  kk;
        integer ii, jj, acc;
        integer axv, awv, pop;
        reg [63:0] bxi, bwj;
        begin
            acc = 0;
            for(ii=0; ii<3; ii=ii+1) begin
                for(jj=0; jj<3; jj=jj+1) begin
                    axv = ax[ii*3 +: 3];
                    awv = aw[jj*3 +: 3];
                    bxi = bx[ii*64 +: 64];
                    bwj = bw[jj*64 +: 64];
                    pop = ref_xnorpop(bxi, bwj, kk);
                    acc = acc + axv * awv * pop;
                end
            end
            ref_mlb = acc;
        end
    endfunction

    // -------------------------------------------------------
    // Tests
    // -------------------------------------------------------
    initial begin
        $dumpfile("tb_MLB_MAC_3_bit.vcd");
        $dumpvars(0, tb_MLB_MAC_3_bit);
        errors   = 0;
        rst      = 1;
        valid_in = 0;
        alpha_x  = 0;
        alpha_w  = 0;
        axi      = 0;
        awi      = 0;
        K        = 13'd64;
        @(posedge clk); #1;
        rst = 0;

        // ==============================================================
        // GROUP A — K = N = 64  (single pass)
        // ==============================================================

        // Test 1: all bits match, all alpha=1
        // XNOR all 1s → P=64, contrib=64, K/N=1 cycle
        // each pair: 1*1*64 = 64, 9 pairs → 576
        K       = 13'd64;
        alpha_x = pack_alpha(3'd1);
        alpha_w = pack_alpha(3'd1);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd576, 1, 1);

        // Test 2: all bits mismatch (a=1s, b=0s), all alpha=1
        // P=0, contrib=-64, 9 pairs → -576
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-28'sd576, 2, 1);

        // Test 3: all zeros in both → all match, same as test 1
        axi = pack_bits(64'h0000000000000000);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(28'sd576, 3, 1);

        // Test 4: half match half mismatch → P=32, contrib=0
        // xnor of 0xAAAA...(alternating) with 0xFFFF... = 0x5555...
        // P=32, 2*32-64=0 → all 9 pairs give 0 → mlb=0
        axi = pack_bits(64'hAAAAAAAAAAAAAAAA);
        awi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd0, 4, 1);

        // Test 5: alpha_x=2, alpha_w=3, all match
        // each pair: 2*3*64=384 but only diagonal alpha pairs?
        // No: MLB sums ALL i,j pairs:
        // Σ_i Σ_j ax[i]*aw[j] = (2+2+2)*(3+3+3) = 6*9 = 54
        // Each xnorpop = 64 → mlb = 54*64 = 3456
        alpha_x = pack_alpha(3'd2);
        alpha_w = pack_alpha(3'd3);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd3456, 5, 1);

        // Test 6: alpha_x=7, alpha_w=7, all match
        // Σ ax[i]*aw[j] = (7+7+7)*(7+7+7) = 21*21 = 441
        // mlb = 441*64 = 28224
        alpha_x = pack_alpha(3'd7);
        alpha_w = pack_alpha(3'd7);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd28224, 6, 1);

        // Test 7: alpha_x=7, alpha_w=7, all mismatch
        // mlb = 441 * (-64) = -28224
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-28'sd28224, 7, 1);

        // Test 8: different alpha per bit-slice
        // alpha_x = [1,2,3], alpha_w = [1,2,3], all match
        // Σ_i Σ_j ax[i]*aw[j]
        //   i=0: ax=1 → j=0,1,2: 1*1+1*2+1*3 = 6
        //   i=1: ax=2 → j=0,1,2: 2*1+2*2+2*3 = 12
        //   i=2: ax=3 → j=0,1,2: 3*1+3*2+3*3 = 18
        //   total = 36, each xnorpop=64 → mlb = 36*64 = 2304
        alpha_x = {3'd3, 3'd2, 3'd1};   // [2:0]=bit0=1, [5:3]=bit1=2, [8:6]=bit2=3
        alpha_w = {3'd3, 3'd2, 3'd1};
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd2304, 8, 1);

        // Test 9: alpha_x=0 in all → result must be 0
        alpha_x = pack_alpha(3'd0);
        alpha_w = pack_alpha(3'd7);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd0, 9, 1);

        // Test 10: alpha_w=0 → result must be 0
        alpha_x = pack_alpha(3'd7);
        alpha_w = pack_alpha(3'd0);
        run_and_check(28'sd0, 10, 1);

        // ==============================================================
        // GROUP B — K = 128  (2 passes, same pattern both cycles)
        // ==============================================================

        // Test 11: K=128, all match, alpha=1 → 2 cycles × 64 = 128 per pair
        // mlb = 9 * 1*1 * 128 = 1152
        K       = 13'd128;
        alpha_x = pack_alpha(3'd1);
        alpha_w = pack_alpha(3'd1);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd1152, 11, 2);

        // Test 12: K=128, all mismatch, alpha=1
        // mlb = 9 * 1*1 * (-128) = -1152
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-28'sd1152, 12, 2);

        // Test 13: K=128, half match → contrib=0 each cycle → mlb=0
        axi = pack_bits(64'hAAAAAAAAAAAAAAAA);
        awi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd0, 13, 2);

        // Test 14: K=128, all match, alpha_x=3, alpha_w=3
        // Σ_i Σ_j ax[i]*aw[j] = 9 pairs × (3×3) = 81, xnorpop = 128 → mlb = 81*128 = 10368
        alpha_x = pack_alpha(3'd3);
        alpha_w = pack_alpha(3'd3);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd10368, 14, 2);

        // ==============================================================
        // GROUP C — K = 256  (4 passes)
        // ==============================================================

        // Test 15: K=256, all match, alpha=1
        // mlb = 9 * 256 = 2304
        K       = 13'd256;
        alpha_x = pack_alpha(3'd1);
        alpha_w = pack_alpha(3'd1);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd2304, 15, 4);

        // Test 16: K=256, all mismatch, alpha=1
        // mlb = 9 * (-256) = -2304
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-28'sd2304, 16, 4);

        // Test 17: K=256, alpha=2, all match
        // Σ ax*aw = 36*4 = 144... No:
        // all ax[i]=2, all aw[j]=2 → Σ_ij = 9*(2*2)=36
        // xnorpop=256 → mlb = 36*256 = 9216
        alpha_x = pack_alpha(3'd2);
        alpha_w = pack_alpha(3'd2);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(28'sd9216, 17, 4);

        // ==============================================================
        // GROUP D — Randomised reference check
        // ==============================================================

// ==============================================================
        // GROUP D — Randomised reference check
        // ==============================================================

        // Test 18: K=64, mixed pattern, alpha=[1,2,3],[3,2,1]
        // Use ref_mlb function to compute expected
        K       = 13'd64;
        alpha_x = {3'd1, 3'd2, 3'd3};
        alpha_w = {3'd3, 3'd2, 3'd1};
        axi     = {64'hF0F0F0F0F0F0F0F0, 64'hAAAAAAAAAAAAAAAA, 64'hFFFFFFFFFFFFFFFF};
        awi     = {64'h0F0F0F0F0F0F0F0F, 64'h5555555555555555, 64'hFFFFFFFFFFFFFFFF};
        
        // Pass the function result directly!
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi, K), 18, 1);

        // Test 19: K=128, same pattern × 2 cycles
        K = 13'd128;
        
        // Pass the function result directly!
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi, K), 19, 2);

        // Test 20: K=64, alpha all 1, checkerboard pattern
        // bx=0xAAAA..., bw=0xAAAA... → XNOR=0xFFFF... → P=64, pop=64
        K       = 13'd64;
        alpha_x = pack_alpha(3'd1);
        alpha_w = pack_alpha(3'd1);
        axi     = pack_bits(64'hAAAAAAAAAAAAAAAA);
        awi     = pack_bits(64'hAAAAAAAAAAAAAAAA);
        run_and_check(28'sd576, 20, 1);   // 9 * 1*1 * 64 = 576

        // ==============================================================
        // Summary
        // ==============================================================
        if(errors == 0)
            $display("ALL 20 TESTS PASSED");
        else
            $display("%0d TEST(S) FAILED", errors);

        $finish;
    end

endmodule