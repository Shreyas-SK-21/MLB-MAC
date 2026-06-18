`timescale 1ns / 1ps

module tb_MLB_MAC_8_bit;

    // -------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------
    reg  [63:0]  alpha_x, alpha_w;
    reg  [511:0] axi, awi;
    reg          clk, rst, valid_in;
    wire signed [29:0] mlb;
    wire               done;

    MLB_8 dut (
        .mlb      (mlb),
        .done     (done),
        .alpha_x  (alpha_x),
        .alpha_w  (alpha_w),
        .axi      (axi),
        .awi      (awi),
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
    // Task: drive valid_in, wait for done, and check result
    // -------------------------------------------------------
    task run_and_check;
        input signed [29:0] expected;
        input integer        test_id;
        integer c;
        begin
            // Reset accumulator and pipeline state
            @(posedge clk); #1;
            rst      = 1;
            valid_in = 0;
            @(posedge clk); #1;
            rst = 0;
            @(posedge clk); #1;

            // FIX: Pulse valid_in for exactly 1 cycle, then wait 3 cycles 
            // for the pipeline to clear and the multiplexer state to increment.
            // We repeat this 4 times to feed all 4 states into the hardware safely.
            for (c = 0; c < 4; c = c + 1) begin
                valid_in = 1;
                @(posedge clk); #1;
                valid_in = 0;
                
                // Wait for the 3-cycle pipeline latency:
                // 1. xnor_popcount finishes
                // 2. basis_mult stage 1 finishes
                // 3. basis_mult stage 2 finishes (sub_unit_done asserted, state increments)
                @(posedge clk); #1;
                @(posedge clk); #1;
                @(posedge clk); #1;
            end

            // Wait for the final done signal from MLB_8
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
    // Helper: pack identical 8-bit alpha value into all 8 slots
    // -------------------------------------------------------
    function [63:0] pack_alpha;
        input [7:0] v;
        begin
            pack_alpha = {v, v, v, v, v, v, v, v};
        end
    endfunction

    // -------------------------------------------------------
    // Helper: fill all 8 bit-slices with the same 64-bit pattern
    // -------------------------------------------------------
    function [511:0] pack_bits;
        input [63:0] pat;
        begin
            pack_bits = {pat, pat, pat, pat, pat, pat, pat, pat};
        end
    endfunction

    integer i;

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
            for(c=0; c<64; c=c+1) if(xnn[c]) p = p+1;
            ref_xnorpop = 2*p - 64;
        end
    endfunction

    // -------------------------------------------------------
    // Reference: full MLB result (8x8)
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
                    // Treat alphas as positive/unsigned 8-bit values per the implementation
                    axv = ax[ii*8 +: 8];
                    awv = aw[jj*8 +: 8];
                    bxi = bx[ii*64 +: 64];
                    bwj = bw[jj*64 +: 64];
                    pop = ref_xnorpop(bxi, bwj);
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
        $dumpfile("tb_MLB_MAC_8_bit.vcd");
        $dumpvars(0, tb_MLB_MAC_8_bit);
        errors   = 0;
        rst      = 1;
        valid_in = 0;
        alpha_x  = 0;
        alpha_w  = 0;
        axi      = 0;
        awi      = 0;
        @(posedge clk); #1;
        rst = 0;

        // ==============================================================
        // GROUP A — Uniform Patterns
        // ==============================================================

        // Test 1: all bits match, all alpha=1
        // XNOR all 1s → P=64, contrib=64
        // 64 pairs × (1*1*64) = 4096
        alpha_x = pack_alpha(8'd1);
        alpha_w = pack_alpha(8'd1);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(30'sd4096, 1);

        // Test 2: all bits mismatch (a=1s, b=0s), all alpha=1
        // P=0, contrib=-64, 64 pairs → -4096
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-30'sd4096, 2);

        // Test 3: all zeros in both → all match, same as test 1
        axi = pack_bits(64'h0000000000000000);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(30'sd4096, 3);

        // Test 4: half match half mismatch → P=32, contrib=0
        // mlb should be 0 since all contributions are exactly 0
        axi = pack_bits(64'hAAAAAAAAAAAAAAAA);
        awi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(30'sd0, 4);

        // Test 5: alpha_x=2, alpha_w=3, all match
        // Σ ax[i]*aw[j] = (8*2)*(8*3) = 16 * 24 = 384
        // mlb = 384 * 64 = 24576
        alpha_x = pack_alpha(8'd2);
        alpha_w = pack_alpha(8'd3);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(30'sd24576, 5);

        // Test 6: alpha_x=7, alpha_w=7, all match
        // Σ ax[i]*aw[j] = (8*7)*(8*7) = 56 * 56 = 3136
        // mlb = 3136 * 64 = 200704
        alpha_x = pack_alpha(8'd7);
        alpha_w = pack_alpha(8'd7);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(30'sd200704, 6);

        // Test 7: alpha_x=7, alpha_w=7, all mismatch
        // mlb = 3136 * (-64) = -200704
        axi = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi = pack_bits(64'h0000000000000000);
        run_and_check(-30'sd200704, 7);

        // Test 8: different alpha per bit-slice, all match
        // alpha_x = [1,2,3,4,5,6,7,8] -> Sum = 36
        // alpha_w = [1,2,3,4,5,6,7,8] -> Sum = 36
        // Σ_ij = 36 * 36 = 1296
        // mlb = 1296 * 64 = 82944
        alpha_x = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        alpha_w = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(30'sd82944, 8);

        // Test 9: alpha_x=0 in all → result must be 0
        alpha_x = pack_alpha(8'd0);
        alpha_w = pack_alpha(8'd7);
        axi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        awi     = pack_bits(64'hFFFFFFFFFFFFFFFF);
        run_and_check(30'sd0, 9);

        // Test 10: alpha_w=0 → result must be 0
        alpha_x = pack_alpha(8'd7);
        alpha_w = pack_alpha(8'd0);
        run_and_check(30'sd0, 10);

        // ==============================================================
        // GROUP B — Randomised Reference Checks
        // ==============================================================

        // Test 11: Checkerboard vs Solid
        alpha_x = pack_alpha(8'd1);
        alpha_w = pack_alpha(8'd1);
        axi     = pack_bits(64'hAAAAAAAAAAAAAAAA);
        awi     = pack_bits(64'hAAAAAAAAAAAAAAAA);
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi), 11);

        // Test 12: Complex mixed alpha and mixed data patterns
        alpha_x = {8'd8, 8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1};
        alpha_w = {8'd1, 8'd2, 8'd3, 8'd4, 8'd5, 8'd6, 8'd7, 8'd8};
        axi = {
            64'hFFFFFFFFFFFFFFFF, 64'h0000000000000000, 
            64'hAAAAAAAAAAAAAAAA, 64'h5555555555555555, 
            64'hF0F0F0F0F0F0F0F0, 64'h0F0F0F0F0F0F0F0F, 
            64'hCCCCCCCCCCCCCCCC, 64'h3333333333333333
        };
        awi = {
            64'hFFFFFFFFFFFFFFFF, 64'hFFFFFFFFFFFFFFFF, 
            64'hAAAAAAAAAAAAAAAA, 64'hAAAAAAAAAAAAAAAA, 
            64'hF0F0F0F0F0F0F0F0, 64'hF0F0F0F0F0F0F0F0, 
            64'hCCCCCCCCCCCCCCCC, 64'hCCCCCCCCCCCCCCCC
        };
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi), 12);

        // Test 13: Edge case values
        alpha_x = {8'd255, 8'd0, 8'd255, 8'd0, 8'd255, 8'd0, 8'd255, 8'd0};
        alpha_w = {8'd0, 8'd255, 8'd0, 8'd255, 8'd0, 8'd255, 8'd0, 8'd255};
        axi     = pack_bits(64'h123456789ABCDEF0);
        awi     = pack_bits(64'h0FEDCBA987654321);
        run_and_check(ref_mlb(alpha_x, alpha_w, axi, awi), 13);

        // ==============================================================
        // Summary
        // ==============================================================
        if(errors == 0)
            $display("ALL 13 TESTS PASSED");
        else
            $display("%0d TEST(S) FAILED", errors);

        $finish;
    end

endmodule