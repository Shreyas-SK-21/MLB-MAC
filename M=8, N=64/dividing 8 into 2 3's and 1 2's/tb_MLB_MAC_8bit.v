// tb_fair_mlb.v
// ============================================================
// FAIR COMPARISON testbench for MLB_8
// Matched exactly to tb_fair_intmac.v:
//   - Same K values: 64, 128, 256, 512, 1024, 2048
//   - Same input patterns for each test
//   - Same number of tests
//   - Same simulation structure
// This ensures VCD switching activity is comparable
// and power numbers are fair to compare
//
// MLB_8 port map:
//   alpha_x [63:0]  — 8 x 8-bit bases packed, [7:0]=base0..[63:56]=base7
//   alpha_w [63:0]  — same for weights
//   axi     [511:0] — 8 x 64-bit binary planes
//   awi     [511:0] — same for weights
//   K       [12:0]  — dot product length
//   valid_in        — enable accumulation
//   done            — pulses high when result is valid
//   mlb     [37:0]  — signed result
// ============================================================

`timescale 1ns/1ps

module tb_fair_mlb;

    // ---------------------------------------------------------------
    // Parameters — must match tb_fair_intmac exactly
    // ---------------------------------------------------------------
    parameter N     = 64;
    parameter CLK_P = 4;    // 250 MHz

    // ---------------------------------------------------------------
    // DUT ports
    // ---------------------------------------------------------------
    reg                clk, rst, valid_in;
    reg  [63:0]        alpha_x, alpha_w;
    reg  [511:0]       axi, awi;
    reg  [12:0]        K;
    wire signed [38:0] mlb;
    wire               done;

    // ---------------------------------------------------------------
    // DUT
    // ---------------------------------------------------------------
    MLB_8_332 dut (
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

    // ---------------------------------------------------------------
    // Clock
    // ---------------------------------------------------------------
    initial clk = 0;
    always #2 clk = ~clk;

    // ---------------------------------------------------------------
    // Bookkeeping
    // ---------------------------------------------------------------
    integer pass_cnt = 0, fail_cnt = 0, test_id = 0;

    // ---------------------------------------------------------------
    // Fill tasks
    // All 8 planes set to same 64-bit pattern
    // ---------------------------------------------------------------
    task fill_planes_x;
        input [63:0] pattern;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                axi[64*i +: 64] = pattern;
        end
    endtask

    task fill_planes_w;
        input [63:0] pattern;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                awi[64*i +: 64] = pattern;
        end
    endtask

    // Alternating plane patterns (odd planes one pattern, even planes another)
    task fill_alt_planes_x;
        input [63:0] even_pat;
        input [63:0] odd_pat;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                axi[64*i +: 64] = (i % 2 == 0) ? even_pat : odd_pat;
        end
    endtask

    task fill_alt_planes_w;
        input [63:0] even_pat;
        input [63:0] odd_pat;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                awi[64*i +: 64] = (i % 2 == 0) ? even_pat : odd_pat;
        end
    endtask

    // Set all 8 bases to same 8-bit value
    task fill_alpha_x;
        input [7:0] val;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                alpha_x[8*i +: 8] = val;
        end
    endtask

    task fill_alpha_w;
        input [7:0] val;
        integer i;
        begin
            for (i = 0; i < 8; i = i + 1)
                alpha_w[8*i +: 8] = val;
        end
    endtask

    // ---------------------------------------------------------------
    // Reset task
    // ---------------------------------------------------------------
    task do_reset;
        begin
            rst      = 1;
            valid_in = 0;
            @(posedge clk); #1;
            @(posedge clk); #1;
            rst = 0;
        end
    endtask

    // ---------------------------------------------------------------
    // Run task — drive valid_in for K/N cycles then wait for done
    // ---------------------------------------------------------------
    task run_test;
        input integer    Kv;
        input signed [37:0] expected;
        integer cyc, timeout;
        begin
            test_id  = test_id + 1;
            K        = Kv;
            valid_in = 0;

            do_reset;

            // drive valid_in for K/N cycles — same cycle count as int MAC
            valid_in = 1;
            for (cyc = 0; cyc < Kv/N; cyc = cyc + 1) begin
                @(posedge clk); #1;
            end
            valid_in = 0;

            // wait for done
            timeout = 0;
            while (!done && timeout < 50) begin
                @(posedge clk); #1;
                timeout = timeout + 1;
            end

            if (done) begin
                if (mlb === expected) begin
                    $display("[PASS] Test %0d | K=%0d  mlb=%0d (expected %0d)",
                              test_id, Kv, $signed(mlb), $signed(expected));
                    pass_cnt = pass_cnt + 1;
                end else begin
                    $display("[FAIL] Test %0d | K=%0d  mlb=%0d  expected=%0d",
                              test_id, Kv, $signed(mlb), $signed(expected));
                    fail_cnt = fail_cnt + 1;
                end
            end else begin
                $display("[FAIL] Test %0d | K=%0d  done never asserted (timeout)", test_id, Kv);
                fail_cnt = fail_cnt + 1;
            end

            // idle cycles between tests — same as int MAC testbench
            repeat(4) @(posedge clk);
        end
    endtask

    // ---------------------------------------------------------------
    // Stimulus — same groups as tb_fair_intmac
    //
    // MLB reference for all-ones planes, all-same alpha=a:
    //   Each (i,j) unit: xnor_popcount = +64 per cycle (all agree)
    //   contrib per cycle = 2*64 - 64 = 64
    //   acc over K/N cycles = 64 * K/N = K
    //   basis_mult = a*a * K
    //   16 units per MLB_4, 4 MLB_4s = 64 units total
    //   mlb = 64 * a*a * K
    // ---------------------------------------------------------------
    initial begin
        $dumpfile("tb_fair_mlb.vcd");
        $dumpvars(0, tb_fair_mlb);

        rst = 1; valid_in = 0;
        alpha_x = 0; alpha_w = 0;
        axi = 0; awi = 0; K = 0;
        repeat(4) @(posedge clk);

        $display("=============================================================");
        $display(" MLB-8 Fair Comparison Testbench | N=%0d", N);
        $display("=============================================================");

        // -----------------------------------------------------------------
        // GROUP 1 — All-ones planes, alpha=1
        // Each of 64 units: acc=K, basis_mult=1*1*K=K
        // mlb = 64 * 1 * K
        // -----------------------------------------------------------------
        $display("\n-- Group 1: all-ones planes, alpha=1 --");
        fill_planes_x(64'hFFFF_FFFF_FFFF_FFFF);
        fill_planes_w(64'hFFFF_FFFF_FFFF_FFFF);
        fill_alpha_x(8'd1); fill_alpha_w(8'd1);
        run_test(64,  38'sd4096);    // 64*1*1*64
        run_test(128, 38'sd8192);    // 64*1*1*128
        run_test(256, 38'sd16384);   // 64*1*1*256
        run_test(512, 38'sd32768);   // 64*1*1*512

        // -----------------------------------------------------------------
        // GROUP 2 — All-zeros planes (xnor gives all-ones, same as group 1)
        // bx=0, bw=0 → xnor=1 for all bits → same as all-ones
        // -----------------------------------------------------------------
        $display("\n-- Group 2: all-zeros planes (xnor=all-ones) --");
        fill_planes_x(64'h0);
        fill_planes_w(64'h0);
        fill_alpha_x(8'd1); fill_alpha_w(8'd1);
        run_test(64,  38'sd4096);
        run_test(256, 38'sd16384);

        // -----------------------------------------------------------------
        // GROUP 3 — Opposing planes (bx=all-ones, bw=all-zeros)
        // xnor = 0 for all bits → P=0, contrib = 0 - 64 = -64
        // acc over K/N = -64 * K/N = -K
        // basis_mult = 1*1*(-K) = -K
        // mlb = 64 * (-K)
        // -----------------------------------------------------------------
        $display("\n-- Group 3: opposing planes (bx=1s, bw=0s) --");
        fill_planes_x(64'hFFFF_FFFF_FFFF_FFFF);
        fill_planes_w(64'h0);
        fill_alpha_x(8'd1); fill_alpha_w(8'd1);
        run_test(64,  -38'sd4096);   // 64*1*1*(-64)
        run_test(256, -38'sd16384);  // 64*1*1*(-256)

        // -----------------------------------------------------------------
        // GROUP 4 — Checkerboard (zero dot product)
        // bx=0xAAAA..., bw=0x5555... → xnor=0 for all → same as group 3
        // but here half planes cancel:
        // Actually for checker: P=0 for all pairs → contrib=-64 → same as group3
        // Use alternating planes to get zero: half planes +K, half -K
        // bx_even=all-ones, bx_odd=all-zeros
        // bw_even=all-ones, bw_odd=all-zeros
        // even×even: contrib=+64, odd×odd: contrib=+64
        // even×odd:  contrib=-64, odd×even: contrib=-64
        // sum = 16*(+K) + 16*(+K) + 16*(-K) + 16*(-K) = 0
        // -----------------------------------------------------------------
        $display("\n-- Group 4: alternating planes (zero sum) --");
        fill_alt_planes_x(64'hFFFF_FFFF_FFFF_FFFF, 64'h0);
        fill_alt_planes_w(64'hFFFF_FFFF_FFFF_FFFF, 64'h0);
        fill_alpha_x(8'd1); fill_alpha_w(8'd1);
        run_test(64,  38'sd0);
        run_test(256, 38'sd0);

        // -----------------------------------------------------------------
        // GROUP 5 — Non-unity alpha, all-ones planes
        // mlb = 64 * alpha_x * alpha_w * K
        // alpha_x=7, alpha_w=9: 64 * 7 * 9 * K = 4032 * K
        // -----------------------------------------------------------------
        $display("\n-- Group 5: non-unity alpha, all-ones planes --");
        fill_planes_x(64'hFFFF_FFFF_FFFF_FFFF);
        fill_planes_w(64'hFFFF_FFFF_FFFF_FFFF);
        fill_alpha_x(8'd7); fill_alpha_w(8'd9);
        run_test(64,  38'sd258048);   // 64*7*9*64  = 4032*64
        run_test(128, 38'sd516096);   // 64*7*9*128 = 4032*128
        run_test(512, 38'sd2064384);  // 64*7*9*512 = 4032*512

        // -----------------------------------------------------------------
        // GROUP 6 — Non-unity alpha + alternating planes (zero sum)
        // Same alternating pattern as group 4 → sum=0 for any alpha
        // mlb = 0 for any alpha
        // -----------------------------------------------------------------
        $display("\n-- Group 6: non-unity alpha + alternating planes --");
        fill_alt_planes_x(64'hFFFF_FFFF_FFFF_FFFF, 64'h0);
        fill_alt_planes_w(64'hFFFF_FFFF_FFFF_FFFF, 64'h0);
        fill_alpha_x(8'd5); fill_alpha_w(8'd5);
        run_test(64,   38'sd0);
        run_test(256,  38'sd0);
        run_test(1024, 38'sd0);

        // -----------------------------------------------------------------
        // GROUP 7 — Max alpha (255), all-ones planes
        // mlb = 64 * 255 * 255 * K = 64 * 65025 * K
        // -----------------------------------------------------------------
        $display("\n-- Group 7: max alpha (255), all-ones planes --");
        fill_planes_x(64'hFFFF_FFFF_FFFF_FFFF);
        fill_planes_w(64'hFFFF_FFFF_FFFF_FFFF);
        fill_alpha_x(8'd255); fill_alpha_w(8'd255);
        run_test(64,  38'sd266342400);   // 64*65025*64
        run_test(128, 38'sd532684800);   // 64*65025*128

        // ---------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------
        $display("\n=============================================================");
        $display(" Results: %0d PASSED | %0d FAILED | %0d TOTAL",
                  pass_cnt, fail_cnt, test_id);
        $display("=============================================================\n");

        $finish;
    end

    // ---------------------------------------------------------------
    // Watchdog
    // ---------------------------------------------------------------
    initial begin
        #10000000;
        $display("[WATCHDOG] Timeout at %0t", $time);
        $finish;
    end

endmodule
