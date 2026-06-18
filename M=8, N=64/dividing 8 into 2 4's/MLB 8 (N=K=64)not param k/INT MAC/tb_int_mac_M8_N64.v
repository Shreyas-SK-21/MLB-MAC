// tb_fair_intmac.v
// ============================================================
// FAIR COMPARISON testbench for int_mac_M8_N64
// Matched exactly to tb_fair_mlb.v:
//   - Same K values: 64, 128, 256, 512, 1024, 2048
//   - Same input patterns for each test
//   - Same number of tests
//   - Same simulation structure
// This ensures VCD switching activity is comparable
// and power numbers are fair to compare
// ============================================================

`timescale 1ns/1ps

module tb_fair_intmac;

    // ---------------------------------------------------------------
    // Parameters — must match tb_fair_mlb exactly
    // ---------------------------------------------------------------
    parameter N     = 64;
    parameter CLK_P = 4;    // 250 MHz, same as paper

    // ---------------------------------------------------------------
    // DUT ports
    // ---------------------------------------------------------------
    reg                clk, rst, load;
    reg  [511:0]       a_flat;       // N=64 lanes x 8-bit
    reg  [511:0]       b_flat;
    reg  [7:0]         alpha_x;
    reg  [7:0]         alpha_w;
    reg  signed [15:0] beta_xw;
    wire signed [46:0] result;

    // ---------------------------------------------------------------
    // DUT
    // ---------------------------------------------------------------
    int_mac_M8_N64 dut (
        .clk     (clk),
        .rst     (rst),
        .load    (load),
        .a_flat  (a_flat),
        .b_flat  (b_flat),
        .alpha_x (alpha_x),
        .alpha_w (alpha_w),
        .beta_xw (beta_xw),
        .result  (result)
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
    integer k;

    // ---------------------------------------------------------------
    // Fill tasks — same patterns as MLB testbench
    // ---------------------------------------------------------------
    // Fill all 64 lanes with same 8-bit value
    task fill_a;
        input [7:0] val;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1)
                a_flat[8*i +: 8] = val;
        end
    endtask

    task fill_b;
        input [7:0] val;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1)
                b_flat[8*i +: 8] = val;
        end
    endtask

    // Fill alternating pattern across lanes
    task fill_alt_a;
        input [7:0] val_even;
        input [7:0] val_odd;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1)
                a_flat[8*i +: 8] = (i % 2 == 0) ? val_even : val_odd;
        end
    endtask

    task fill_alt_b;
        input [7:0] val_even;
        input [7:0] val_odd;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1)
                b_flat[8*i +: 8] = (i % 2 == 0) ? val_even : val_odd;
        end
    endtask

    // ---------------------------------------------------------------
    // Reset task
    // ---------------------------------------------------------------
    task do_reset;
        begin
            rst  = 1;
            load = 0;
            @(posedge clk); #1;
            @(posedge clk); #1;
            rst  = 0;
        end
    endtask

    // ---------------------------------------------------------------
    // Run task — accumulate for K/N cycles (matched to MLB K cycles)
    // ---------------------------------------------------------------
    task run_test;
        input integer    Kv;         // dot product length
        input [7:0]      ax, aw;     // alpha values
        input [15:0]     bxw;        // beta offset
        input signed [46:0] expected;
        integer cyc;
        begin
            test_id = test_id + 1;
            alpha_x = ax;
            alpha_w = aw;
            beta_xw = bxw;

            do_reset;

            // accumulate for exactly K/N cycles — same as MLB
            load = 1;
            for (cyc = 0; cyc < Kv/N; cyc = cyc + 1) begin
                @(posedge clk); #1;
            end
            load = 0;
            @(posedge clk); #1;  // settle

            if (result === expected) begin
                $display("[PASS] Test %0d | K=%0d  result=%0d (expected %0d)",
                          test_id, Kv, $signed(result), $signed(expected));
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] Test %0d | K=%0d  result=%0d  expected=%0d",
                          test_id, Kv, $signed(result), $signed(expected));
                fail_cnt = fail_cnt + 1;
            end

            // idle cycles between tests — matched to MLB done-wait cycles
            repeat(4) @(posedge clk);
        end
    endtask

    // ---------------------------------------------------------------
    // Expected value helper
    // alpha_prod * (N * K/N * xd * wd) + beta
    // For uniform input: sum = (K/N cycles) * N lanes * xd * wd
    //                        = K * xd * wd
    // scaled = alpha_x * alpha_w * K * xd * wd + beta_xw
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // Stimulus — same groups as MLB testbench
    // ---------------------------------------------------------------
    initial begin
        $dumpfile("tb_fair_intmac.vcd");
        $dumpvars(0, tb_fair_intmac);

        rst = 1; load = 0;
        a_flat = 0; b_flat = 0;
        alpha_x = 0; alpha_w = 0; beta_xw = 0;
        repeat(4) @(posedge clk);

        $display("=============================================================");
        $display(" INT-MAC Fair Comparison Testbench | N=%0d", N);
        $display("=============================================================");

        // -----------------------------------------------------------------
        // GROUP 1 — All +1 inputs, alpha=1 (matches MLB Group 1 all-ones)
        // sum = K * 1 * 1 = K, scaled = 1*1*K + 0 = K
        // -----------------------------------------------------------------
        $display("\n-- Group 1: all +1 inputs, alpha=1 --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(64,   8'd1, 8'd1, 16'sd0, 47'sd64);
        run_test(128,  8'd1, 8'd1, 16'sd0, 47'sd128);
        run_test(256,  8'd1, 8'd1, 16'sd0, 47'sd256);
        run_test(512,  8'd1, 8'd1, 16'sd0, 47'sd512);

        // -----------------------------------------------------------------
        // GROUP 2 — All +1 inputs (matches MLB Group 2 all-zeros xnor=all-ones)
        // -----------------------------------------------------------------
        $display("\n-- Group 2: all +1 inputs --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(64,  8'd1, 8'd1, 16'sd0, 47'sd64);
        run_test(256, 8'd1, 8'd1, 16'sd0, 47'sd256);

        // -----------------------------------------------------------------
        // GROUP 3 — All -1 activations, +1 weights (matches MLB Group 3)
        // sum = K * (-1) * 1 = -K
        // -----------------------------------------------------------------
        $display("\n-- Group 3: all -1 activations, +1 weights --");
        fill_a(8'hFF); fill_b(8'sd1);  // 8'hFF = -1 signed
        run_test(64,  8'd1, 8'd1, 16'sd0, -47'sd64);
        run_test(256, 8'd1, 8'd1, 16'sd0, -47'sd256);

        // -----------------------------------------------------------------
        // GROUP 4 — Alternating +1/-1 (matches MLB checkerboard zero sum)
        // Even lanes: +1*+1=+1, Odd lanes: -1*+1=-1, sum=0
        // -----------------------------------------------------------------
        $display("\n-- Group 4: alternating +1/-1 lanes (zero sum) --");
        fill_alt_a(8'sd1, 8'hFF); fill_alt_b(8'sd1, 8'sd1);
        run_test(64,  8'd1, 8'd1, 16'sd0, 47'sd0);
        run_test(256, 8'd1, 8'd1, 16'sd0, 47'sd0);

        // -----------------------------------------------------------------
        // GROUP 5 — Non-unity alpha (matches MLB Group 5 non-uniform bases)
        // sum = K * 1 * 1 = K, scaled = alpha_x*alpha_w*K
        // -----------------------------------------------------------------
        $display("\n-- Group 5: non-unity alpha --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(64,  8'd7, 8'd9, 16'sd0, 47'sd4032);   // 7*9*64=4032
        run_test(128, 8'd7, 8'd9, 16'sd0, 47'sd8064);   // 7*9*128=8064
        run_test(512, 8'd7, 8'd9, 16'sd0, 47'sd32256);  // 7*9*512=32256

        // -----------------------------------------------------------------
        // GROUP 6 — Non-unity alpha + mixed inputs
        // -----------------------------------------------------------------
        $display("\n-- Group 6: non-unity alpha + mixed inputs --");
        fill_alt_a(8'sd2, 8'hFE); fill_alt_b(8'sd2, 8'sd2);
        // even: 2*2=4, odd: -2*2=-4, sum=0 per cycle, scaled=0+beta
        run_test(64,  8'd5, 8'd5, 16'sd999,  47'sd999);
        run_test(256, 8'd5, 8'd5, 16'sd999,  47'sd999);
        run_test(1024,8'd5, 8'd5, 16'sd999,  47'sd999);

        // -----------------------------------------------------------------
        // GROUP 7 — Max alpha (matches MLB Group 7 unit bases sanity)
        // sum = K * 1 * 1 = K, scaled = 255*255*K
        // -----------------------------------------------------------------
        $display("\n-- Group 7: max alpha (255*255) --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(64,  8'd255, 8'd255, 16'sd0, 47'sd4161600);   // 65025*64
        run_test(128, 8'd255, 8'd255, 16'sd0, 47'sd8323200);  // 65025*128

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
