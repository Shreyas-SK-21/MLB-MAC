// tb_fair_intmac.v
// ============================================================
// FAIR COMPARISON testbench for int_mac_M8_N64_K128
//   - Updated for K=128 (N=64, 2-cycle accumulation)
//   - Uses valid_in and waits for done signal
// ============================================================

`timescale 1ns/1ps

module tb_fair_intmac;

    // ---------------------------------------------------------------
    // Parameters
    // ---------------------------------------------------------------
    parameter N     = 64;
    parameter CLK_P = 4;    // 250 MHz, same as paper

    // ---------------------------------------------------------------
    // DUT ports
    // ---------------------------------------------------------------
    reg                clk, rst, valid_in;
    reg  [511:0]       a_flat;       // N=64 lanes x 8-bit
    reg  [511:0]       b_flat;
    reg  [7:0]         alpha_x;
    reg  [7:0]         alpha_w;
    reg  signed [15:0] beta_xw;
    wire signed [46:0] result;
    wire               done;

    // ---------------------------------------------------------------
    // DUT
    // ---------------------------------------------------------------
    int_mac_M8_N64_K128 dut (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        .a_flat   (a_flat),
        .b_flat   (b_flat),
        .alpha_x  (alpha_x),
        .alpha_w  (alpha_w),
        .beta_xw  (beta_xw),
        .result   (result),
        .done     (done)
    );

    // ---------------------------------------------------------------
    // Clock
    // ---------------------------------------------------------------
    initial clk = 0;
    always #(CLK_P/2) clk = ~clk;

    // ---------------------------------------------------------------
    // Bookkeeping
    // ---------------------------------------------------------------
    integer pass_cnt = 0, fail_cnt = 0, test_id = 0;

    // ---------------------------------------------------------------
    // Fill tasks
    // ---------------------------------------------------------------
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
            rst      = 1;
            valid_in = 0;
            @(posedge clk); #1;
            @(posedge clk); #1;
            rst      = 0;
            @(posedge clk); #1;
        end
    endtask

    // ---------------------------------------------------------------
    // Run task — drive valid_in for 2 cycles, wait for done
    // ---------------------------------------------------------------
    task run_test;
        input [7:0]         ax, aw;     // alpha values
        input [15:0]        bxw;        // beta offset
        input signed [46:0] expected;
        begin
            test_id = test_id + 1;
            alpha_x = ax;
            alpha_w = aw;
            beta_xw = bxw;

            do_reset;

            // Drive K=128 (2 chunks of 64)
            valid_in = 1;
            @(posedge clk); #1; // Cycle 0
            @(posedge clk); #1; // Cycle 1
            valid_in = 0;

            // Wait for pipeline valid signal
            wait(done);
            #1;

            if (result === expected) begin
                $display("[PASS] Test %0d | K=128  result=%0d (expected %0d)",
                          test_id, $signed(result), $signed(expected));
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] Test %0d | K=128  result=%0d  expected=%0d",
                          test_id, $signed(result), $signed(expected));
                fail_cnt = fail_cnt + 1;
            end

            // idle cycles between tests
            repeat(4) @(posedge clk);
        end
    endtask

    // ---------------------------------------------------------------
    // Stimulus
    // ---------------------------------------------------------------
    initial begin
        $dumpfile("tb_fair_intmac.vcd");
        $dumpvars(0, tb_fair_intmac);

        rst = 1; valid_in = 0;
        a_flat = 0; b_flat = 0;
        alpha_x = 0; alpha_w = 0; beta_xw = 0;
        repeat(4) @(posedge clk);

        $display("=============================================================");
        $display(" INT-MAC Fair Comparison Testbench | N=64, K=128");
        $display("=============================================================");

        // -----------------------------------------------------------------
        // GROUP 1 — All +1 inputs, alpha=1
        // sum = 128 * 1 * 1 = 128, scaled = 1*1*128 + 0 = 128
        // -----------------------------------------------------------------
        $display("\n-- Group 1: all +1 inputs, alpha=1 --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(8'd1, 8'd1, 16'sd0, 47'sd128);

        // -----------------------------------------------------------------
        // GROUP 2 — All +1 inputs (repeat verification)
        // -----------------------------------------------------------------
        $display("\n-- Group 2: all +1 inputs --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(8'd1, 8'd1, 16'sd0, 47'sd128);

        // -----------------------------------------------------------------
        // GROUP 3 — All -1 activations, +1 weights
        // sum = 128 * (-1) * 1 = -128
        // -----------------------------------------------------------------
        $display("\n-- Group 3: all -1 activations, +1 weights --");
        fill_a(8'hFF); fill_b(8'sd1);  // 8'hFF = -1 signed
        run_test(8'd1, 8'd1, 16'sd0, -47'sd128);

        // -----------------------------------------------------------------
        // GROUP 4 — Alternating +1/-1
        // Even lanes: +1*+1=+1, Odd lanes: -1*+1=-1, sum=0
        // -----------------------------------------------------------------
        $display("\n-- Group 4: alternating +1/-1 lanes (zero sum) --");
        fill_alt_a(8'sd1, 8'hFF); fill_alt_b(8'sd1, 8'sd1);
        run_test(8'd1, 8'd1, 16'sd0, 47'sd0);

        // -----------------------------------------------------------------
        // GROUP 5 — Non-unity alpha
        // sum = 128 * 1 * 1 = 128, scaled = alpha_x*alpha_w*128
        // 7 * 9 * 128 = 8064
        // -----------------------------------------------------------------
        $display("\n-- Group 5: non-unity alpha --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(8'd7, 8'd9, 16'sd0, 47'sd8064);

        // -----------------------------------------------------------------
        // GROUP 6 — Non-unity alpha + mixed inputs (offset test)
        // even: 2*2=4, odd: -2*2=-4, sum=0 per cycle, scaled=0+beta
        // -----------------------------------------------------------------
        $display("\n-- Group 6: non-unity alpha + mixed inputs --");
        fill_alt_a(8'sd2, 8'hFE); fill_alt_b(8'sd2, 8'sd2);
        run_test(8'd5, 8'd5, 16'sd999, 47'sd999);

        // -----------------------------------------------------------------
        // GROUP 7 — Max alpha
        // sum = 128 * 1 * 1 = 128, scaled = 255*255*128 = 8,323,200
        // -----------------------------------------------------------------
        $display("\n-- Group 7: max alpha (255*255) --");
        fill_a(8'sd1); fill_b(8'sd1);
        run_test(8'd255, 8'd255, 16'sd0, 47'sd8323200);

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
        #100000; // Scaled down for fixed K=128 suite
        $display("[WATCHDOG] Timeout at %0t", $time);
        $finish;
    end

endmodule