// tb_int_mac_M3_N64.v
// ============================================================
// Testbench for int_mac_M3_N64
//
// Test plan:
//   TC1  — All zeros (expect 0)
//   TC2  — All ones activations, all ones weights (max positive)
//   TC3  — All -1 activations, all +1 weights  (max negative sum)
//   TC4  — Mixed known values, hand-calculated expected result
//   TC5  — Reset mid-accumulation, result must be 0 after reset
//   TC6  — alpha_x=0 forces result to beta_xw only
//   TC7  — alpha_w=0 forces result to beta_xw only
//   TC8  — Single non-zero lane, rest zero
// ============================================================

`timescale 1ns/1ps

module tb_int_mac_M3_N64;

// ── DUT ports ──────────────────────────────────────────────
reg         clk;
reg         rst;
reg         load;
reg  [191:0] a_flat;
reg  [191:0] b_flat;
reg  [2:0]  alpha_x;
reg  [2:0]  alpha_w;
reg  [5:0]  beta_xw;
wire signed [20:0] result;

// ── DUT instantiation ──────────────────────────────────────
int_mac_M3_N64 dut (
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

// ── Clock: 10 ns period ────────────────────────────────────
initial clk = 0;
always #5 clk = ~clk;

// ── Helpers ────────────────────────────────────────────────
integer pass_count, fail_count;

// Sign-extend 3-bit to integer for display
function integer se3;
    input [2:0] v;
    begin
        se3 = v[2] ? ($signed(v)) : v;
    end
endfunction

// Fill all 64 lanes of a_flat with the same 3-bit value
task fill_a;
    input [2:0] val;
    integer k;
    begin
        for (k = 0; k < 64; k = k + 1)
            a_flat[3*k +: 3] = val;
    end
endtask

// Fill all 64 lanes of b_flat with the same 3-bit value
task fill_b;
    input [2:0] val;
    integer k;
    begin
        for (k = 0; k < 64; k = k + 1)
            b_flat[3*k +: 3] = val;
    end
endtask

// Reset DUT, wait 2 cycles
task do_reset;
    begin
        rst  = 1;
        load = 0;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst  = 0;
    end
endtask

// Drive load for exactly 3 cycles (M=3 accumulation cycles)
// a_flat / b_flat must be set before calling
task do_accumulate;
    begin
        load = 1;
        @(posedge clk); #1;   // cycle 1
        @(posedge clk); #1;   // cycle 2
        @(posedge clk); #1;   // cycle 3
        load = 0;
        @(posedge clk); #1;   // let combinational outputs settle
    end
endtask

// Check result against expected; print PASS/FAIL
task check;
    input [63*8-1:0] tc_name;   // string
    input signed [20:0] expected;
    begin
        if (result === expected) begin
            $display("PASS  %-30s | expected=%0d  got=%0d",
                     tc_name, $signed(expected), $signed(result));
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL  %-30s | expected=%0d  got=%0d",
                     tc_name, $signed(expected), $signed(result));
            fail_count = fail_count + 1;
        end
    end
endtask

// ── Main test sequence ─────────────────────────────────────
integer i;
integer lane_prod, expected_sum, expected_result;

initial begin
    pass_count = 0;
    fail_count = 0;

    // --------------------------------------------------------
    // TC1: All zeros
    //   a=0, w=0, alpha_x=1, alpha_w=1, beta_xw=0
    //   expected: 1*1*(64 * 3 * 0*0) + 0 = 0
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b000); fill_b(3'b000);
    alpha_x = 3'd1; alpha_w = 3'd1; beta_xw = 6'sd0;
    do_accumulate;
    check("TC1: all zeros", 21'sd0);

    // --------------------------------------------------------
    // TC2: All +1 activations, all +1 weights
    //   Each lane: xd=1, wd=1, product=1 accumulated 3 times → acc=3
    //   Tree sum: 64 × 3 = 192
    //   alpha_x=1, alpha_w=1 → scaled = 1*1*192 = 192
    //   beta_xw=0 → result = 192
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b001); fill_b(3'b001);
    alpha_x = 3'd1; alpha_w = 3'd1; beta_xw = 6'sd0;
    do_accumulate;
    check("TC2: all +1x+1", 21'sd192);

    // --------------------------------------------------------
    // TC3: All -1 activations (3'b111), all +1 weights
    //   product = -1*1 = -1, accumulated 3 times → acc = -3
    //   Tree sum: 64 × (-3) = -192
    //   alpha_x=1, alpha_w=1, beta_xw=0 → result = -192
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b111); fill_b(3'b001);
    alpha_x = 3'd1; alpha_w = 3'd1; beta_xw = 6'sd0;
    do_accumulate;
    check("TC3: all -1x+1", -21'sd192);

    // --------------------------------------------------------
    // TC4: Max magnitude inputs
    //   xd = +3 (3'b011), wd = -4 (3'b100)
    //   product = 3 * (-4) = -12, accumulated 3 times → acc = -36
    //   Tree sum: 64 × (-36) = -2304
    //   alpha_x=3, alpha_w=3 → alpha_prod = 9
    //   scaled = 9 * (-2304) = -20736
    //   beta_xw = 0 → result = -20736
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b011); fill_b(3'b100);
    alpha_x = 3'd3; alpha_w = 3'd3; beta_xw = 6'sd0;
    do_accumulate;
    check("TC4: max magnitude", -21'sd20736);

    // --------------------------------------------------------
    // TC5: Reset mid-accumulation
    //   Start accumulating, reset after 1 cycle, finish — result = beta_xw
    // --------------------------------------------------------
    rst = 0; load = 0;
    fill_a(3'b001); fill_b(3'b001);
    alpha_x = 3'd1; alpha_w = 3'd1; beta_xw = 6'sd5;
    // 1 accumulation cycle
    load = 1; @(posedge clk); #1;
    // mid-reset
    rst = 1; @(posedge clk); #1;
    rst = 0;
    // 2 more cycles (but acc was cleared, so only these 2 count... actually
    // acc=0 after rst so 2 cycles of 1*1=1 → acc=2 per lane)
    @(posedge clk); #1;
    @(posedge clk); #1;
    load = 0;
    @(posedge clk); #1;
    // After reset: acc held 0, then 2 cycles of product=1 → acc=2 per lane
    // sum = 64*2=128, scaled=1*1*128=128, +beta_xw=5 → 133
    check("TC5: reset mid-accum", 21'sd133);

    // --------------------------------------------------------
    // TC6: alpha_x = 0 → scaled = 0, result = beta_xw
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b011); fill_b(3'b011);
    alpha_x = 3'd0; alpha_w = 3'd3; beta_xw = 6'sd7;
    do_accumulate;
    check("TC6: alpha_x=0", 21'sd7);

    // --------------------------------------------------------
    // TC7: alpha_w = 0 → scaled = 0, result = beta_xw
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b011); fill_b(3'b011);
    alpha_x = 3'd3; alpha_w = 3'd0; beta_xw = -6'sd8;
    do_accumulate;
    check("TC7: alpha_w=0", -21'sd8);

    // --------------------------------------------------------
    // TC8: Single non-zero lane (lane 0 only)
    //   lane 0: xd=2 (3'b010), wd=2 (3'b010), product=4, acc=12
    //   all other lanes: xd=0, wd=0 → acc=0
    //   Tree sum = 12
    //   alpha_x=2, alpha_w=2 → alpha_prod=4
    //   scaled = 4*12 = 48
    //   beta_xw = 0 → result = 48
    // --------------------------------------------------------
    do_reset;
    a_flat = 192'b0; b_flat = 192'b0;
    a_flat[2:0] = 3'b010;   // lane 0 xd = +2
    b_flat[2:0] = 3'b010;   // lane 0 wd = +2
    alpha_x = 3'd2; alpha_w = 3'd2; beta_xw = 6'sd0;
    do_accumulate;
    check("TC8: single lane", 21'sd48);

    // --------------------------------------------------------
    // TC9: Positive beta_xw offset with zero inputs
    //   All inputs 0, alpha=1, beta_xw=+31 (max positive 6-bit signed)
    //   result = 0 + 31 = 31
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b000); fill_b(3'b000);
    alpha_x = 3'd1; alpha_w = 3'd1; beta_xw = 6'sd31;
    do_accumulate;
    check("TC9: positive beta offset", 21'sd31);

    // --------------------------------------------------------
    // TC10: Negative beta_xw offset
    //   All inputs 0, alpha=1, beta_xw=-32 (most negative 6-bit signed)
    //   result = 0 + (-32) = -32
    // --------------------------------------------------------
    do_reset;
    fill_a(3'b000); fill_b(3'b000);
    alpha_x = 3'd1; alpha_w = 3'd1; beta_xw = -6'sd32;
    do_accumulate;
    check("TC10: negative beta offset", -21'sd32);

    // ── Summary ───────────────────────────────────────────
    $display("--------------------------------------------------");
    $display("Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
    $display("--------------------------------------------------");

    $finish;
end

// ── Timeout watchdog ───────────────────────────────────────
initial begin
    #10000;
    $display("TIMEOUT — simulation exceeded 10000 ns");
    $finish;
end

// ── Waveform dump ──────────────────────────────────────────
initial begin
    $dumpfile("tb_int_mac_M3_N64.vcd");
    $dumpvars(0, tb_int_mac_M3_N64);
end

endmodule