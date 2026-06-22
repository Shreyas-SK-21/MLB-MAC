// tb_int_mac_64.v
// ============================================================
// Testbench for int_mac_64  (M=4, N=64)
//
// Test plan:
//   TC1  — All zeros                          → result = 0
//   TC2  — All lanes +1×+1                   → 64×4×1  = 256, scaled by αx=αw=1
//   TC3  — All lanes −1×+1                   → −256
//   TC4  — Max positive  (+7 × +7)           → 64×4×49, αx=αw=7
//   TC5  — Max negative  (+7 × −8)           → 64×4×(−56), αx=αw=1
//   TC6  — Reset mid-accumulation            → acc clears, only post-reset cycles count
//   TC7  — alpha_x = 0                       → result = beta_xw only
//   TC8  — alpha_w = 0                       → result = beta_xw only
//   TC9  — Single active lane (lane 0 only)  → only lane 0 contributes
//   TC10 — Max positive beta_xw (+127)       → zero inputs, offset = +127
//   TC11 — Max negative beta_xw (−128)       → zero inputs, offset = −128
//   TC12 — Alternating +1/−1 across lanes   → sum cancels to 0
// ============================================================

`timescale 1ns/1ps

module tb_int_mac_64;

// ── DUT ports ──────────────────────────────────────────────
reg          clk;
reg          rst;
reg          load;
reg  [255:0] a_flat;
reg  [255:0] b_flat;
reg  [3:0]   alpha_x;
reg  [3:0]   alpha_w;
reg  [7:0]   beta_xw;

wire signed [20:0] result;

// ── DUT instantiation ──────────────────────────────────────
int_mac_64 dut (
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

// ── Counters ───────────────────────────────────────────────
integer pass_count, fail_count;

// ── Tasks ──────────────────────────────────────────────────

// Fill all 64 lanes of a_flat with the same 4-bit value
task fill_a;
    input [3:0] val;
    integer k;
    begin
        for (k = 0; k < 64; k = k + 1)
            a_flat[4*k +: 4] = val;
    end
endtask

// Fill all 64 lanes of b_flat with the same 4-bit value
task fill_b;
    input [3:0] val;
    integer k;
    begin
        for (k = 0; k < 64; k = k + 1)
            b_flat[4*k +: 4] = val;
    end
endtask

// Synchronous reset — holds for 2 cycles
task do_reset;
    begin
        rst  = 1;
        load = 0;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst  = 0;
    end
endtask

// Drive load HIGH for exactly 4 cycles (M=4 accumulation cycles)
// a_flat / b_flat must be set before calling
task do_accumulate;
    begin
        load = 1;
        @(posedge clk); #1;   // cycle 1
        @(posedge clk); #1;   // cycle 2
        @(posedge clk); #1;   // cycle 3
        @(posedge clk); #1;   // cycle 4
        load = 0;
        @(posedge clk); #1;   // settle
    end
endtask

// Check and print PASS/FAIL
task check;
    input [79:0] tc_name;   // up to 10 chars
    input signed [20:0] expected;
    begin
        if (result === expected) begin
            $display("PASS  %-32s | expected=%0d  got=%0d",
                     tc_name, $signed(expected), $signed(result));
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL  %-32s | expected=%0d  got=%0d",
                     tc_name, $signed(expected), $signed(result));
            fail_count = fail_count + 1;
        end
    end
endtask

// ── Main test sequence ─────────────────────────────────────
integer k;

initial begin
    pass_count = 0;
    fail_count = 0;
    a_flat = 0; b_flat = 0;
    alpha_x = 0; alpha_w = 0; beta_xw = 0;
    rst = 0; load = 0;

    // --------------------------------------------------------
    // TC1: All zeros
    //   product = 0, acc = 0, sum = 0, scaled = 0, result = 0
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0000); fill_b(4'b0000);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = 8'sd0;
    do_accumulate;
    check("TC1: all zeros", 21'sd0);

    // --------------------------------------------------------
    // TC2: All lanes +1 × +1, αx=αw=1, β=0
    //   per lane: product=1, acc after 4 cycles = 4
    //   tree sum = 64 × 4 = 256
    //   scaled = 1 × 1 × 256 = 256
    //   result = 256
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0001); fill_b(4'b0001);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = 8'sd0;
    do_accumulate;
    check("TC2: all +1x+1", 21'sd256);

    // --------------------------------------------------------
    // TC3: All lanes −1 × +1, αx=αw=1, β=0
    //   4'b1111 = −1 in signed 4-bit
    //   product = −1, acc = −4 after 4 cycles
    //   tree sum = 64 × (−4) = −256
    //   result = −256
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b1111); fill_b(4'b0001);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = 8'sd0;
    do_accumulate;
    check("TC3: all -1x+1", -21'sd256);

    // --------------------------------------------------------
    // TC4: Max positive product (+7 × +7), αx=αw=7, β=0
    //   4'b0111 = +7
    //   product = 49, acc = 196 after 4 cycles
    //   tree sum = 64 × 196 = 12544
    //   alpha_prod = 7 × 7 = 49
    //   scaled = 49 × 12544 = 614656
    //   result = 614656  (fits in 21 bits signed: max ~1M)
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0111); fill_b(4'b0111);
    alpha_x = 4'd7; alpha_w = 4'd7; beta_xw = 8'sd0;
    do_accumulate;
    check("TC4: max pos +7x+7 a=w=7", 21'sd614656);

    // --------------------------------------------------------
    // TC5: Max negative product (+7 × −8), αx=αw=1, β=0
    //   4'b1000 = −8 in signed 4-bit
    //   product = 7 × (−8) = −56, acc = −224 after 4 cycles
    //   tree sum = 64 × (−224) = −14336
    //   scaled = 1 × 1 × (−14336) = −14336
    //   result = −14336
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0111); fill_b(4'b1000);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = 8'sd0;
    do_accumulate;
    check("TC5: max neg +7x-8", -21'sd14336);

    // --------------------------------------------------------
    // TC6: Reset mid-accumulation
    //   Run 2 cycles of +1×+1, then reset, then 2 more cycles
    //   Post-reset acc gets 2 cycles of product=1 → acc=2 per lane
    //   tree sum = 64 × 2 = 128
    //   αx=αw=1, β=0 → result = 128
    // --------------------------------------------------------
    rst = 0; load = 0;
    fill_a(4'b0001); fill_b(4'b0001);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = 8'sd0;
    // 2 cycles before reset
    load = 1; @(posedge clk); #1;
               @(posedge clk); #1;
    // reset
    rst = 1; @(posedge clk); #1;
    rst = 0;
    // 2 cycles after reset (load still high from before)
               @(posedge clk); #1;
               @(posedge clk); #1;
    load = 0;
    @(posedge clk); #1;
    check("TC6: reset mid-accum", 21'sd128);

    // --------------------------------------------------------
    // TC7: alpha_x = 0 → scaled = 0, result = beta_xw
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0111); fill_b(4'b0111);
    alpha_x = 4'd0; alpha_w = 4'd7; beta_xw = 8'sd15;
    do_accumulate;
    check("TC7: alpha_x=0", 21'sd15);

    // --------------------------------------------------------
    // TC8: alpha_w = 0 → scaled = 0, result = beta_xw
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0111); fill_b(4'b0111);
    alpha_x = 4'd7; alpha_w = 4'd0; beta_xw = -8'sd20;
    do_accumulate;
    check("TC8: alpha_w=0", -21'sd20);

    // --------------------------------------------------------
    // TC9: Single active lane — lane 0 only, rest zero
    //   lane 0: xd=3 (4'b0011), wd=3 → product=9, acc=36
    //   all others: 0
    //   tree sum = 36
    //   αx=2, αw=2 → alpha_prod=4
    //   scaled = 4 × 36 = 144
    //   β=0 → result = 144
    // --------------------------------------------------------
    do_reset;
    a_flat = 256'b0; b_flat = 256'b0;
    a_flat[3:0] = 4'b0011;   // lane 0 xd = +3
    b_flat[3:0] = 4'b0011;   // lane 0 wd = +3
    alpha_x = 4'd2; alpha_w = 4'd2; beta_xw = 8'sd0;
    do_accumulate;
    check("TC9: single lane", 21'sd144);

    // --------------------------------------------------------
    // TC10: Max positive beta_xw (+127), zero inputs
    //   result = 0 + 127 = 127
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0000); fill_b(4'b0000);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = 8'sd127;
    do_accumulate;
    check("TC10: beta_xw=+127", 21'sd127);

    // --------------------------------------------------------
    // TC11: Max negative beta_xw (−128), zero inputs
    //   result = 0 + (−128) = −128
    // --------------------------------------------------------
    do_reset;
    fill_a(4'b0000); fill_b(4'b0000);
    alpha_x = 4'd1; alpha_w = 4'd1; beta_xw = -8'sd128;
    do_accumulate;
    check("TC11: beta_xw=-128", -21'sd128);

    // --------------------------------------------------------
    // TC12: Alternating +1/−1 across lanes (even=+1, odd=−1)
    //   products all = ±1, but each acc accumulates 4× same sign
    //   even lanes: acc = +4, odd lanes: acc = −4
    //   32 even + 32 odd → tree sum = 32×4 + 32×(−4) = 0
    //   scaled = 0, result = beta_xw = 5
    // --------------------------------------------------------
    do_reset;
    for (k = 0; k < 64; k = k + 1) begin
        a_flat[4*k +: 4] = (k % 2 == 0) ? 4'b0001 : 4'b1111; // +1 or −1
        b_flat[4*k +: 4] = 4'b0001;                            // always +1
    end
    alpha_x = 4'd3; alpha_w = 4'd3; beta_xw = 8'sd5;
    do_accumulate;
    check("TC12: alternating lanes", 21'sd5);

    // ── Summary ───────────────────────────────────────────
    $display("--------------------------------------------------");
    $display("Results: %0d PASSED, %0d FAILED", pass_count, fail_count);
    $display("--------------------------------------------------");

    $finish;
end

// ── Timeout watchdog ───────────────────────────────────────
initial begin
    #20000;
    $display("TIMEOUT — simulation exceeded 20000 ns");
    $finish;
end

// ── Waveform dump ──────────────────────────────────────────
initial begin
    $dumpfile("tb_int_mac_64.vcd");
    $dumpvars(0, tb_int_mac_64);
end

endmodule
