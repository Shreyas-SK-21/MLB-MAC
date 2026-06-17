`timescale 1ns/1ps
module tb_mlb_mac;
integer pass_count;
integer fail_count;

task check;
    input [63:0]  got;
    input [63:0]  expected;
    input [255:0] label;
    begin
        if ($signed(got) === $signed(expected)) begin
            $display("  PASS  %0s  (got=%0d)", label, $signed(got));
            pass_count = pass_count + 1;
        end else begin
            $display("  FAIL  %0s  got=%0d  expected=%0d",
                     label, $signed(got), $signed(expected));
            fail_count = fail_count + 1;
        end
    end
endtask

// ---------------------------------------------------------------------------
// DUT signals
// ---------------------------------------------------------------------------

// --- xnor_popcount_4_bit (N=64 implementation) ---
reg  [63:0] xp_a, xp_b;
wire signed [7:0] xp_out;

// --- basis_multiplier ---
reg  signed [7:0] bm_xp;
reg  [3:0] bm_ax, bm_aw;
wire signed [16:0] bm_out;

// --- MLB_unit ---
reg  [3:0] mu_ax, mu_aw;
reg  [63:0] mu_axi, mu_awi;
wire signed [16:0] mu_out;

// --- MLB_4 ---
reg  [15:0] m4_ax, m4_aw;
reg  [255:0] m4_xi, m4_wi;
wire signed [20:0] m4_out;

// ---------------------------------------------------------------------------
// Instantiations
// ---------------------------------------------------------------------------

xnor_popcount_4_bit dut_xp (
    .signed_output(xp_out),
    .a(xp_a), .b(xp_b)
);

basis_multiplier dut_bm (
    .basis_mult(bm_out),
    .xnor_popcount(bm_xp),
    .alpha_x(bm_ax), .alpha_w(bm_aw)
);

MLB_unit dut_mu (
    .out(mu_out),
    .alpha_x(mu_ax), .alpha_w(mu_aw),
    .axi(mu_axi), .awi(mu_awi)
);

MLB_4 dut_m4 (
    .mlb(m4_out),
    .alpha_x(m4_ax), .alpha_w(m4_aw),
    .axi(m4_xi), .awi(m4_wi)
);

// ---------------------------------------------------------------------------
// VCD 
// ---------------------------------------------------------------------------
initial begin
    $dumpfile("mlb_mac_sim.vcd");
    $dumpvars(0, tb_mlb_mac);
end

// ---------------------------------------------------------------------------
// Main test sequence
// ---------------------------------------------------------------------------
initial begin
    pass_count = 0;
    fail_count = 0;

    // Default inputs to 0
    {xp_a, xp_b}           = 128'h0;
    {bm_xp, bm_ax, bm_aw}  = 16'h0000;
    {mu_ax, mu_aw}         = 8'h00;
    {mu_axi, mu_awi}       = 128'h0;
    {m4_ax, m4_aw}         = 32'h0;
    {m4_xi, m4_wi}         = 512'h0;
    #10;

    // =====================================================================
    // 1. XNOR_POPCOUNT_4_BIT (64-bit elements)
    //    Output = 2P - 64
    // =====================================================================
    $display("\n--- 1. xnor_popcount_4_bit (N=64) ---");

    // TC1.1: P=64 (all match) → xp = +64
    xp_a=64'h0; xp_b=64'h0; #10;
    check({{56{xp_out[7]}},xp_out}, 64'd64, "XP TC1.1 P=64 a=0 b=0 → +64");

    // TC1.2: P=64 (all 1s match) → xp = +64
    xp_a={64{1'b1}}; xp_b={64{1'b1}}; #10;
    check({{56{xp_out[7]}},xp_out}, 64'd64, "XP TC1.2 P=64 a=~0 b=~0 → +64");

    // TC1.3: P=63 → xp = +62 (1 mismatch at bit 0)
    xp_a=64'h0; xp_b=64'h1; #10;
    check({{56{xp_out[7]}},xp_out}, 64'd62, "XP TC1.3 P=63 a=0 b=1 → +62");

    // TC1.4: P=32 → xp = 0 (exactly half bits match)
    xp_a=64'h0; xp_b=64'h00000000FFFFFFFF; #10;
    check({{56{xp_out[7]}},xp_out}, 64'd0, "XP TC1.4 P=32 Half-match → 0");

    // TC1.5: P=0 → xp = -64 (all mismatched)
    xp_a={64{1'b1}}; xp_b=64'h0; #10;
    check({{56{xp_out[7]}},xp_out}, -64'd64, "XP TC1.5 P=0 a=~0 b=0 → -64");

    // =====================================================================
    // 2. BASIS_MULTIPLIER
    //    out = xnor_popcount × (alpha_x × alpha_w)
    //    Output: signed [16:0]
    // =====================================================================
    $display("\n--- 2. basis_multiplier ---");

    // TC2.1: xp=0 → 0
    bm_xp=8'sd0; bm_ax=4'hF; bm_aw=4'hF; #10;
    check({{47{bm_out[16]}},bm_out}, 64'd0, "BM TC2.1 xp=0 ax=15 aw=15 → 0");

    // TC2.2: xp=+64, ax=1, aw=1 → 64×1=64
    bm_xp=8'sd64; bm_ax=4'h1; bm_aw=4'h1; #10;
    check({{47{bm_out[16]}},bm_out}, 64'd64, "BM TC2.2 xp=+64 ax=1 aw=1 → 64");

    // TC2.3: xp=+64, ax=3, aw=5 → 64×15=960
    bm_xp=8'sd64; bm_ax=4'h3; bm_aw=4'h5; #10;
    check({{47{bm_out[16]}},bm_out}, 64'd960, "BM TC2.3 xp=+64 ax=3 aw=5 → 960");

    // TC2.4: xp=-64, ax=3, aw=5 → -64×15=-960
    bm_xp=-8'sd64; bm_ax=4'h3; bm_aw=4'h5; #10;
    check({{47{bm_out[16]}},bm_out}, -64'd960, "BM TC2.4 xp=-64 ax=3 aw=5 → -960");

    // TC2.5: xp=+64, ax=15, aw=15 → 64×225=14400 
    bm_xp=8'sd64; bm_ax=4'hF; bm_aw=4'hF; #10;
    check({{47{bm_out[16]}},bm_out}, 64'd14400, "BM TC2.5 xp=+64 ax=15 aw=15 → 14400");

    // TC2.6: xp=-64, ax=15, aw=15 → -64×225=-14400 
    bm_xp=-8'sd64; bm_ax=4'hF; bm_aw=4'hF; #10;
    check({{47{bm_out[16]}},bm_out}, -64'd14400, "BM TC2.6 xp=-64 ax=15 aw=15 → -14400");

    // =====================================================================
    // 3. MLB_UNIT
    //    out = (2P - 64) × alpha_x × alpha_w, signed [16:0]
    // =====================================================================
    $display("\n--- 3. MLB_unit ---");

    // TC3.1: axi=awi=0, alpha=1 → P=64, xp=+64, out=+64
    mu_ax=4'h1; mu_aw=4'h1; mu_axi=64'h0; mu_awi=64'h0; #10;
    check({{47{mu_out[16]}},mu_out}, 64'd64, "MU TC3.1 all-0 bits alpha=1 → +64");

    // TC3.2: axi=awi=~0, alpha=1 → P=64, xp=+64, out=+64
    mu_ax=4'h1; mu_aw=4'h1; mu_axi={64{1'b1}}; mu_awi={64{1'b1}}; #10;
    check({{47{mu_out[16]}},mu_out}, 64'd64, "MU TC3.2 all-1 bits alpha=1 → +64");

    // TC3.3: axi=~0, awi=0, alpha=1 → P=0, xp=-64, out=-64
    mu_ax=4'h1; mu_aw=4'h1; mu_axi={64{1'b1}}; mu_awi=64'h0; #10;
    check({{47{mu_out[16]}},mu_out}, -64'd64, "MU TC3.3 anti-corr bits alpha=1 → -64");

    // TC3.4: axi=awi=~0, ax=3, aw=5 → xp=+64, 64×15=960
    mu_ax=4'h3; mu_aw=4'h5; mu_axi={64{1'b1}}; mu_awi={64{1'b1}}; #10;
    check({{47{mu_out[16]}},mu_out}, 64'd960, "MU TC3.4 xp=+64 ax=3 aw=5 → 960");

    // TC3.5: xp=+64, ax=15, aw=15 → 64×225=14400
    mu_ax=4'hF; mu_aw=4'hF; mu_axi={64{1'b1}}; mu_awi={64{1'b1}}; #10;
    check({{47{mu_out[16]}},mu_out}, 64'd14400, "MU TC3.5 xp=+64 ax=15 aw=15 → 14400");

    // =====================================================================
    // 4. MLB_4  (4×4 array processing 64-bit blocks)
    //    MLB_4 output = sum_{i=0}^{3} sum_{j=0}^{3} MLB_unit(i,j)
    // =====================================================================
    $display("\n--- 4. MLB_4 (top-level) ---");

    // TC4.1: All inputs zero → output 0
    m4_ax=16'h0000; m4_aw=16'h0000;
    m4_xi={256{1'b0}}; m4_wi={256{1'b0}}; #10;
    check({{43{m4_out[20]}},m4_out}, 64'd0,
          "M4 TC4.1 all-zero inputs/alphas → 0");

    // TC4.2: All bit-planes=1, all alpha=1
    // Each of 16 units: P=64, xp=+64, 64×1×1=64 → total = 16×64 = 1024
    m4_ax=16'h1111; m4_aw=16'h1111;
    m4_xi={256{1'b1}}; m4_wi={256{1'b1}}; #10;
    check({{43{m4_out[20]}},m4_out}, 64'd1024,
          "M4 TC4.2 all-1 bits alpha=1 → 1024");

    // TC4.3: Anti-correlated bits, alpha=1
    // Every pair: P=0, xp=-64 → total = 16×(-64) = -1024
    m4_ax=16'h1111; m4_aw=16'h1111;
    m4_xi={256{1'b1}}; m4_wi={256{1'b0}}; #10;
    check({{43{m4_out[20]}},m4_out}, -64'd1024,
          "M4 TC4.3 anti-corr axi=1 awi=0 alpha=1 → -1024");

    // TC4.4: Single active unit at (0,0), alpha_x[0]=3, alpha_w[0]=5
    // Only unit(0,0) nonzero: xp=+64, 64×3×5=960
    m4_ax=16'h0003; m4_aw=16'h0005;
    m4_xi={256{1'b1}}; m4_wi={256{1'b1}}; #10;
    check({{43{m4_out[20]}},m4_out}, 64'd960,
          "M4 TC4.4 single unit(0,0) ax=3 aw=5 xp=+64 → 960");

    // TC4.5: Two active channels with opposing signs
    // Channel 0 (bits [63:0]): 11...1 vs 11...1 → match
    // Channel 1 (bits [127:64]): 00...0 vs 11...1 → mismatch
    m4_ax=16'h0021; m4_aw=16'h0011;
    m4_xi={ 128'b0, 64'b0, {64{1'b1}} }; 
    m4_wi={ 128'b0, {64{1'b1}}, {64{1'b1}} }; #10;
    // unit(0,0): +64
    // unit(0,1): +64
    // unit(1,0): -64 × 2 × 1 = -128
    // unit(1,1): -64 × 2 × 1 = -128
    // Total = 64+64-128-128 = -128
    check({{43{m4_out[20]}},m4_out}, -64'd128,
          "M4 TC4.5 mixed-sign channels → -128");

    // TC4.6: All four channels active, equal weight
    m4_ax=16'h1111; m4_aw=16'h2222;
    m4_xi={256{1'b1}}; m4_wi={256{1'b1}}; #10;
    // 16 units × (64 × 1 × 2) = 16 × 128 = 2048
    check({{43{m4_out[20]}},m4_out}, 64'd2048,
          "M4 TC4.6 ax=1,1,1,1 aw=2,2,2,2 xp=+64 → 2048");

    // TC4.7: Maximum possible output bound verification (Saturation Check)
    m4_ax=16'hFFFF; m4_aw=16'hFFFF;
    m4_xi={256{1'b1}}; m4_wi={256{1'b1}}; #10;
    check({{43{m4_out[20]}},m4_out}, 64'd230400,
          "M4 TC4.7 maximum bound sum (16x14400) → 230400");

    // =====================================================================
    // Final report
    // =====================================================================
    $display("\n======================================================");
    $display("  TEST SUMMARY: %0d PASSED,  %0d FAILED",
             pass_count, fail_count);
    $display("======================================================");
    if (fail_count == 0)
        $display("  ALL TESTS PASSED");
    else
        $display("  *** FAILURES DETECTED — check waveform in mlb_mac_sim.vcd ***");
    $display("");

    #20;
    $finish;
end

endmodule