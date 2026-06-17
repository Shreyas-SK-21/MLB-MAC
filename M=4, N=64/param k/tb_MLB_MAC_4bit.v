
// =============================================================================
// Testbench : tb_MLB_MAC_4bit
// DUT       : MLB_4  (M=4, N=64, parameterized K)
// Paper ref : MLB-MAC, Fig. 6 — xnor-popcnt + Acc (Part 1),
//             Basis-Multiplier + Reduction tree (Part 2)
//
// Architecture recap (Fig. 6):
//   - M=4 bits  → 4×4 = 16 MLB_unit cells, each = xnor_popcount_64 + basis_mult
//   - N=64      → each MLB_unit processes 64 activation-weight pairs per clock
//   - K         → dot-product length; MLB_unit accumulates over ceil(K/64) cycles
//   - alpha_x[15:0]  → four 4-bit bases for activations  (α_x1..α_x4, packed)
//   - alpha_w[15:0]  → four 4-bit bases for weights      (α_w1..α_w4, packed)
//   - axi[255:0]     → 4 bit-planes of activation, each 64 bits wide
//   - awi[255:0]     → 4 bit-planes of weights,     each 64 bits wide
//   - mlb[28:0]      → signed dot-product result
//   - done           → pulses high for one cycle when mlb is valid
// =============================================================================

`timescale 1ns/1ps

module tb_MLB_MAC_4bit;

    // -------------------------------------------------------------------------
    // Parameters  — only K needs to change between test cases
    // -------------------------------------------------------------------------
    parameter M     = 4;          // data width (bits per value)
    parameter N     = 64;         // parallelism / MAC-array width  (hardwired in DUT)
    parameter CLK_P = 4;          // clock period in ns  → 250 MHz

    // K must be a multiple of N=64 so that cycles_needed = K/64 is exact.
    parameter K_TEST0 = 64;       // K == N  (single cycle)
    parameter K_TEST1 = 128;      // K  = 2N
    parameter K_TEST2 = 256;
    parameter K_TEST3 = 512;
    parameter K_TEST4 = 1024;
    parameter K_TEST5 = 2048;     // representative large Conv2D dot-product

    // -------------------------------------------------------------------------
    // DUT ports
    // -------------------------------------------------------------------------
    wire signed [28:0] mlb;
    wire               done;
    reg  [15:0]        alpha_x, alpha_w;   // 4 × 4-bit bases, packed
    reg  [255:0]       axi, awi;           // 4 × 64-bit binary planes, packed
    reg  [12:0]        K;
    reg                clk, rst, valid_in;

    // -------------------------------------------------------------------------
    // DUT instantiation
    // -------------------------------------------------------------------------
    MLB_4 dut (
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

    // -------------------------------------------------------------------------
    // Clock generation  (250 MHz)
    // -------------------------------------------------------------------------
    initial clk = 0;
    always #(CLK_P/2) clk = ~clk;

    // -------------------------------------------------------------------------
    // Bookkeeping
    // -------------------------------------------------------------------------
    integer pass_cnt = 0, fail_cnt = 0, test_id = 0;
    integer cycle_start, cycle_end;
    real    latency_ns;

    // -------------------------------------------------------------------------
    // Reference model
    // -------------------------------------------------------------------------
    function automatic signed [28:0] reference_dot;
        input [15:0]  ax, aw;           // packed 4-bit bases
        input [255:0] bx, bw;           // packed 64-bit binary planes
        input [12:0]  Kv;
        
        integer ii, jj, kk, bit_idx;
        reg [63:0]  plane_x, plane_w, xnn;
        reg [6:0]   P;
        reg signed [7:0]  contrib;
        reg signed [15:0] xp_out;
        reg signed [24:0] cell_out;
        reg signed [28:0] acc;
        reg [7:0]   alpha_prod;
        reg signed [8:0] alpha_prod_s;
        
        begin
            acc = 29'sd0;
            for (ii = 0; ii < M; ii = ii+1) begin
                for (jj = 0; jj < M; jj = jj+1) begin
                    // --- xnor_popcount over all K bits (N=64 per cycle) ---
                    xp_out = 16'sd0;
                    for (kk = 0; kk < Kv; kk = kk+64) begin
                        plane_x = bx[ii*64 +: 64];   
                        plane_w = bw[jj*64 +: 64];   
                        xnn     = ~(plane_x ^ plane_w);
                        
                        // Universal popcount compatible with standard Verilog
                        P = 0;
                        for (bit_idx = 0; bit_idx < 64; bit_idx = bit_idx + 1) begin
                            P = P + xnn[bit_idx];
                        end
                        
                        contrib = {1'b0, P, 1'b0} - 8'd64; // 2P - 64
                        xp_out  = xp_out + {{8{contrib[7]}}, contrib};
                    end
                    // --- basis multiplier ---
                    alpha_prod   = ax[ii*4 +: 4] * aw[jj*4 +: 4];
                    alpha_prod_s = {1'b0, alpha_prod};
                    cell_out     = alpha_prod_s * xp_out;
                    acc          = acc + cell_out;
                end
            end
            reference_dot = acc;
        end
    endfunction

    // -------------------------------------------------------------------------
    // Task : run one test vector
    // -------------------------------------------------------------------------
    task automatic run_test;
        input [12:0]  Kv;
        input [15:0]  ax, aw;
        input [255:0] bx, bw;
        input [28:0]  expected;

        integer cyc;
        integer timeout;
    begin
        test_id   = test_id + 1;
        K         = Kv;
        alpha_x   = ax;
        alpha_w   = aw;
        axi       = bx;
        awi       = bw;
        valid_in  = 0;

        // Reset
        rst = 1;
        @(posedge clk); #1;
        @(posedge clk); #1;
        rst = 0;

        // Drive valid_in for K/N cycles
        cycle_start = $time;
        cyc = 0;
        repeat (Kv / N) begin
            valid_in = 1;
            @(posedge clk); #1;
            cyc = cyc + 1;
        end
        valid_in = 0;

        // Wait for done with timeout
        timeout = 0;
        while (!done && timeout < 50) begin
            @(posedge clk); #1;
            timeout = timeout + 1;
        end
        cycle_end   = $time;
        latency_ns  = (cycle_end - cycle_start) * 1.0;

        // Check against Reference Model
        if (done) begin
            if (mlb === expected) begin
                $display("[PASS] Test %0d | K=%0d  mlb=%0d (expected %0d) | latency=%.1f ns",
                          test_id, Kv, mlb, expected, latency_ns);
                pass_cnt = pass_cnt + 1;
            end else begin
                $display("[FAIL] Test %0d | K=%0d  mlb=%0d  expected=%0d | latency=%.1f ns",
                          test_id, Kv, $signed(mlb), $signed(expected), latency_ns);
                fail_cnt = fail_cnt + 1;
            end
        end else begin
            $display("[FAIL] Test %0d | K=%0d  done never asserted (timeout)", test_id, Kv);
            fail_cnt = fail_cnt + 1;
        end

        repeat (4) @(posedge clk);
    end
    endtask

    // -------------------------------------------------------------------------
    // Helper : compute expected value on the fly
    // -------------------------------------------------------------------------
    `define EXPECT(ax,aw,bx,bw,kv) reference_dot(ax, aw, bx, bw, kv)

    // -------------------------------------------------------------------------
    // Stimulus
    // -------------------------------------------------------------------------
    initial begin
        $dumpfile("tb_MLB_MAC_4bit.vcd");
        $dumpvars(0, tb_MLB_MAC_4bit);

        clk = 0; rst = 1; valid_in = 0;
        alpha_x = 0; alpha_w = 0; axi = 0; awi = 0; K = 0;
        repeat (4) @(posedge clk);

        $display("=============================================================");
        $display(" MLB-MAC Testbench  |  M=%0d  N=%0d  (Fig. 6, paper)", M, N);
        $display("=============================================================");

        // -----------------------------------------------------------------
        // GROUP 1 — Sweep K  with all-ones binary planes
        // -----------------------------------------------------------------
        $display("\n-- Group 1: all-ones planes, integer bases (α=[1,2,4,8]) --");
        begin : g1
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd8, 4'd4, 4'd2, 4'd1};   
            aw = {4'd8, 4'd4, 4'd2, 4'd1};
            bx = {4{64'hFFFF_FFFF_FFFF_FFFF}};
            bw = {4{64'hFFFF_FFFF_FFFF_FFFF}};

            run_test(K_TEST0, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST0));
            run_test(K_TEST1, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST1));
            run_test(K_TEST2, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST2));
            run_test(K_TEST3, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST3));
            run_test(K_TEST4, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST4));
            run_test(K_TEST5, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST5));
        end

        // -----------------------------------------------------------------
        // GROUP 2 — All-zeros binary planes
        // -----------------------------------------------------------------
        $display("\n-- Group 2: all-zeros planes (xnor still all-1) --");
        begin : g2
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd8, 4'd4, 4'd2, 4'd1};
            aw = {4'd8, 4'd4, 4'd2, 4'd1};
            bx = {4{64'h0000_0000_0000_0000}};
            bw = {4{64'h0000_0000_0000_0000}};
            run_test(K_TEST0, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST0));
            run_test(K_TEST2, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST2));
        end

        // -----------------------------------------------------------------
        // GROUP 3 — Opposing planes: bx=all-ones, bw=all-zeros
        // -----------------------------------------------------------------
        $display("\n-- Group 3: opposing planes (bx=1s, bw=0s) → negative output --");
        begin : g3
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd8, 4'd4, 4'd2, 4'd1};
            aw = {4'd8, 4'd4, 4'd2, 4'd1};
            bx = {4{64'hFFFF_FFFF_FFFF_FFFF}};
            bw = {4{64'h0000_0000_0000_0000}};
            run_test(K_TEST0, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST0));
            run_test(K_TEST2, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST2));
            run_test(K_TEST4, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST4));
        end

        // -----------------------------------------------------------------
        // GROUP 4 — Checkerboard pattern  (alternating 0101…)
        // -----------------------------------------------------------------
        $display("\n-- Group 4: checkerboard pattern → zero dot-product --");
        begin : g4
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd8, 4'd4, 4'd2, 4'd1};
            aw = {4'd8, 4'd4, 4'd2, 4'd1};
            bx = {4{64'hAAAA_AAAA_AAAA_AAAA}}; 
            bw = {4{64'h5555_5555_5555_5555}}; 
            run_test(K_TEST0, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST0));
            run_test(K_TEST2, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST2));
        end

        // -----------------------------------------------------------------
        // GROUP 5 — Non-uniform (MLB) bases 
        // -----------------------------------------------------------------
        $display("\n-- Group 5: non-uniform MLB bases (α_x={3,5,7,9}, α_w={2,4,6,8}) --");
        begin : g5
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd9, 4'd7, 4'd5, 4'd3};
            aw = {4'd8, 4'd6, 4'd4, 4'd2};
            bx = {4{64'hFFFF_FFFF_FFFF_FFFF}};
            bw = {4{64'hFFFF_FFFF_FFFF_FFFF}};
            run_test(K_TEST0, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST0));
            run_test(K_TEST1, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST1));
            run_test(K_TEST3, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST3));
        end

        // -----------------------------------------------------------------
        // GROUP 6 — Mixed planes
        // -----------------------------------------------------------------
        $display("\n-- Group 6: mixed per-plane patterns --");
        begin : g6
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd8, 4'd4, 4'd2, 4'd1};
            aw = {4'd8, 4'd4, 4'd2, 4'd1};
            bx = {64'hAAAA_AAAA_AAAA_AAAA,
                  64'h5555_5555_5555_5555,
                  64'h0000_0000_0000_0000,
                  64'hFFFF_FFFF_FFFF_FFFF};
            bw = {64'h5555_5555_5555_5555,
                  64'hFFFF_FFFF_FFFF_FFFF,
                  64'hFFFF_FFFF_FFFF_FFFF,
                  64'h0000_0000_0000_0000};
            run_test(K_TEST0, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST0));
            run_test(K_TEST2, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST2));
            run_test(K_TEST4, ax, aw, bx, bw, `EXPECT(ax,aw,bx,bw,K_TEST4));
        end

        // -----------------------------------------------------------------
        // GROUP 7 — Unit bases (α=1 for all)
        // -----------------------------------------------------------------
        $display("\n-- Group 7: unit bases (α=1) sanity check --");
        begin : g7
            reg [15:0] ax, aw;
            reg [255:0] bx, bw;
            ax = {4'd1, 4'd1, 4'd1, 4'd1};
            aw = {4'd1, 4'd1, 4'd1, 4'd1};
            bx = {4{64'hFFFF_FFFF_FFFF_FFFF}};
            bw = {4{64'hFFFF_FFFF_FFFF_FFFF}};
            run_test(K_TEST0, ax, aw, bx, bw, 29'sd1024);   
        end

        // -----------------------------------------------------------------
        // Summary
        // -----------------------------------------------------------------
        $display("\n=============================================================");
        $display(" Results: %0d PASSED  |  %0d FAILED  |  %0d TOTAL",
                  pass_cnt, fail_cnt, test_id);
        $display("=============================================================\n");

        if (fail_cnt == 0)
            $display(" *** ALL TESTS PASSED ***");
        else
            $display(" *** %0d TEST(S) FAILED — check DUT bugs listed above ***", fail_cnt);

        $finish;
    end

    // -------------------------------------------------------------------------
    // Watchdog
    // -------------------------------------------------------------------------
    initial begin
        #500000;
        $display("[WATCHDOG] Simulation timed out at %0t ns", $time);
        $finish;
    end

endmodule