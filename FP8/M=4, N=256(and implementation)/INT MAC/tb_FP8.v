`timescale 1ns/1ps

module tb_fp8_int_top;

    // ------------------------------------------------------------------
    // DUT signals
    // ------------------------------------------------------------------
    reg  clk, rst, valid_in;
    reg  [2047:0] fp8_activations, fp8_weights;
    wire signed [20:0] wide_integer_sum;
    wire signed [8:0]   shared_exponent;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    // Instantiate the INT MAC top-level wrapper
    fp8_int_top dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .fp8_activations(fp8_activations),
        .fp8_weights(fp8_weights),
        .wide_integer_sum(wide_integer_sum),
        .shared_exponent(shared_exponent)
    );

    // ------------------------------------------------------------------
    // Clock
    // ------------------------------------------------------------------
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------
    // Per-lane stimulus storage
    // ------------------------------------------------------------------
    reg [7:0] act_lane [0:255];
    reg [7:0] wt_lane  [0:255];

    // ------------------------------------------------------------------
    // Waveform dump
    // ------------------------------------------------------------------
    initial begin
        $dumpfile("int_mac_sim.vcd");
        $dumpvars(0, tb_fp8_int_top);
    end

    // ------------------------------------------------------------------
    // Random FP8 (E4M3) lane generator
    // ------------------------------------------------------------------
    function [7:0] rand_fp8(input allow_zero);
        reg [31:0] r1, r2, r3;
        reg [3:0]  exp_f;
        begin
            r1 = $random;
            r2 = $random;
            r3 = $random;
            if (allow_zero)
                exp_f = r1 % 16;
            else
                exp_f = (r1 % 15) + 1;
            rand_fp8 = {r2[0], exp_f, r3[2:0]};
        end
    endfunction

    // ------------------------------------------------------------------
    // Helpers to build directed stimulus
    // ------------------------------------------------------------------
    task clear_all_lanes;
        integer i;
        begin
            for (i = 0; i < 256; i = i + 1) begin
                act_lane[i] = 8'b0;
                wt_lane[i]  = 8'b0;
            end
        end
    endtask

task set_lane(input integer is_weight, input integer idx, input sign, input [3:0] exp_f, input [2:0] mant_f);
        begin
            if (is_weight)
                wt_lane[idx] = {sign, exp_f, mant_f};
            else
                act_lane[idx] = {sign, exp_f, mant_f};
        end
    endtask

task fill_random_lanes(input allow_zero);
        integer i;
        begin
            for (i = 0; i < 256; i = i + 1) begin
                act_lane[i] = rand_fp8(allow_zero);
                wt_lane[i]  = rand_fp8(allow_zero);
            end
        end
    endtask

task pack_vectors;
        integer i;
        begin
            for (i = 0; i < 256; i = i + 1) begin
                fp8_activations[i*8 +: 8] = act_lane[i];
                fp8_weights[i*8 +: 8]     = wt_lane[i];
            end
        end
    endtask

    // ------------------------------------------------------------------
    // Behavioral reference model (UPDATED: sign-after-multiply, no >>1 truncation)
    // ------------------------------------------------------------------
task compute_reference(output reg signed [20:0] ref_result,
                           output reg [8:0]         ref_exponent);
        integer i;
        reg sx, sw;
        reg [3:0] ex, ew;
        reg [3:0] mx, mw;
        reg [3:0] max_ex, max_ew;
        integer shift_x, shift_w;
        reg [3:0] smx, smw;
        
        integer prod;
        integer total_sum;
        begin
            max_ex = 4'd0;
            max_ew = 4'd0;
            for (i = 0; i < 256; i = i + 1) begin
                ex = act_lane[i][6:3];
                ew = wt_lane[i][6:3];
                if (ex > max_ex) max_ex = ex;
                if (ew > max_ew) max_ew = ew;
            end

            total_sum = 0;
            for (i = 0; i < 256; i = i + 1) begin
                sx = act_lane[i][7];
                ex = act_lane[i][6:3];
                mx = (ex != 4'd0) ? {1'b1, act_lane[i][2:0]} : 4'd0;

                sw = wt_lane[i][7];
                ew = wt_lane[i][6:3];
                mw = (ew != 4'd0) ? {1'b1, wt_lane[i][2:0]} : 4'd0;

                shift_x = max_ex - ex;
                shift_w = max_ew - ew;

                smx = mx >> shift_x;
                smw = mw >> shift_w;

                // Full mantissa magnitude used -- no >>1 truncation
                // Sign applied AFTER product (XOR rule), matching new int_mac_256
                prod = smx * smw;
                if (sx ^ sw)
                    total_sum = total_sum - prod;
                else
                    total_sum = total_sum + prod;
            end

            ref_result   = total_sum;
            // No >>1 compensation: same formula as FP8 MLB (14 bias + 6 mantissa scale = 20)
            ref_exponent = $signed({5'b0, max_ex}) + $signed({5'b0, max_ew}) - 9'sd20;
        end
    endtask

    // ------------------------------------------------------------------
    // Apply and Check
    // ------------------------------------------------------------------
task apply_and_check(input [8*40-1:0] name);
        reg signed [20:0] expected_result;
        reg [8:0]         expected_exponent;
        begin
            test_num = test_num + 1;
            pack_vectors;
            compute_reference(expected_result, expected_exponent);

            // 1. Clear the Accumulators in the MAC array
            @(negedge clk);
            rst = 1'b1;
            
            // 2. Release reset and load data
            @(negedge clk);
            rst = 1'b0;
            valid_in = 1'b1;
            
            // 3. Stop loading (let reduction tree combinational logic settle)
            @(negedge clk);
            valid_in = 1'b0;

            // 4. Verify results immediately (Tree adder is combinational off the acc)
            if ((wide_integer_sum !== expected_result) ||
                (shared_exponent !== expected_exponent)) begin
                $display("[TEST %0d] %-0s : FAIL", test_num, name);
                $display("           expected : sum=%0d  exp=%0d", expected_result, expected_exponent);
                $display("           got      : sum=%0d  exp=%0d", wide_integer_sum, shared_exponent);
                fail_count = fail_count + 1;
            end
            else begin
                $display("[TEST %0d] %-0s : PASS  (sum=%0d, exp=%0d)", test_num, name, wide_integer_sum, shared_exponent);
                pass_count = pass_count + 1;
            end

            @(posedge clk);
        end
    endtask

    // ------------------------------------------------------------------
    // Main stimulus
    // ------------------------------------------------------------------
    integer t;
    integer i;
    initial begin
        rst      = 1'b1;
        valid_in = 1'b0;
        fp8_activations = 2048'b0;
        fp8_weights     = 2048'b0;
        clear_all_lanes;

        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        // ---------------- Directed corner cases ----------------
        clear_all_lanes;
        apply_and_check("All zeros");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b0, 4'd7, 3'b000);
        end
        apply_and_check("All positive equal value");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b1, 4'd7, 3'b000);
        end
        apply_and_check("All positive act, all negative weight");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, 1'b1, 4'd7, 3'b011);
            set_lane(1, i, 1'b1, 4'd7, 3'b011);
        end
        apply_and_check("Both negative (signs cancel)");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, i[0], 4'd9, 3'b010);
            set_lane(1, i, 1'b0,  4'd9, 3'b101);
        end
        apply_and_check("Alternating activation sign");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd15, 3'b111);
            set_lane(1, i, 1'b0, 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, all positive");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, i[0],  4'd15, 3'b111);
            set_lane(1, i, ~i[0], 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, alternating signs");

        for (i = 0; i < 256; i = i + 1) begin
            if (i[0]) begin
                set_lane(0, i, 1'b0, 4'd0, 3'b101);
                set_lane(1, i, 1'b0, 4'd0, 3'b011);
            end else begin
                set_lane(0, i, 1'b0, 4'd8, 3'b001);
                set_lane(1, i, 1'b0, 4'd8, 3'b001);
            end
        end
        apply_and_check("Subnormal flush (half lanes exp=0)");

        clear_all_lanes;
        set_lane(0, 0, 1'b0, 4'd10, 3'b110);
        set_lane(1, 0, 1'b1, 4'd10, 3'b110);
        apply_and_check("Single nonzero lane");

        clear_all_lanes;
        set_lane(0, 0, 1'b0, 4'd15, 3'b111);
        set_lane(1, 0, 1'b0, 4'd15, 3'b111);
        for (i = 1; i < 256; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd1, 3'b000);
            set_lane(1, i, 1'b0, 4'd1, 3'b000);
        end
        apply_and_check("Wide dynamic range (shift-to-zero stress)");

        for (i = 0; i < 256; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd0, 3'b111);
            set_lane(1, i, 1'b0, 4'd12, 3'b101);
        end
        apply_and_check("All activations flushed to zero");

        // ---------------- Randomized cases ----------------
        for (t = 0; t < 50; t = t + 1) begin
            if (t % 3 == 0)
                fill_random_lanes(1'b1);  // allow zero-exponent lanes
            else
                fill_random_lanes(1'b0);  // normal numbers only
            apply_and_check("Random test");
        end

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
            $display("  *** FAILURES DETECTED -- check waveform in int_mac_sim.vcd ***");
        $display("");

        $finish;
    end

endmodule
