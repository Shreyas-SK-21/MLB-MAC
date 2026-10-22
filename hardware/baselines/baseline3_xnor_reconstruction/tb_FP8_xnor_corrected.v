`timescale 1ns/1ps
// ===========================================================================
// tb_FP8_xnor_corrected.v  --  Testbench for fp8_mlb_top_xnor_corrected
// ===========================================================================
//
// Tests:
// 1. All directed corner cases (zeros, signs, exponents, subnormals)
// 2. 50 random test vectors
// 3. Exact latency measurement (valid_in assertion to mac_done)
// 4. STREAMING THROUGHPUT TEST (consecutive back-to-back blocks to verify II=1)
// ===========================================================================

module tb_fp8_mlb_xnor_corrected;

    reg  clk, rst, valid_in;
    reg  [511:0] fp8_activations, fp8_weights;
    wire signed [20:0] wide_integer_sum;
    wire signed [8:0]  shared_exponent;
    wire               mac_done;

    integer pass_count = 0;
    integer fail_count = 0;
    integer test_num   = 0;

    // -----------------------------------------------------------------------
    // DUT: 1-cycle pipelined XNOR MLB-MAC
    // -----------------------------------------------------------------------
    fp8_mlb_top_xnor_corrected dut (
        .clk             (clk),
        .rst             (rst),
        .valid_in        (valid_in),
        .fp8_activations (fp8_activations),
        .fp8_weights     (fp8_weights),
        .wide_integer_sum(wide_integer_sum),
        .shared_exponent (shared_exponent),
        .mac_done        (mac_done)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    reg [7:0] act_lane [0:63];
    reg [7:0] wt_lane  [0:63];

    initial begin
        $dumpfile("mlb_mac_xnor_corrected_sim.vcd");
        $dumpvars(0, tb_fp8_mlb_xnor_corrected);
    end

    // -----------------------------------------------------------------------
    // Stimulus generation helpers
    // -----------------------------------------------------------------------
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

    task clear_all_lanes;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                act_lane[i] = 8'b0;
                wt_lane[i]  = 8'b0;
            end
        end
    endtask

    task set_lane(input integer is_weight, input integer idx,
                  input sign, input [3:0] exp_f, input [2:0] mant_f);
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
            for (i = 0; i < 64; i = i + 1) begin
                act_lane[i] = rand_fp8(allow_zero);
                wt_lane[i]  = rand_fp8(allow_zero);
            end
        end
    endtask

    task pack_vectors;
        integer i;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                fp8_activations[i*8 +: 8] = act_lane[i];
                fp8_weights[i*8 +: 8]     = wt_lane[i];
            end
        end
    endtask

    // -----------------------------------------------------------------------
    // Reference model for FP8 E4M3 dot product
    // -----------------------------------------------------------------------
    task compute_reference(output reg signed [20:0] ref_result,
                           output reg signed [8:0]  ref_exponent);
        integer i;
        reg sx, sw;
        reg [3:0] ex, ew, mx, mw, max_ex, max_ew, smx, smw;
        integer shift_x, shift_w, prod, pos_sum, neg_sum;
        begin
            max_ex = 4'd0;
            max_ew = 4'd0;
            for (i = 0; i < 64; i = i + 1) begin
                ex = act_lane[i][6:3];
                ew = wt_lane[i][6:3];
                if (ex > max_ex) max_ex = ex;
                if (ew > max_ew) max_ew = ew;
            end
            pos_sum = 0;
            neg_sum = 0;
            for (i = 0; i < 64; i = i + 1) begin
                sx = act_lane[i][7];
                ex = act_lane[i][6:3];
                mx = (ex != 4'd0) ? {1'b1, act_lane[i][2:0]} : 4'd0;
                sw = wt_lane[i][7];
                ew = wt_lane[i][6:3];
                mw = (ew != 4'd0) ? {1'b1, wt_lane[i][2:0]} : 4'd0;
                shift_x = max_ex - ex;
                shift_w = max_ew - ew;
                smx = (shift_x > 3) ? 4'd0 : (mx >> shift_x);
                smw = (shift_w > 3) ? 4'd0 : (mw >> shift_w);
                prod = smx * smw;
                if (sx ^ sw)  neg_sum = neg_sum + prod;
                else          pos_sum = pos_sum + prod;
            end
            ref_result   = pos_sum - neg_sum;
            ref_exponent = $signed({5'b0, max_ex}) + $signed({5'b0, max_ew}) - 9'sd14;
        end
    endtask

    // -----------------------------------------------------------------------
    // Single-shot test task
    // -----------------------------------------------------------------------
    task apply_and_check(input [8*40-1:0] name);
        reg signed [20:0] expected_result;
        reg signed [8:0]  expected_exponent;
        integer timeout;
        begin
            test_num = test_num + 1;
            pack_vectors;
            compute_reference(expected_result, expected_exponent);

            @(negedge clk);
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            timeout = 0;
            while (!mac_done && timeout < 2000) begin
                @(posedge clk);
                timeout = timeout + 1;
            end

            @(negedge clk);
            if (timeout >= 2000) begin
                $display("[TEST %0d] %-0s : TIMEOUT", test_num, name);
                fail_count = fail_count + 1;
            end else if ((wide_integer_sum !== expected_result) ||
                         (shared_exponent  !== expected_exponent)) begin
                $display("[TEST %0d] %-0s : FAIL", test_num, name);
                $display("           expected : sum=%0d  exp=%0d",
                          expected_result, expected_exponent);
                $display("           got      : sum=%0d  exp=%0d",
                          wide_integer_sum, shared_exponent);
                fail_count = fail_count + 1;
            end else begin
                $display("[TEST %0d] %-0s : PASS  (sum=%0d, exp=%0d)",
                          test_num, name, wide_integer_sum, shared_exponent);
                pass_count = pass_count + 1;
            end
            @(posedge clk);
        end
    endtask

    // -----------------------------------------------------------------------
    // Latency Measurement Task
    // -----------------------------------------------------------------------
    task measure_latency;
        integer cyc;
        begin
            fill_random_lanes(1'b0);
            pack_vectors;
            @(negedge clk);
            valid_in = 1'b1;
            cyc = 0;
            @(negedge clk);
            valid_in = 1'b0;
            while (!mac_done && cyc < 100) begin
                @(posedge clk);
                cyc = cyc + 1;
            end
            $display("");
            $display("=== LATENCY MEASUREMENT (XNOR) ===");
            $display("    Cycles from valid_in assert to mac_done : %0d", cyc);
            $display("    Expected : 5 clock cycles");
            if (cyc == 5)
                $display("    LATENCY VERIFIED");
            else
                $display("    WARNING: Unexpected latency");
            $display("==================================");
            $display("");
            @(posedge clk);
        end
    endtask

    // -----------------------------------------------------------------------
    // Streaming throughput test (II = 1 sustained)
    // -----------------------------------------------------------------------
    localparam N_STREAM = 8;
    reg [511:0] stream_act [0:N_STREAM-1];
    reg [511:0] stream_wt  [0:N_STREAM-1];
    reg signed [20:0] stream_ref_sum [0:N_STREAM-1];
    reg signed [8:0]  stream_ref_exp [0:N_STREAM-1];

    task streaming_test;
        integer k, match, gap_detected;
        integer send_idx, rec_idx, total_cycles;
        reg signed [20:0] got_sum [0:N_STREAM-1];
        reg signed [8:0]  got_exp [0:N_STREAM-1];
        integer           done_cycles [0:N_STREAM-1];
        begin
            $display("");
            $display("=== STREAMING THROUGHPUT TEST (XNOR II=1) ===");
            $display("    Sending %0d blocks back-to-back ...", N_STREAM);

            for (k = 0; k < N_STREAM; k = k + 1) begin
                fill_random_lanes(1'b0);
                pack_vectors;
                stream_act[k] = fp8_activations;
                stream_wt[k]  = fp8_weights;
                compute_reference(stream_ref_sum[k], stream_ref_exp[k]);
            end

            send_idx     = 0;
            rec_idx      = 0;
            total_cycles = 0;

            while (rec_idx < N_STREAM && total_cycles < 200) begin
                @(negedge clk);
                if (send_idx < N_STREAM) begin
                    fp8_activations = stream_act[send_idx];
                    fp8_weights     = stream_wt[send_idx];
                    valid_in        = 1'b1;
                    send_idx        = send_idx + 1;
                end else begin
                    valid_in        = 1'b0;
                    fp8_activations = 512'b0;
                    fp8_weights     = 512'b0;
                end

                @(posedge clk);
                total_cycles = total_cycles + 1;
                if (mac_done) begin
                    got_sum[rec_idx]     = wide_integer_sum;
                    got_exp[rec_idx]     = shared_exponent;
                    done_cycles[rec_idx] = total_cycles;
                    rec_idx              = rec_idx + 1;
                end
            end

            @(negedge clk);
            valid_in = 1'b0;

            match = 1;
            if (rec_idx < N_STREAM) begin
                $display("  ERROR: Only collected %0d of %0d outputs", rec_idx, N_STREAM);
                match = 0;
            end else begin
                for (k = 0; k < N_STREAM; k = k + 1) begin
                    if ((got_sum[k] !== stream_ref_sum[k]) ||
                        (got_exp[k] !== stream_ref_exp[k])) begin
                        $display("  Block %0d MISMATCH: expected sum=%0d exp=%0d, got sum=%0d exp=%0d",
                                  k, stream_ref_sum[k], stream_ref_exp[k],
                                  got_sum[k], got_exp[k]);
                        match = 0;
                    end
                end
            end

            gap_detected = 0;
            if (rec_idx == N_STREAM) begin
                for (k = 1; k < N_STREAM; k = k + 1) begin
                    if (done_cycles[k] != done_cycles[k-1] + 1) begin
                        $display("  GAP DETECTED between block %0d and %0d", k-1, k);
                        gap_detected = 1;
                    end
                end
            end else begin
                gap_detected = 1;
            end

            $display("    Results correct : %s", match        ? "YES" : "NO");
            $display("    Consecutive done: %s", !gap_detected ? "YES (II = 1 sustained)" : "NO");
            if (match && !gap_detected)
                $display("    *** XNOR II = 1 CONFIRMED -- pipelining verified ***");
            else
                $display("    *** STREAMING TEST FAILED ***");
            $display("=============================================");
            $display("");

            if (match && !gap_detected) pass_count = pass_count + 1;
            else                         fail_count = fail_count + 1;
        end
    endtask

    // -----------------------------------------------------------------------
    // Main stimulus
    // -----------------------------------------------------------------------
    integer t, i;
    initial begin
        rst      = 1'b1;
        valid_in = 1'b0;
        fp8_activations = 512'b0;
        fp8_weights     = 512'b0;
        clear_all_lanes;

        repeat (3) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        clear_all_lanes;
        apply_and_check("All zeros");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b0, 4'd7, 3'b000);
        end
        apply_and_check("All positive equal value");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd7, 3'b000);
            set_lane(1, i, 1'b1, 4'd7, 3'b000);
        end
        apply_and_check("All positive act, all negative weight");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, 1'b1, 4'd7, 3'b011);
            set_lane(1, i, 1'b1, 4'd7, 3'b011);
        end
        apply_and_check("Both negative (signs cancel)");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, i[0], 4'd9, 3'b010);
            set_lane(1, i, 1'b0,  4'd9, 3'b101);
        end
        apply_and_check("Alternating activation sign");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd15, 3'b111);
            set_lane(1, i, 1'b0, 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, all positive");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, i[0],  4'd15, 3'b111);
            set_lane(1, i, ~i[0], 4'd15, 3'b111);
        end
        apply_and_check("Max magnitude, alternating signs");

        for (i = 0; i < 64; i = i + 1) begin
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
        for (i = 1; i < 64; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd1, 3'b000);
            set_lane(1, i, 1'b0, 4'd1, 3'b000);
        end
        apply_and_check("Wide dynamic range (shift-to-zero stress)");

        for (i = 0; i < 64; i = i + 1) begin
            set_lane(0, i, 1'b0, 4'd0, 3'b111);
            set_lane(1, i, 1'b0, 4'd12, 3'b101);
        end
        apply_and_check("All activations flushed to zero");

        for (t = 0; t < 50; t = t + 1) begin
            if (t % 3 == 0)
                fill_random_lanes(1'b1);
            else
                fill_random_lanes(1'b0);
            apply_and_check("Random test");
        end

        measure_latency;
        streaming_test;

        $display("======================================================");
        $display("  TEST SUMMARY: %0d PASSED,  %0d FAILED",
                  pass_count, fail_count);
        $display("======================================================");
        if (fail_count == 0)
            $display("  ALL TESTS PASSED -- XNOR corrected baseline verified");
        else
            $display("  *** FAILURES DETECTED -- check waveform ***");
        $display("");
        $finish;
    end

endmodule
