`timescale 1ns/1ps

module tb_FP8_NATIVE_MAC_9;

    localparam N       = 9;
    localparam BUS_W   = 72;
    localparam ACC_W   = 41;
    localparam NUM_VEC = 16; // 11 edge cases + 5 random

    reg clk, rst, valid_in;
    reg  [BUS_W-1:0] fp8_act, fp8_wgt;
    wire done;
    wire signed [ACC_W-1:0] final_acc;

    integer i;
    integer errors;

    reg [BUS_W-1:0] act_vec [0:NUM_VEC-1];
    reg [BUS_W-1:0] wgt_vec [0:NUM_VEC-1];
    reg signed [ACC_W-1:0] exp_vec [0:NUM_VEC-1];
    reg [255:0] test_name [0:NUM_VEC-1]; // enough room for readable ascii names

    FP8_NATIVE_MAC_9 dut (
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .fp8_act(fp8_act),
        .fp8_wgt(fp8_wgt),
        .done(done),
        .final_acc(final_acc)
    );

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        clk = 0; rst = 1; valid_in = 0;
        fp8_act = 0; fp8_wgt = 0;
        errors = 0;

        test_name[0] = "all_zero";
        act_vec[0]  = 72'h000000000000000000;
        wgt_vec[0]  = 72'h000000000000000000;
        exp_vec[0]  = 41'sh00000000000;
        test_name[1] = "act_zero_wgt_max";
        act_vec[1]  = 72'h000000000000000000;
        wgt_vec[1]  = 72'h7e7e7e7e7e7e7e7e7e;
        exp_vec[1]  = 41'sh00000000000;
        test_name[2] = "act_max_wgt_zero";
        act_vec[2]  = 72'h7e7e7e7e7e7e7e7e7e;
        wgt_vec[2]  = 72'h000000000000000000;
        exp_vec[2]  = 41'sh00000000000;
        test_name[3] = "max_pos_times_max_pos";
        act_vec[3]  = 72'h7e7e7e7e7e7e7e7e7e;
        wgt_vec[3]  = 72'h7e7e7e7e7e7e7e7e7e;
        exp_vec[3]  = 41'sh06e40000000;
        test_name[4] = "max_neg_times_max_pos";
        act_vec[4]  = 72'hfefefefefefefefefe;
        wgt_vec[4]  = 72'h7e7e7e7e7e7e7e7e7e;
        exp_vec[4]  = 41'sh191c0000000;
        test_name[5] = "max_neg_times_max_neg";
        act_vec[5]  = 72'hfefefefefefefefefe;
        wgt_vec[5]  = 72'hfefefefefefefefefe;
        exp_vec[5]  = 41'sh06e40000000;
        test_name[6] = "min_normal_sq";
        act_vec[6]  = 72'h080808080808080808;
        wgt_vec[6]  = 72'h080808080808080808;
        exp_vec[6]  = 41'sh00000000240;
        test_name[7] = "max_subnormal_sq";
        act_vec[7]  = 72'h070707070707070707;
        wgt_vec[7]  = 72'h070707070707070707;
        exp_vec[7]  = 41'sh000000001b9;
        test_name[8] = "min_subnormal_sq";
        act_vec[8]  = 72'h010101010101010101;
        wgt_vec[8]  = 72'h010101010101010101;
        exp_vec[8]  = 41'sh00000000009;
        test_name[9] = "alternating_sign";
        act_vec[9]  = 72'h7efe7efe7efe7efe7e;
        wgt_vec[9]  = 72'h7e7e7e7e7e7e7e7e7e;
        exp_vec[9]  = 41'sh00c40000000;
        test_name[10] = "alternating_max_zero";
        act_vec[10]  = 72'h7e007e007e007e007e;
        wgt_vec[10]  = 72'h7e007e007e007e007e;
        exp_vec[10]  = 41'sh03d40000000;
        test_name[11] = "random_0";
        act_vec[11]  = 72'h8c6d7955f427f9065c;
        wgt_vec[11]  = 72'hac3252fc276468102d;
        exp_vec[11]  = 41'sh1febfa40960;
        test_name[12] = "random_1";
        act_vec[12]  = 72'hdb6eea4746b54407ed;
        wgt_vec[12]  = 72'hd36e899b99eae85708;
        exp_vec[12]  = 41'sh000c5c9ac00;
        test_name[13] = "random_2";
        act_vec[13]  = 72'h4761a188e311adadbe;
        wgt_vec[13]  = 72'h98d9e67d0521a0ae42;
        exp_vec[13]  = 41'sh1fff5d24710;
        test_name[14] = "random_3";
        act_vec[14]  = 72'h0ccb488bbc348c6064;
        wgt_vec[14]  = 72'ha6ce5c4e47eca596d7;
        exp_vec[14]  = 41'sh1fff59be5a0;
        test_name[15] = "random_4";
        act_vec[15]  = 72'h1e42218aa3877473cf;
        wgt_vec[15]  = 72'hfde31ae829f51c1dfa;
        exp_vec[15]  = 41'sh00023b1b9c0;

        @(negedge clk); @(negedge clk);
        rst = 0;

        for (i = 0; i < NUM_VEC; i = i + 1) begin
            @(negedge clk);
            fp8_act  = act_vec[i];
            fp8_wgt  = wgt_vec[i];
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            @(negedge clk);
            wait (done == 1'b1 || 1); // stage 3 registers same cycle 'done' asserts
            #0.1;

            if (final_acc !== exp_vec[i]) begin
                errors = errors + 1;
                $display("[FAIL] vec %0d (%0s): expected=%0d got=%0d",
                          i, test_name[i], exp_vec[i], final_acc);
            end else begin
                $display("[PASS] vec %0d (%0s): got=%0d", i, test_name[i], final_acc);
            end
        end

        if (errors == 0)
            $display("\n==== ALL %0d TESTS PASSED for FP8_NATIVE_MAC_9 ====", NUM_VEC);
        else
            $display("\n==== %0d / %0d TESTS FAILED for FP8_NATIVE_MAC_9 ====", errors, NUM_VEC);

        $finish;
    end

endmodule
