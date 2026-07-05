`timescale 1ns/1ps

module tb_FP8_NATIVE_MAC_25;

    localparam N       = 25;
    localparam BUS_W   = 200;
    localparam ACC_W   = 42;
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

    FP8_NATIVE_MAC_25 dut (
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
        act_vec[0]  = 200'h00000000000000000000000000000000000000000000000000;
        wgt_vec[0]  = 200'h00000000000000000000000000000000000000000000000000;
        exp_vec[0]  = 42'sh00000000000;
        test_name[1] = "act_zero_wgt_max";
        act_vec[1]  = 200'h00000000000000000000000000000000000000000000000000;
        wgt_vec[1]  = 200'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[1]  = 42'sh00000000000;
        test_name[2] = "act_max_wgt_zero";
        act_vec[2]  = 200'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        wgt_vec[2]  = 200'h00000000000000000000000000000000000000000000000000;
        exp_vec[2]  = 42'sh00000000000;
        test_name[3] = "max_pos_times_max_pos";
        act_vec[3]  = 200'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        wgt_vec[3]  = 200'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[3]  = 42'sh13240000000;
        test_name[4] = "max_neg_times_max_pos";
        act_vec[4]  = 200'hfefefefefefefefefefefefefefefefefefefefefefefefefe;
        wgt_vec[4]  = 200'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[4]  = 42'sh2cdc0000000;
        test_name[5] = "max_neg_times_max_neg";
        act_vec[5]  = 200'hfefefefefefefefefefefefefefefefefefefefefefefefefe;
        wgt_vec[5]  = 200'hfefefefefefefefefefefefefefefefefefefefefefefefefe;
        exp_vec[5]  = 42'sh13240000000;
        test_name[6] = "min_normal_sq";
        act_vec[6]  = 200'h08080808080808080808080808080808080808080808080808;
        wgt_vec[6]  = 200'h08080808080808080808080808080808080808080808080808;
        exp_vec[6]  = 42'sh00000000640;
        test_name[7] = "max_subnormal_sq";
        act_vec[7]  = 200'h07070707070707070707070707070707070707070707070707;
        wgt_vec[7]  = 200'h07070707070707070707070707070707070707070707070707;
        exp_vec[7]  = 42'sh000000004c9;
        test_name[8] = "min_subnormal_sq";
        act_vec[8]  = 200'h01010101010101010101010101010101010101010101010101;
        wgt_vec[8]  = 200'h01010101010101010101010101010101010101010101010101;
        exp_vec[8]  = 42'sh00000000019;
        test_name[9] = "alternating_sign";
        act_vec[9]  = 200'h7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7e;
        wgt_vec[9]  = 200'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[9]  = 42'sh00c40000000;
        test_name[10] = "alternating_max_zero";
        act_vec[10]  = 200'h7e007e007e007e007e007e007e007e007e007e007e007e007e;
        wgt_vec[10]  = 200'h7e007e007e007e007e007e007e007e007e007e007e007e007e;
        exp_vec[10]  = 42'sh09f40000000;
        test_name[11] = "random_0";
        act_vec[11]  = 200'hae1f138a03393f9969ad33239645b7f5a1c47f75b069161d6b;
        wgt_vec[11]  = 200'h835a0da0737c849da4ba790001b9c912cb5a12a87ea7d1a45b;
        exp_vec[11]  = 42'sh000229bd2f2;
        test_name[12] = "random_1";
        act_vec[12]  = 200'h0fdd3339b7d1e2cac94ab762aee6f813e0d14763299b0a776e;
        wgt_vec[12]  = 200'h9250b87a0292d2373e3252344e77f77762540040d7350f8aba;
        exp_vec[12]  = 42'sh002e0447c2a;
        test_name[13] = "random_2";
        act_vec[13]  = 200'hfa1287167d2c8831315df04f56d68e070d834c4b845438aacf;
        wgt_vec[13]  = 200'h7508a4a21cacbc14a6f8d31ffa6b0b024ccc7deedcf251a4be;
        exp_vec[13]  = 42'sh3fb4518def4;
        test_name[14] = "random_3";
        act_vec[14]  = 200'hd88db44c00cdf232f48ee0a6fb0ec251cea98dc152d0f5e842;
        wgt_vec[14]  = 200'he7cbbcd9373d6ea9b6e0efd7c6cb8e75baf11617b6e7478b52;
        exp_vec[14]  = 42'sh3ff60b55e94;
        test_name[15] = "random_4";
        act_vec[15]  = 200'hb5ee9eaf285fdb2635d8d1b08b737dc6e42d3129fe79be79a2;
        wgt_vec[15]  = 200'haf202a13c44a24867c41f9ac097d0837ff6dc0639402a4377f;
        exp_vec[15]  = 42'sh00613273b5d;

        @(negedge clk); @(negedge clk);
        rst = 0;

        for (i = 0; i < NUM_VEC; i = i + 1) begin
            @(negedge clk);
            fp8_act  = act_vec[i];
            fp8_wgt  = wgt_vec[i];
            valid_in = 1'b1;
            @(negedge clk);
            valid_in = 1'b0;

            // Pipeline is 2 cycles deep (stage1 reg -> stage2 comb -> stage3 reg)
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
            $display("\n==== ALL %0d TESTS PASSED for FP8_NATIVE_MAC_25 ====", NUM_VEC);
        else
            $display("\n==== %0d / %0d TESTS FAILED for FP8_NATIVE_MAC_25 ====", errors, NUM_VEC);

        $finish;
    end

endmodule
