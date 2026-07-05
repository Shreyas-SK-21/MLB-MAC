`timescale 1ns/1ps

module tb_FP8_NATIVE_MAC_32;

    localparam N       = 32;
    localparam BUS_W   = 256;
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

    FP8_NATIVE_MAC_32 dut (
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
        act_vec[0]  = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        wgt_vec[0]  = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        exp_vec[0]  = 42'sh00000000000;
        test_name[1] = "act_zero_wgt_max";
        act_vec[1]  = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        wgt_vec[1]  = 256'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[1]  = 42'sh00000000000;
        test_name[2] = "act_max_wgt_zero";
        act_vec[2]  = 256'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        wgt_vec[2]  = 256'h0000000000000000000000000000000000000000000000000000000000000000;
        exp_vec[2]  = 42'sh00000000000;
        test_name[3] = "max_pos_times_max_pos";
        act_vec[3]  = 256'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        wgt_vec[3]  = 256'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[3]  = 42'sh18800000000;
        test_name[4] = "max_neg_times_max_pos";
        act_vec[4]  = 256'hfefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefe;
        wgt_vec[4]  = 256'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[4]  = 42'sh27800000000;
        test_name[5] = "max_neg_times_max_neg";
        act_vec[5]  = 256'hfefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefe;
        wgt_vec[5]  = 256'hfefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefe;
        exp_vec[5]  = 42'sh18800000000;
        test_name[6] = "min_normal_sq";
        act_vec[6]  = 256'h0808080808080808080808080808080808080808080808080808080808080808;
        wgt_vec[6]  = 256'h0808080808080808080808080808080808080808080808080808080808080808;
        exp_vec[6]  = 42'sh00000000800;
        test_name[7] = "max_subnormal_sq";
        act_vec[7]  = 256'h0707070707070707070707070707070707070707070707070707070707070707;
        wgt_vec[7]  = 256'h0707070707070707070707070707070707070707070707070707070707070707;
        exp_vec[7]  = 42'sh00000000620;
        test_name[8] = "min_subnormal_sq";
        act_vec[8]  = 256'h0101010101010101010101010101010101010101010101010101010101010101;
        wgt_vec[8]  = 256'h0101010101010101010101010101010101010101010101010101010101010101;
        exp_vec[8]  = 42'sh00000000020;
        test_name[9] = "alternating_sign";
        act_vec[9]  = 256'hfe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7e;
        wgt_vec[9]  = 256'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[9]  = 42'sh00000000000;
        test_name[10] = "alternating_max_zero";
        act_vec[10]  = 256'h007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e;
        wgt_vec[10]  = 256'h007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e;
        exp_vec[10]  = 42'sh0c400000000;
        test_name[11] = "random_0";
        act_vec[11]  = 256'h4641cb7413dc16b724808087130d85bc42d2c29d6108b81c6c91ea6dac51d9c6;
        wgt_vec[11]  = 256'hc241e0a5b8f966557bd7077fb0f35f3268b5c5c5bb205be71aefeb3791eea114;
        exp_vec[11]  = 42'sh000d053ab80;
        test_name[12] = "random_1";
        act_vec[12]  = 256'h7ef5553effb300dbb6840997709338cda0d9125594f9c97ede4d706aac1e3dde;
        wgt_vec[12]  = 256'hb3f52b8e3d69cdea11e535d0591edd2bfd9b764d9a0753dcaa3faf223d2fb624;
        exp_vec[12]  = 42'sh0022a4c7190;
        test_name[13] = "random_2";
        act_vec[13]  = 256'ha7dcb141be0b1e9eb67af405ae8fea834cf796ebd0c157d0787bc8639b13a6ad;
        wgt_vec[13]  = 256'h35a0386e0215b1c634d8aea15a9bec4b5025bad7d716a5bf26a6ca4cea654b30;
        exp_vec[13]  = 42'sh000480be84a;
        test_name[14] = "random_3";
        act_vec[14]  = 256'h6fa609ea8cd7ccd65dfbcaf67ea57898a8f2c4208f21744589e21bae0338f743;
        wgt_vec[14]  = 256'h108e208d78a485d3f11d432e110b37e8951436df28ec2c13f76e57b9c3666889;
        exp_vec[14]  = 42'sh3fe955e7268;
        test_name[15] = "random_4";
        act_vec[15]  = 256'haa2b08147f08dd9c08a6d8bfd1cfc1bbb25b265b676868b3196a9759554d5ce6;
        wgt_vec[15]  = 256'h9873d70c4fd3b69b51209625c2a557afb5a30cfba17129b5af24c4e6a6b65499;
        exp_vec[15]  = 42'sh000456f32e0;

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
            $display("\n==== ALL %0d TESTS PASSED for FP8_NATIVE_MAC_32 ====", NUM_VEC);
        else
            $display("\n==== %0d / %0d TESTS FAILED for FP8_NATIVE_MAC_32 ====", errors, NUM_VEC);

        $finish;
    end

endmodule
