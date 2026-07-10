`timescale 1ns/1ps

module tb_FP8_NATIVE_MAC_64;

    localparam N       = 64;
    localparam BUS_W   = 512;
    localparam ACC_W   = 43;
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

    FP8_NATIVE_MAC_64 dut (
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
        act_vec[0]  = 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        wgt_vec[0]  = 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        exp_vec[0]  = 43'sh00000000000;
        test_name[1] = "act_zero_wgt_max";
        act_vec[1]  = 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        wgt_vec[1]  = 512'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[1]  = 43'sh00000000000;
        test_name[2] = "act_max_wgt_zero";
        act_vec[2]  = 512'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        wgt_vec[2]  = 512'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
        exp_vec[2]  = 43'sh00000000000;
        test_name[3] = "max_pos_times_max_pos";
        act_vec[3]  = 512'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        wgt_vec[3]  = 512'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[3]  = 43'sh31000000000;
        test_name[4] = "max_neg_times_max_pos";
        act_vec[4]  = 512'hfefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefe;
        wgt_vec[4]  = 512'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[4]  = 43'sh4f000000000;
        test_name[5] = "max_neg_times_max_neg";
        act_vec[5]  = 512'hfefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefe;
        wgt_vec[5]  = 512'hfefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefe;
        exp_vec[5]  = 43'sh31000000000;
        test_name[6] = "min_normal_sq";
        act_vec[6]  = 512'h08080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808;
        wgt_vec[6]  = 512'h08080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808;
        exp_vec[6]  = 43'sh00000001000;
        test_name[7] = "max_subnormal_sq";
        act_vec[7]  = 512'h07070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707;
        wgt_vec[7]  = 512'h07070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707;
        exp_vec[7]  = 43'sh00000000c40;
        test_name[8] = "min_subnormal_sq";
        act_vec[8]  = 512'h01010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101;
        wgt_vec[8]  = 512'h01010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101;
        exp_vec[8]  = 43'sh00000000040;
        test_name[9] = "alternating_sign";
        act_vec[9]  = 512'hfe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7efe7e;
        wgt_vec[9]  = 512'h7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e;
        exp_vec[9]  = 43'sh00000000000;
        test_name[10] = "alternating_max_zero";
        act_vec[10]  = 512'h007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e;
        wgt_vec[10]  = 512'h007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e007e;
        exp_vec[10]  = 43'sh18800000000;
        test_name[11] = "random_0";
        act_vec[11]  = 512'h6d2ed14cbc36798bd82c06db1119332bf441e9d9d698f193f856166424e344a540e24b3a5f2ad8bfdb125cedae92496a15c754a3ebbe912145442ff66bbb1c7f;
        wgt_vec[11]  = 512'h816dc6ff438d2a3a0837297025fe8f34b4f1a2168054d4848580604d920192022e26245c889a1194d723ed656a3be19d8860a9d27982111efef8b94c86309c9e;
        exp_vec[11]  = 43'sh7fd869e9c64;
        test_name[12] = "random_1";
        act_vec[12]  = 512'h8385f88c96154144646984f5521514e22183de6965636fe23525aa410f2067a906bdb115ff1c0c60d8f38bbbc8e33521042b8d4a54b14522528d9ea000e08c3f;
        wgt_vec[12]  = 512'hb22a596aa9182c7d784d952af7babe1811da41da75df77816eec71579f1f588347ce163f1db72883f1cb964113b641d016e7335b47411f228ae161d12482a545;
        exp_vec[12]  = 43'sh002f05357e8;
        test_name[13] = "random_2";
        act_vec[13]  = 512'h4bf83c276b00882ec81dcb5af0c4a44a346c52ba9c9b43440e538e918a1188152880dbb4da143c2851e0a1f04c3e19e252f4539702cced0f096d48c5ec8bad14;
        wgt_vec[13]  = 512'hf868a5449cac23c44e69b629badcb56f474ca50a763d7e608b8d18c6bf86ba3cb2eea31e2c1d317c7238c4bd91bc9c1854dbf8db63956ab527e34169a179ab5b;
        exp_vec[13]  = 43'sh7fe79c61af2;
        test_name[14] = "random_3";
        act_vec[14]  = 512'h71180c67deca4c179baa8e34f84a35d1837ed818881d2c61a130906868df9d2313d0de1c9fde6bb396d8c645199225e81c629484c79fa3841f1f5da6263c29f3;
        wgt_vec[14]  = 512'hced3382e27075b5d744ed25227bfd9a6df271835c28414acaaa8476c093fa38ad018f36fc77f1a3a3db1ef4b39717e681eeef81890a15ecf6943711c8925475c;
        exp_vec[14]  = 43'sh7ff48165e90;
        test_name[15] = "random_4";
        act_vec[15]  = 512'hf4f30e008dc3874c27a3a9beb119e53cbbdb0179d7fc3b4780b341476d09df088eac7ccd984ea34dae777a460065eea76ea77f06d2f049afeb41102cd590f791;
        wgt_vec[15]  = 512'h09aa388c742261d4de28337c40834f51f86176d36fcd231305b400a9a87f806712bd77f95f16c77f207183310558ddf1a17efd6077828014fddd0811aea5302a;
        exp_vec[15]  = 43'sh7fde847ce3c;

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
            $display("\n==== ALL %0d TESTS PASSED for FP8_NATIVE_MAC_64 ====", NUM_VEC);
        else
            $display("\n==== %0d / %0d TESTS FAILED for FP8_NATIVE_MAC_64 ====", errors, NUM_VEC);

        $finish;
    end

endmodule
