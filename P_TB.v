`timescale 1ns/1ps

module MLB_4_tb;

    reg  [15:0] alpha_x;
    reg  [15:0] alpha_w;
    reg  [15:0] axi;
    reg  [15:0] awi;

    wire [12:0] mlb;

    MLB_4 dut(
        .mlb(mlb),
        .alpha_x(alpha_x),
        .alpha_w(alpha_w),
        .axi(axi),
        .awi(awi)
    );

    initial begin

        $display("Time\talpha_x\talpha_w\taxi\tawi\tmlb");
        $monitor("%0t\t%h\t%h\t%h\t%h\t%d",
                  $time, alpha_x, alpha_w, axi, awi, mlb);

        //---------------------------------------
        // TEST 1 : Everything zero
        //---------------------------------------
        alpha_x = 16'h0000;
        alpha_w = 16'h0000;
        axi     = 16'h0000;
        awi     = 16'h0000;
        #10;

        //---------------------------------------
        // TEST 2 : XNOR all match
        //---------------------------------------
        alpha_x = 16'h1111;
        alpha_w = 16'h1111;
        axi     = 16'hFFFF;
        awi     = 16'hFFFF;
        #10;

        //---------------------------------------
        // TEST 3 : XNOR all mismatch
        //---------------------------------------
        alpha_x = 16'h1111;
        alpha_w = 16'h1111;
        axi     = 16'hFFFF;
        awi     = 16'h0000;
        #10;

        //---------------------------------------
        // TEST 4 : Maximum possible values
        //---------------------------------------
        alpha_x = 16'hFFFF;
        alpha_w = 16'hFFFF;
        axi     = 16'hFFFF;
        awi     = 16'hFFFF;
        #10;

        //---------------------------------------
        // TEST 5 : Mixed pattern
        //---------------------------------------
        alpha_x = 16'h1234;
        alpha_w = 16'h5678;
        axi     = 16'hAAAA;
        awi     = 16'hF0F0;
        #10;

        //---------------------------------------
        // TEST 6 : One active MLB unit
        //---------------------------------------
        alpha_x = 16'h000F;
        alpha_w = 16'h000F;
        axi     = 16'h000F;
        awi     = 16'h000F;
        #10;

        //---------------------------------------
        // TEST 7 : Alternating matches
        //---------------------------------------
        alpha_x = 16'h1111;
        alpha_w = 16'h2222;
        axi     = 16'hAAAA;
        awi     = 16'h5555;
        #10;

        $finish;

    end

endmodule