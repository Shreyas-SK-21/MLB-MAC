`timescale 1ns / 1ps

module tb_MLB_unit;
    reg [15:0] alpha_x;
    reg [15:0] alpha_w;
    reg [15:0] axi;
    reg [15:0] awi;
    wire [36:0] out;

    MLB_unit uut (
        .out(out),
        .alpha_x(alpha_x),
        .alpha_w(alpha_w),
        .axi(axi),
        .awi(awi)
    );

    initial begin
        alpha_x = 0;
        alpha_w = 0;
        axi = 0;
        awi = 0;

        $display("      16-BIT MLB UNIT TESTBENCH         ");

        // Wait 100 ns for global reset to finish
        #100;

        // TEST 1: All Zeros
        $display("TEST 1:");
        alpha_x = 16'h0000;
        alpha_w = 16'h0000;
        axi     = 16'h0000;
        awi     = 16'h0000;
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %15d", alpha_x, alpha_w, axi, awi, out);

        $display("TEST 2:");
        alpha_x = 16'h0001;
        alpha_w = 16'h0002;
        axi     = 16'hFFFF;
        awi     = 16'hFFFF;
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %15d", alpha_x, alpha_w, axi, awi, out);

        $display("TEST 3:");
        alpha_x = 16'hFFFF;
        alpha_w = 16'hFFFF;
        axi     = 16'hFFFF;
        awi     = 16'hFFFF;
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %15d", alpha_x, alpha_w, axi, awi, out);

        $display("TEST 4:");
        alpha_x = 16'h0A1B;
        alpha_w = 16'h1234;
        axi     = 16'hAAAA; // 10101010...
        awi     = 16'h5555; // 01010101... (Popcount = 0)
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %15d", alpha_x, alpha_w, axi, awi, out);

        $display("STARTING 10 RANDOM TESTS...");
        repeat(10) begin
            alpha_x = $random;
            alpha_w = $random;
            axi     = $random;
            awi     = $random;
            #10;
            $display("RANDOM TEST");
            $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %15d", alpha_x, alpha_w, axi, awi, out);
        end

        $finish;
    end

endmodule