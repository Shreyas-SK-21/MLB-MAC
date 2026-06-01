`timescale 1ns / 1ps

module tb_MLB_unit_8bit;

    reg [7:0] alpha_x;
    reg [7:0] alpha_w;
    reg [7:0] axi;
    reg [7:0] awi;
    wire [19:0] out;

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

        $display("      8-BIT MLB UNIT TESTBENCH          ");

        #100;
        $display("TEST 1:");
        alpha_x = 8'h00;
        alpha_w = 8'h00;
        axi     = 8'h00;
        awi     = 8'h00;
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %10d", alpha_x, alpha_w, axi, awi, out);
        $display("TEST 2:");
        alpha_x = 8'h01;
        alpha_w = 8'h02;
        axi     = 8'hFF;
        awi     = 8'hFF;
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %10d", alpha_x, alpha_w, axi, awi, out);

        $display("TEST 3:");
        alpha_x = 8'hFF;
        alpha_w = 8'hFF;
        axi     = 8'hFF;
        awi     = 8'hFF;
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %10d", alpha_x, alpha_w, axi, awi, out);

        $display("TEST 4:");
        alpha_x = 8'h1B;
        alpha_w = 8'h34;
        axi     = 8'hAA; // 10101010
        awi     = 8'h55; // 01010101 (Popcount = 0)
        #10;
        $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %10d", alpha_x, alpha_w, axi, awi, out);

        $display("STARTING 10 RANDOM TESTS...");
        repeat(10) begin
            // $random naturally truncates down to the 8-bit width of the registers
            alpha_x = $random;
            alpha_w = $random;
            axi     = $random;
            awi     = $random;
            #10;
            $display("RANDOM TEST");
            $display("alpha_x = %h\nalpha_w = %h\naxi     = %h\nawi     = %h\nout     = %10d", alpha_x, alpha_w, axi, awi, out);
        end

        $finish;
    end

endmodule