`timescale 1ns/1ps

module tb_MLB_4;

    reg  [15:0] alpha_x;
    reg  [15:0] alpha_w;
    reg  [15:0] axi;
    reg  [15:0] awi;

    wire [11:0] mlb;

    MLB_4 dut(
        .mlb(mlb),
        .alpha_x(alpha_x),
        .alpha_w(alpha_w),
        .axi(axi),
        .awi(awi)
    );

    initial begin
        $display("      4-BIT MLB MAC TESTBENCH");
        alpha_x = 16'h1111;
        alpha_w = 16'h2222;
        axi     = 16'hFFFF;
        awi     = 16'hFFFF;
        #10;
        $display("TEST 1");
        $display("alpha_x = %h", alpha_x);
        $display("alpha_w = %h", alpha_w);
        $display("axi     = %h", axi);
        $display("awi     = %h", awi);
        $display("mlb     = %d", mlb);

        alpha_x = 16'h3333;
        alpha_w = 16'h2222;
        axi     = 16'hFFFF;
        awi     = 16'h0000;

        #10;
        $display("TEST 2");
        $display("mlb = %d", mlb);
        alpha_x = 16'h1234;
        alpha_w = 16'h4321;
        axi     = 16'hA5A5;
        awi     = 16'h5A5A;

        #10;
        $display("TEST 3");
        $display("alpha_x = %h", alpha_x);
        $display("alpha_w = %h", alpha_w);
        $display("axi     = %h", axi);
        $display("awi     = %h", awi);
        $display("mlb     = %d", mlb);
        alpha_x = 16'hFFFF;
        alpha_w = 16'hFFFF;
        axi     = 16'hFFFF;
        awi     = 16'hFFFF;

        #10;
        $display("TEST 4");
        $display("mlb = %d", mlb);

        repeat(10) begin
            alpha_x = $random;
            alpha_w = $random;
            axi     = $random;
            awi     = $random;
            #10;
            $display("RANDOM TEST");
            $display("alpha_x = %h", alpha_x);
            $display("alpha_w = %h", alpha_w);
            $display("axi     = %h", axi);
            $display("awi     = %h", awi);
            $display("mlb     = %d", mlb);
        end

        $finish;
    end
endmodule