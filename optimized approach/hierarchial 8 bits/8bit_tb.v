// final output should be $768 \times 64 = 49152$.

`timescale 1ns/1ps

module tb_MLB_8;

    // --------------------------------------------------------
    // Signals
    // --------------------------------------------------------
    reg                  clk;
    reg                  rst;
    reg                  valid_in;
    
    reg  [63:0]          alpha_x;
    reg  [63:0]          alpha_w;
    reg  [511:0]         axi;
    reg  [511:0]         awi;
    reg  [12:0]          K;
    
    wire signed [37:0]   mlb;
    wire                 done;

    // --------------------------------------------------------
    // Device Under Test (DUT)
    // --------------------------------------------------------
    MLB_8 dut (
        .mlb(mlb),
        .done(done),
        .alpha_x(alpha_x),
        .alpha_w(alpha_w),
        .axi(axi),
        .awi(awi),
        .K(K),
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in)
    );

    // --------------------------------------------------------
    // Clock Generation (250MHz -> 4ns period)
    // --------------------------------------------------------
    initial clk = 0;
    always #2 clk = ~clk;

    // --------------------------------------------------------
    // Stimulus and Self-Check
    // --------------------------------------------------------
    initial begin
        // 1. Initialize 
        rst      = 1;
        valid_in = 0;
        
        // K = 128 means K>>6 (cycles_needed) = 2 cycles of accumulation
        K        = 13'd128; 
        
        // Perfect matching bits for maximum popcount (Popcount = 64 per unit)
        axi      = {512{1'b1}}; 
        awi      = {512{1'b1}};
        
        // 8 chunks of 8-bit alphas
        alpha_x  = {8{8'd2}}; 
        alpha_w  = {8{8'd3}}; 

        // 2. Release Reset
        #10;
        rst = 0;
        
        // 3. Start Accumulation
        @(posedge clk);
        valid_in = 1;
        
        // Wait for 2 accumulation cycles (K=128 / N=64)
        @(posedge clk);
        @(posedge clk);
        valid_in = 0;

        // 4. Wait for propagation through the pipelined hierarchy
        wait (done == 1'b1);
        @(posedge clk); // Give one cycle for the top level sum to register

        // 5. Check Result
        if (mlb === 38'sd49152) begin
            $display("========================================");
            $display("[PASS] MLB_8 Hierarchical MAC computed successfully!");
            $display("Expected: 49152 | Got: %0d", mlb);
            $display("========================================");
        end else begin
            $display("========================================");
            $display("[FAIL] MLB_8 Hierarchical MAC computation failed!");
            $display("Expected: 49152 | Got: %0d", mlb);
            $display("========================================");
        end

        #20 $finish;
    end

endmodule