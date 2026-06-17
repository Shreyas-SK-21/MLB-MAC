// The final result should be $(768 \times 8) + 15 = 6159$.

`timescale 1ns/1ps

module tb_int_mac_M8_N64_opt;

    // --------------------------------------------------------
    // Signals
    // --------------------------------------------------------
    reg                   clk;
    reg                   rst;
    reg                   load;
    reg                   en_reduce;
    
    reg  signed [511:0]   a_flat;
    reg  signed [511:0]   b_flat;
    
    reg         [7:0]     alpha_x;
    reg         [7:0]     alpha_w;
    reg  signed [15:0]    beta_xw;
    
    wire signed [46:0]    result;

    // --------------------------------------------------------
    // Device Under Test (DUT)
    // --------------------------------------------------------
    int_mac_M8_N64_opt dut (
        .clk(clk),
        .rst(rst),
        .load(load),
        .en_reduce(en_reduce),
        .a_flat(a_flat),
        .b_flat(b_flat),
        .alpha_x(alpha_x),
        .alpha_w(alpha_w),
        .beta_xw(beta_xw),
        .result(result)
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
        rst       = 1;
        load      = 0;
        en_reduce = 0;
        
        // Populate 64 lanes with a=2 and b=3
        a_flat  = {64{8'sd2}}; 
        b_flat  = {64{8'sd3}}; 
        
        alpha_x = 8'd2;
        alpha_w = 8'd4;
        beta_xw = 16'sd15;

        // 2. Release Reset
        #10;
        rst = 0;

        // 3. Accumulate for 2 cycles (K/N = 2)
        @(posedge clk);
        load = 1;
        @(posedge clk);
        @(posedge clk);
        load = 0; // Stop loading

        // 4. Trigger Reduction and Scaling
        en_reduce = 1;
        @(posedge clk);
        en_reduce = 0;

        // 5. Wait for pipeline propagation
        @(posedge clk);
        @(posedge clk);

        // 6. Check Result
        if (result === 47'sd6159) begin
            $display("========================================");
            $display("[PASS] Integer MAC computed successfully!");
            $display("Expected: 6159 | Got: %0d", result);
            $display("========================================");
        end else begin
            $display("========================================");
            $display("[FAIL] Integer MAC computation failed!");
            $display("Expected: 6159 | Got: %0d", result);
            $display("========================================");
        end

        #20 $finish;
    end

endmodule