// ============================================================
// Optimized Integer MAC Array: M=8, N=64
// Includes clock gating, pre-multiplied alphas, and behavioral
// reduction for optimal synthesis.
// ============================================================

module int_mac_M8_N64_opt (
    input                   clk,
    input                   rst,
    input                   load,       // enables accumulation each cycle
    input                   en_reduce,  // ENABLE SIGNAL: pulses high when K/N accumulations are done

    input  signed [511:0]   a_flat,     // xd_k, k=0..63
    input  signed [511:0]   b_flat,     // wd_k, k=0..63

    input         [7:0]     alpha_x,    
    input         [7:0]     alpha_w,    
    input  signed [15:0]    beta_xw,    

    output signed [46:0]    result      
);

    // --------------------------------------------------------
    // Stage 1: MAC Unit (64 parallel mults + 24b accumulators)
    // --------------------------------------------------------
    wire signed [15:0] product [0:63];
    reg  signed [23:0] acc     [0:63];

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : gen_mac_lane
            assign product[i] = $signed(a_flat[8*i +: 8]) * $signed(b_flat[8*i +: 8]);
            
            always @(posedge clk) begin
                if (rst)
                    acc[i] <= 24'sd0;
                else if (load)
                    // Sign extend 16-bit product to 24-bit and accumulate
                    acc[i] <= acc[i] + {{8{product[i][15]}}, product[i]};
            end
        end
    endgenerate

    // --------------------------------------------------------
    // Stage 2: Reduction Tree (Behavioral for optimal synthesis)
    // --------------------------------------------------------
    reg signed [29:0] combinational_sum;
    integer j;
    
    always @(*) begin
        combinational_sum = 30'sd0;
        // Synthesis tools will automatically optimize this into 
        // a highly efficient multi-operand adder tree (Wallace/Dadda)
        for (j = 0; j < 64; j = j + 1) begin
            combinational_sum = combinational_sum + acc[j];
        end
    end

    // --------------------------------------------------------
    // Pipeline Registers with Clock Gating
    // --------------------------------------------------------
    reg  signed [29:0]  s6_reg;
    reg         [15:0]  alpha_prod_reg; // Pre-computed to save Stage 3 delay
    reg  signed [15:0]  beta_xw_reg;

    always @(posedge clk) begin
        if (rst) begin
            s6_reg         <= 30'sd0;
            alpha_prod_reg <= 16'd0;
            beta_xw_reg    <= 16'sd0;
        end else if (en_reduce) begin 
            // CLOCK GATING: Only toggle these when accumulation phase is complete.
            // This prevents the massive downstream combinational logic from switching every cycle.
            s6_reg         <= combinational_sum;
            alpha_prod_reg <= alpha_x * alpha_w; // Area win: multiply before registering
            beta_xw_reg    <= beta_xw;
        end
    end

    // --------------------------------------------------------
    // Stage 3: Scaling and Offset
    // --------------------------------------------------------
    wire signed [16:0] alpha_prod_s;
    wire signed [46:0] scaled;

    // Zero extend the unsigned product for signed multiplication
    assign alpha_prod_s = {1'b0, alpha_prod_reg};
    
    // Scale the sum
    assign scaled = s6_reg * alpha_prod_s;
    
    // Add offset (sign extending beta to match 47-bit datapath)
    assign result = scaled + {{31{beta_xw_reg[15]}}, beta_xw_reg};

endmodule