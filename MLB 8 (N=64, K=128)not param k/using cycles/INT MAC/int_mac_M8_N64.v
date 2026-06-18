// ============================================================
// Integer MAC Array: M=8, N=64, K=128
// Architecture follows Fig. 4 of MLB-MAC paper:
//   Stage 1 - MAC Unit: N=64 parallel M-bit multipliers + 2-cycle accumulators
//   Stage 2 - Reduce Module: 6-level binary tree adder over 64 accumulators
//            + pipeline register (matches MLB registered depth)
//   Stage 3 - Scaling & Offset: multiply by alpha_x*alpha_w, add beta_xw
//
// Updates for K=128:
//   - Requires temporal multiplexing over K/N = 2 cycles.
//   - valid_in must be held high for 2 cycles with new a_flat/b_flat chunks.
//   - An internal state machine auto-clears the accumulators on cycle 0 
//     and outputs a `done` signal when the pipeline completes.
// ============================================================

module int_mac_M8_N64_K128 (
    input                   clk,
    input                   rst,
    input                   valid_in,   // Held high for 2 cycles for K=128

    // N=64 lanes × M=8 bits = 512 bits each
    input  signed [511:0]   a_flat,     // xd_k
    input  signed [511:0]   b_flat,     // wd_k

    // Scaling and offset parameters
    input         [7:0]     alpha_x,    // activation scaling factor (unsigned)
    input         [7:0]     alpha_w,    // weight scaling factor     (unsigned)
    input  signed [15:0]    beta_xw,    // pre-computed offset β_xw  (signed)

    output signed [46:0]    result,     // dot(x^q, w^q) = alpha_x*alpha_w*sum + beta_xw
    output                  done        // Pulses high when `result` is valid
);

    // --------------------------------------------------------
    // Control Logic for 2-Cycle Accumulation (K=128)
    // --------------------------------------------------------
    reg cycle_cnt;
    reg stage1_done; // Asserts when 2-cycle accumulation finishes
    reg done_reg;    // Asserts when pipeline stage 2 finishes

    always @(posedge clk) begin
        if (rst) begin
            cycle_cnt   <= 1'b0;
            stage1_done <= 1'b0;
            done_reg    <= 1'b0;
        end else begin
            if (valid_in) begin
                cycle_cnt <= ~cycle_cnt;
                // Once we hit the second cycle, the accumulator will have the final sum
                if (cycle_cnt == 1'b1) begin
                    stage1_done <= 1'b1;
                end else begin
                    stage1_done <= 1'b0;
                end
            end else begin
                stage1_done <= 1'b0;
            end
            
            // Shift into the final pipeline validation register
            done_reg <= stage1_done;
        end
    end

    assign done = done_reg;

    // --------------------------------------------------------
    // Stage 1: MAC Unit
    //   N=64 parallel 8-bit signed multipliers + 24-bit accumulators
    // --------------------------------------------------------
    wire signed [15:0] product [0:63];
    reg  signed [23:0] acc     [0:63];

    genvar i;
    generate
        for (i = 0; i < 64; i = i + 1) begin : gen_mac_lane
            assign product[i] = $signed(a_flat[8*i +: 8])
                               * $signed(b_flat[8*i +: 8]);
                               
            always @(posedge clk) begin
                if (rst) begin
                    acc[i] <= 24'sd0;
                end else if (valid_in) begin
                    if (cycle_cnt == 1'b0) begin
                        // Cycle 0: Overwrite old accumulator data
                        acc[i] <= {{8{product[i][15]}}, product[i]};
                    end else begin
                        // Cycle 1: Accumulate the second K=64 chunk
                        acc[i] <= acc[i] + {{8{product[i][15]}}, product[i]};
                    end
                end
            end
        end
    endgenerate

    // --------------------------------------------------------
    // Stage 2: Reduce Module (6-level binary tree, combinational)
    // --------------------------------------------------------
    wire signed [24:0] s1 [0:31];
    wire signed [25:0] s2 [0:15];
    wire signed [26:0] s3 [0:7];
    wire signed [27:0] s4 [0:3];
    wire signed [28:0] s5 [0:1];
    wire signed [29:0] s6;

    genvar j;
    generate
        for (j = 0; j < 32; j = j + 1) begin : gen_r1
            assign s1[j] = $signed(acc[2*j]) + $signed(acc[2*j+1]);
        end
        for (j = 0; j < 16; j = j + 1) begin : gen_r2
            assign s2[j] = $signed(s1[2*j]) + $signed(s1[2*j+1]);
        end
        for (j = 0; j < 8; j = j + 1) begin : gen_r3
            assign s3[j] = $signed(s2[2*j]) + $signed(s2[2*j+1]);
        end
        for (j = 0; j < 4; j = j + 1) begin : gen_r4
            assign s4[j] = $signed(s3[2*j]) + $signed(s3[2*j+1]);
        end
        for (j = 0; j < 2; j = j + 1) begin : gen_r5
            assign s5[j] = $signed(s4[2*j]) + $signed(s4[2*j+1]);
        end
    endgenerate

    assign s6 = $signed(s5[0]) + $signed(s5[1]);

    // --------------------------------------------------------
    // Pipeline Register (between Stage 2 and Stage 3)
    //   Breaks the combinational path to match MLB registered depth.
    // --------------------------------------------------------
    reg  signed [29:0]  s6_reg;
    reg         [7:0]   alpha_x_reg;
    reg         [7:0]   alpha_w_reg;
    reg  signed [15:0]  beta_xw_reg;

    always @(posedge clk) begin
        if (rst) begin
            s6_reg      <= 30'sd0;
            alpha_x_reg <= 8'd0;
            alpha_w_reg <= 8'd0;
            beta_xw_reg <= 16'sd0;
        end else if (stage1_done) begin
            // Only capture the final reduced sum when the 2-cycle accumulation is fully done
            s6_reg      <= s6;
            alpha_x_reg <= alpha_x;
            alpha_w_reg <= alpha_w;
            beta_xw_reg <= beta_xw;
        end
    end

    // --------------------------------------------------------
    // Stage 3: Scaling and Offset (Combinational Output)
    // --------------------------------------------------------
    wire [15:0]        alpha_prod;
    wire signed [16:0] alpha_prod_s;
    wire signed [46:0] scaled;

    assign alpha_prod   = alpha_x_reg * alpha_w_reg;
    assign alpha_prod_s = {1'b0, alpha_prod};

    assign scaled = $signed(s6_reg) * $signed(alpha_prod_s);
    assign result = scaled + {{31{beta_xw_reg[15]}}, beta_xw_reg};

endmodule