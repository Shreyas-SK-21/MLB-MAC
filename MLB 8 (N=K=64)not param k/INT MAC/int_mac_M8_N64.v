// ============================================================
// Integer MAC Array: M=8, N=64
// Architecture follows Fig. 4 of MLB-MAC paper:
//   Stage 1 - MAC Unit:      N=64 parallel M-bit multipliers + accumulators
//   Stage 2 - Reduce Module: 6-level binary tree adder over 64 accumulators
//            + pipeline register (matches MLB registered depth)
//   Stage 3 - Scaling & Offset: multiply by alpha_x*alpha_w, add beta_xw
//
// Pipeline register added between Stage 2 and Stage 3 to:
//   - Break the long combinational cone (64 multipliers → tree → alpha scale)
//   - Match MLB's registered pipeline depth (acc → basis_mult register)
//   - Make power comparison fair and timing realistic at 250MHz/45nm
//
// Parameter derivation (M=8, N=64):
//   product[i]  : signed 8b × signed 8b  → 16-bit signed
//   acc[i]      : 24-bit signed (up to K/N=256 accumulations)
//   s6          : 30-bit signed (6-level tree, 64→1)
//   s6_reg      : 30-bit registered (pipeline stage)
//   alpha_prod  : 8×8 unsigned → 16-bit unsigned
//   alpha_prod_s: 17-bit signed (zero extended)
//   scaled      : 30b × 17b → 46-bit signed
//   result      : 47-bit signed (46 + sign guard for beta addition)
// ============================================================

module int_mac_M8_N64 (
    input                   clk,
    input                   rst,
    input                   load,       // enable accumulation each cycle

    // N=64 lanes × M=8 bits = 512 bits each
    input  signed [511:0]   a_flat,     // xd_k, k=0..63
    input  signed [511:0]   b_flat,     // wd_k, k=0..63

    // Scaling and offset parameters (Eq. 3 of paper)
    input         [7:0]     alpha_x,    // activation scaling factor (unsigned)
    input         [7:0]     alpha_w,    // weight scaling factor     (unsigned)
    input  signed [15:0]    beta_xw,    // pre-computed offset β_xw  (signed)

    output signed [46:0]    result      // dot(x^q, w^q) = alpha_x*alpha_w*sum + beta_xw
);

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
                if (rst)
                    acc[i] <= 24'sd0;
                else if (load)
                    acc[i] <= acc[i] + {{8{product[i][15]}}, product[i]};
            end
        end
    endgenerate

    // --------------------------------------------------------
    // Stage 2: Reduce Module (6-level binary tree, combinational)
    //   Level 1: 64→32, width 25
    //   Level 2: 32→16, width 26
    //   Level 3: 16→8,  width 27
    //   Level 4: 8→4,   width 28
    //   Level 5: 4→2,   width 29
    //   Level 6: 2→1,   width 30
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
    //   Breaks the combinational path:
    //     acc[] → tree → alpha_scale → result
    //   into two registered stages matching MLB depth:
    //     acc[] → tree → [REG] → alpha_scale → result
    //   Also registers alpha and beta to keep them aligned
    //   with the pipelined sum.
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
        end else begin
            s6_reg      <= s6;
            alpha_x_reg <= alpha_x;
            alpha_w_reg <= alpha_w;
            beta_xw_reg <= beta_xw;
        end
    end

    // --------------------------------------------------------
    // Stage 3: Scaling and Offset
    //   Uses registered s6_reg and registered alpha/beta
    //   dot(x^q, w^q) = alpha_x * alpha_w * s6_reg + beta_xw
    // --------------------------------------------------------
    wire [15:0]        alpha_prod;
    wire signed [16:0] alpha_prod_s;
    wire signed [46:0] scaled;

    assign alpha_prod   = alpha_x_reg * alpha_w_reg;
    assign alpha_prod_s = {1'b0, alpha_prod};

    assign scaled = $signed(s6_reg) * $signed(alpha_prod_s);
    assign result = scaled + {{31{beta_xw_reg[15]}}, beta_xw_reg};

endmodule
