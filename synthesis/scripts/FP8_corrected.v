// ===========================================================================
// FP8_corrected.v  --  fp8_mlb_top_corrected  (AND-Popcount, N=64, M=4)
// ===========================================================================
//
// CHANGE FROM ORIGINAL FP8.v
// ---------------------------
// Original fp8_mlb_top used a 3-state FSM to TIME-MULTIPLEX one MLB_4 array
// between positive-sign and negative-sign dot-product lanes across TWO clock
// cycles:
//   Cycle 1: mac_shared processes positive lanes  (pos_mask applied)
//   Cycle 2: mac_shared processes negative lanes  (neg_mask applied)
//   result  = pos_reg - shared_sum_out            (requires storing Cycle-1
//                                                  result in pos_reg)
//
// This halved the true hardware throughput to II = 2 cycles per block,
// making the IPS/W calculation in the paper incorrect (Eq. 6 assumed II=1).
//
// THIS FILE CORRECTS THAT:
//   - Removes the FSM entirely (state, is_neg_phase, internal_valid, got_pos,
//     pos_reg are all deleted).
//   - Instantiates TWO MLB_4 arrays: mac_pos and mac_neg.
//   - mac_pos always receives the positive-masked activation planes.
//   - mac_neg always receives the negative-masked activation planes.
//   - Both receive valid_in_d1 in the SAME clock cycle.
//   - Both complete their 3-stage pipeline in the SAME clock cycle.
//   - result = pos_sum - neg_sum is registered in one step.
//   => II = 1 clock cycle  (back-to-back blocks accepted every cycle)
//   => The paper's Eq. 6 cycle count is now honest.
//
// AREA COST OF THE FIX
// ---------------------
// One extra MLB_4 instance is added. At N=64 the MLB array occupies ~16 % of
// total chip area, so the penalty is roughly +16 % on cell count.  The rest
// of the hierarchy (BFP aligners, sign logic, output register) is unchanged.
//
// PIPELINE DEPTH (latency from external valid_in to mac_done)
// -----------------------------------------------------------
//   Cycle +0  valid_in asserted externally
//   Cycle +1  bfp_aligner registers aligned_planes; sign_k_d1 and valid_in_d1
//             are registered.  axi_planes_pos/neg computed combinationally.
//             Both MLB_4 cores begin their 3-stage compute:
//     Cycle +2  and_popcount_4_bit registers f_output (Stage 1)
//     Cycle +3  MLB_unit registers shifted_out          (Stage 2)
//     Cycle +4  MLB_4 registers mlb; pos_done/neg_done fire (Stage 3)
//   Cycle +5  result_reg = pos_sum - neg_sum; mac_done fires
//
// Total pipeline latency: 5 clock cycles.
// Initiation Interval  : 1 clock cycle (fully pipelined).
// ===========================================================================

`timescale 1ns/1ps

module fp8_mlb_top_corrected (
    output signed [20:0] wide_integer_sum, // Final signed integer dot product
    output signed [8:0]  shared_exponent,  // Block exponent: e_max,x+e_max,w-20
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,         // Pulse high for >= 1 cycle per block
    input [511:0]        fp8_activations,  // 64 x FP8 E4M3 activation values
    input [511:0]        fp8_weights       // 64 x FP8 E4M3 weight values
);

    // -----------------------------------------------------------------------
    // Stage 0 : BFP Alignment  (1-cycle registered)
    //
    // Two bfp_aligner instances convert FP8 vectors into 4x64-bit mantissa
    // bit-planes and extract the block-max exponent.  Both are purely
    // combinational internally; they register their outputs at posedge clk,
    // introducing exactly 1 cycle of latency.
    // -----------------------------------------------------------------------
    wire [255:0] axi_planes;   // 4 x 64-bit activation mantissa bit-planes
    wire [255:0] awi_planes;   // 4 x 64-bit weight    mantissa bit-planes
    wire [3:0]   max_exp_x;    // Block-max exponent of activations
    wire [3:0]   max_exp_w;    // Block-max exponent of weights

    bfp_aligner align_x (
        .clk          (clk),
        .fp8_vec      (fp8_activations),
        .aligned_planes(axi_planes),
        .max_exp      (max_exp_x)
    );

    bfp_aligner align_w (
        .clk          (clk),
        .fp8_vec      (fp8_weights),
        .aligned_planes(awi_planes),
        .max_exp      (max_exp_w)
    );

    // -----------------------------------------------------------------------
    // Stage 0 (parallel) : Sign XOR, registered to align with aligned_planes
    //
    // sign_k[i] = sign_x[i] XOR sign_w[i]
    //   0  -> lane i contributes positively  (same signs, product >= 0)
    //   1  -> lane i contributes negatively  (different signs, product < 0)
    //
    // sign_k is purely combinational; sign_k_d1 is registered one cycle so
    // it arrives in the same pipeline stage as aligned_planes from bfp_aligner.
    // -----------------------------------------------------------------------
    genvar i;
    wire [63:0] sign_k;
    generate
        for (i = 0; i < 64; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [63:0] sign_k_d1;
    reg        valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= 64'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 1 : Sign Masking  (combinational, uses registered values)
    //
    // Partition the 4 activation bit-planes into two disjoint sets:
    //   axi_planes_pos : lanes where the product is positive (pos_mask[i]=1)
    //   axi_planes_neg : lanes where the product is negative (neg_mask[i]=1)
    //
    // Weight planes are NOT masked -- they are shared by both MLB_4 cores.
    // This is correct because the weight magnitudes are the same regardless
    // of the sign of the product; only activation lanes are routed differently.
    //
    // Both axi_planes and sign_k_d1 were registered at the PREVIOUS posedge,
    // so this combinational logic is glitch-free between clock edges.
    // -----------------------------------------------------------------------
    wire [63:0] pos_mask = ~sign_k_d1;  // 1 where sign_k=0 (positive product)
    wire [63:0] neg_mask =  sign_k_d1;  // 1 where sign_k=1 (negative product)

    wire [255:0] axi_planes_pos;  // Positive lanes only
    wire [255:0] axi_planes_neg;  // Negative lanes only

    // Apply mask to each of the 4 activation bit-planes simultaneously.
    // (Weight planes awi_planes are shared between mac_pos and mac_neg.)
    assign axi_planes_pos[  63:  0] = axi_planes[  63:  0] & pos_mask;  // Plane 0
    assign axi_planes_pos[ 127: 64] = axi_planes[ 127: 64] & pos_mask;  // Plane 1
    assign axi_planes_pos[ 191:128] = axi_planes[ 191:128] & pos_mask;  // Plane 2
    assign axi_planes_pos[ 255:192] = axi_planes[ 255:192] & pos_mask;  // Plane 3

    assign axi_planes_neg[  63:  0] = axi_planes[  63:  0] & neg_mask;  // Plane 0
    assign axi_planes_neg[ 127: 64] = axi_planes[ 127: 64] & neg_mask;  // Plane 1
    assign axi_planes_neg[ 191:128] = axi_planes[ 191:128] & neg_mask;  // Plane 2
    assign axi_planes_neg[ 255:192] = axi_planes[ 255:192] & neg_mask;  // Plane 3

    // -----------------------------------------------------------------------
    // Stages 1-4 : TWO PARALLEL MLB_4 Arrays  *** KEY ARCHITECTURAL CHANGE ***
    //
    // ORIGINAL: One shared MLB_4 instance, driven through a 3-state FSM.
    //           - State 0->1 : process pos lanes (1 cycle)
    //           - State 1->2 : process neg lanes (1 cycle)
    //           => II = 2 cycles per block.
    //
    // CORRECTED: Two independent MLB_4 instances, both driven by valid_in_d1
    //            at the SAME clock cycle, with disjoint activation inputs.
    //           - mac_pos processes axi_planes_pos in parallel with
    //             mac_neg processing axi_planes_neg.
    //           => II = 1 cycle per block (back-to-back streaming supported).
    //
    // Both cores share the SAME awi_planes (weight bit-planes are sign-
    // independent; the sign is embedded in the pos/neg mask on activations).
    // Both cores receive valid_in_d1 simultaneously and complete in lockstep
    // (identical pipeline depth: 3 stages inside MLB_4).
    // -----------------------------------------------------------------------
    wire signed [19:0] pos_sum;    // AND-popcount result: positive lane contribution
    wire               pos_done;   // Fires when pos_sum is valid

    wire signed [19:0] neg_sum;    // AND-popcount result: negative lane contribution
    wire               neg_done;   // Fires at the SAME cycle as pos_done

    MLB_4 mac_pos (
        .mlb     (pos_sum),
        .done    (pos_done),
        .axi     (axi_planes_pos),  // Positive-masked activation planes
        .awi     (awi_planes),       // Full weight planes (shared)
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    MLB_4 mac_neg (
        .mlb     (neg_sum),
        .done    (neg_done),
        .axi     (axi_planes_neg),  // Negative-masked activation planes
        .awi     (awi_planes),       // Full weight planes (shared)
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    // -----------------------------------------------------------------------
    // Stage 5 : Output Combiner  (1-cycle registered subtraction)
    //
    // pos_done and neg_done fire in the same clock cycle because both MLB_4
    // instances have identical pipeline depth and received valid_in_d1
    // simultaneously.  The final result is computed in ONE registered step.
    //
    // result = pos_sum - neg_sum
    //
    // In the original FSM design this required storing pos_sum into pos_reg
    // and waiting a second cycle for neg_sum.  Here it collapses to a single
    // register stage, which is why II drops from 2 to 1.
    //
    // Range of result: pos_sum in [0, 14400], neg_sum in [0, 14400]
    // result in [-14400, 14400], safely within signed 21-bit range.
    // -----------------------------------------------------------------------
    reg signed [20:0] result_reg;
    reg               done_reg;

    always @(posedge clk) begin
        if (rst) begin
            result_reg <= 21'sd0;
            done_reg   <= 1'b0;
        end else begin
            done_reg <= 1'b0;
            if (pos_done & neg_done) begin
                // Single-cycle subtraction replaces the 2-cycle FSM accumulation.
                result_reg <= pos_sum - neg_sum;
                done_reg   <= 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Shared Exponent Pipeline
    //
    // max_exp_x and max_exp_w are available at Stage 0 (registered by bfp_aligner).
    // The MLB_4 array has 3 internal pipeline stages, and Stage 5 registers result_reg.
    // Total latency from Stage 0 to result_reg is exactly 4 clock cycles.
    // We pipeline shared_exponent by 4 stages so that shared_exponent matches
    // wide_integer_sum cycle-for-cycle under continuous II=1 streaming.
    // -----------------------------------------------------------------------
    wire signed [8:0] shared_exponent_comb = $signed({5'b0, max_exp_x})
                                           + $signed({5'b0, max_exp_w})
                                           - 9'sd20;

    reg signed [8:0] exp_pipe_1, exp_pipe_2, exp_pipe_3, exp_pipe_4;

    always @(posedge clk) begin
        if (rst) begin
            exp_pipe_1 <= 9'sd0;
            exp_pipe_2 <= 9'sd0;
            exp_pipe_3 <= 9'sd0;
            exp_pipe_4 <= 9'sd0;
        end else begin
            exp_pipe_1 <= shared_exponent_comb;
            exp_pipe_2 <= exp_pipe_1;
            exp_pipe_3 <= exp_pipe_2;
            exp_pipe_4 <= exp_pipe_3;
        end
    end

    assign wide_integer_sum = result_reg;
    assign mac_done         = done_reg;
    assign shared_exponent  = exp_pipe_4;

endmodule
