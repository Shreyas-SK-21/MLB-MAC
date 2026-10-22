"""
setup_sweep.py
Sets up the entire FP8_Corrected directory tree for all N in {9, 25, 32, 64, 128, 256, 512}.
Generates:
  1. FP8_corrected.v with unrolled parallel dual MLB_4 arrays (II = 1)
  2. Complete synthesis scripts for SkyWater 130nm library
  3. Copies all submodules and library files for self-contained execution
"""

import os
import shutil

BASE_SRC = r"c:\Users\shrey\Desktop\Projects\MLB-MAC\FP8\synthesized_FP8"
BASE_DST = r"c:\Users\shrey\Desktop\Projects\MLB-MAC\FP8_Corrected"
TECH_LIB_DIR = r"c:\Users\shrey\Desktop\Projects\MLB-MAC\FP8_Corrected\AND_Popcount\Synth files"

N_PARAMS = {
    9:   {'act_w': 71,   'axi_w': 35,   'sign_w': 8,   'mlb_w': 16, 'sum_w': 17},
    25:  {'act_w': 199,  'axi_w': 99,   'sign_w': 24,  'mlb_w': 17, 'sum_w': 18},
    32:  {'act_w': 255,  'axi_w': 127,  'sign_w': 31,  'mlb_w': 18, 'sum_w': 19},
    64:  {'act_w': 511,  'axi_w': 255,  'sign_w': 63,  'mlb_w': 19, 'sum_w': 20},
    128: {'act_w': 1023, 'axi_w': 511,  'sign_w': 127, 'mlb_w': 20, 'sum_w': 21},
    256: {'act_w': 2047, 'axi_w': 1023, 'sign_w': 255, 'mlb_w': 21, 'sum_w': 22},
    512: {'act_w': 4095, 'axi_w': 2047, 'sign_w': 511, 'mlb_w': 22, 'sum_w': 23},
}

def generate_fp8_corrected_verilog(N, p):
    act_w = p['act_w']
    axi_w = p['axi_w']
    sign_w = p['sign_w']
    mlb_w = p['mlb_w']
    sum_w = p['sum_w']
    
    p0_low, p0_high = 0, N - 1
    p1_low, p1_high = N, 2 * N - 1
    p2_low, p2_high = 2 * N, 3 * N - 1
    p3_low, p3_high = 3 * N, 4 * N - 1

    code = f"""// ===========================================================================
// FP8_corrected.v  --  fp8_mlb_top_corrected  (AND-Popcount, N={N}, M=4)
// ===========================================================================
// Fully unrolled parallel dual MLB_4 architecture for sustained II = 1.
// Eliminates 2-cycle FSM time-multiplexing overhead.
// ===========================================================================

`timescale 1ns/1ps

module fp8_mlb_top_corrected (
    output signed [{sum_w}:0] wide_integer_sum,
    output signed [8:0]  shared_exponent,
    output               mac_done,
    input                clk,
    input                rst,
    input                valid_in,
    input [{act_w}:0]        fp8_activations,
    input [{act_w}:0]        fp8_weights
);

    // -----------------------------------------------------------------------
    // Stage 0 : BFP Alignment (1-cycle registered)
    // -----------------------------------------------------------------------
    wire [{axi_w}:0] axi_planes;
    wire [{axi_w}:0] awi_planes;
    wire [3:0]   max_exp_x;
    wire [3:0]   max_exp_w;

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
    // Stage 0 (parallel) : Sign XOR, registered to match aligned_planes
    // -----------------------------------------------------------------------
    genvar i;
    wire [{sign_w}:0] sign_k;
    generate
        for (i = 0; i < {N}; i = i + 1) begin : gen_sign
            assign sign_k[i] = fp8_activations[i*8 + 7] ^ fp8_weights[i*8 + 7];
        end
    endgenerate

    reg [{sign_w}:0] sign_k_d1;
    reg              valid_in_d1;

    always @(posedge clk) begin
        if (rst) begin
            sign_k_d1   <= {N}'b0;
            valid_in_d1 <= 1'b0;
        end else begin
            sign_k_d1   <= sign_k;
            valid_in_d1 <= valid_in;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 1 : Sign Masking
    // -----------------------------------------------------------------------
    wire [{sign_w}:0] pos_mask = ~sign_k_d1;
    wire [{sign_w}:0] neg_mask =  sign_k_d1;

    wire [{axi_w}:0] axi_planes_pos;
    wire [{axi_w}:0] axi_planes_neg;

    assign axi_planes_pos[{p0_high}:{p0_low}] = axi_planes[{p0_high}:{p0_low}] & pos_mask;
    assign axi_planes_pos[{p1_high}:{p1_low}] = axi_planes[{p1_high}:{p1_low}] & pos_mask;
    assign axi_planes_pos[{p2_high}:{p2_low}] = axi_planes[{p2_high}:{p2_low}] & pos_mask;
    assign axi_planes_pos[{p3_high}:{p3_low}] = axi_planes[{p3_high}:{p3_low}] & pos_mask;

    assign axi_planes_neg[{p0_high}:{p0_low}] = axi_planes[{p0_high}:{p0_low}] & neg_mask;
    assign axi_planes_neg[{p1_high}:{p1_low}] = axi_planes[{p1_high}:{p1_low}] & neg_mask;
    assign axi_planes_neg[{p2_high}:{p2_low}] = axi_planes[{p2_high}:{p2_low}] & neg_mask;
    assign axi_planes_neg[{p3_high}:{p3_low}] = axi_planes[{p3_high}:{p3_low}] & neg_mask;

    // -----------------------------------------------------------------------
    // Stages 1-4 : TWO PARALLEL MLB_4 Arrays (II = 1)
    // -----------------------------------------------------------------------
    wire signed [{mlb_w}:0] pos_sum;
    wire               pos_done;

    wire signed [{mlb_w}:0] neg_sum;
    wire               neg_done;

    MLB_4 mac_pos (
        .mlb     (pos_sum),
        .done    (pos_done),
        .axi     (axi_planes_pos),
        .awi     (awi_planes),
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    MLB_4 mac_neg (
        .mlb     (neg_sum),
        .done    (neg_done),
        .axi     (axi_planes_neg),
        .awi     (awi_planes),
        .clk     (clk),
        .rst     (rst),
        .valid_in(valid_in_d1)
    );

    // -----------------------------------------------------------------------
    // Stage 5 : Output Combiner (1-cycle registered subtraction)
    // -----------------------------------------------------------------------
    reg signed [{sum_w}:0] result_reg;
    reg                    done_reg;

    always @(posedge clk) begin
        if (rst) begin
            result_reg <= {sum_w + 1}'sd0;
            done_reg   <= 1'b0;
        end else begin
            done_reg <= 1'b0;
            if (pos_done & neg_done) begin
                result_reg <= pos_sum - neg_sum;
                done_reg   <= 1'b1;
            end
        end
    end

    // -----------------------------------------------------------------------
    // Shared Exponent Pipeline (4 stages to match MLB latency)
    // -----------------------------------------------------------------------
    wire signed [8:0] shared_exponent_comb = $signed({{5'b0, max_exp_x}})
                                           + $signed({{5'b0, max_exp_w}})
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
"""
    return code


def setup_all():
    tech_files = [
        "sky130_fd_sc_hd__tt_025C_1v80.lib",
        "sky130_fd_sc_hd.tlef",
        "sky130_fd_sc_hd_merged.lef"
    ]
    
    for N, p in N_PARAMS.items():
        src_folder_name = f"M=4, N={N}(and implementation)"
        src_n_dir = os.path.join(BASE_SRC, src_folder_name)
        dst_n_dir = os.path.join(BASE_DST, src_folder_name)
        
        print(f"[*] Setting up N={N} in {dst_n_dir}...")
        
        # 1. MLB BFP Corrected
        corr_verilog_dir = os.path.join(dst_n_dir, "MLB BFP Corrected", "Verilog codes")
        corr_synth_dir   = os.path.join(dst_n_dir, "MLB BFP Corrected", "Synth files")
        os.makedirs(corr_verilog_dir, exist_ok=True)
        os.makedirs(corr_synth_dir, exist_ok=True)
        
        # Copy common modules from src MLB BFP
        src_mlb_v = os.path.join(src_n_dir, "MLB BFP", "Verilog codes")
        for f in ["bpf_aligner.v", "and_popcount_4_bit.v", "MLB_MAC_unit.v", "MLB_MAC_4_bit.v"]:
            src_f = os.path.join(src_mlb_v, f)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(corr_verilog_dir, f))
                shutil.copy2(src_f, os.path.join(corr_synth_dir, f))
        
        # Generate FP8_corrected.v
        fp8_corr_code = generate_fp8_corrected_verilog(N, p)
        with open(os.path.join(corr_verilog_dir, "FP8_corrected.v"), "w") as f:
            f.write(fp8_corr_code)
        with open(os.path.join(corr_synth_dir, "FP8_corrected.v"), "w") as f:
            f.write(fp8_corr_code)
            
        # Copy tech files into corr_synth_dir
        for tf in tech_files:
            src_tf = os.path.join(TECH_LIB_DIR, tf)
            if os.path.exists(src_tf):
                shutil.copy2(src_tf, os.path.join(corr_synth_dir, tf))
                
        # Generate synth_and_corrected_130.ys
        ys_corr = f"""design -reset
read_verilog bpf_aligner.v and_popcount_4_bit.v MLB_MAC_unit.v MLB_MAC_4_bit.v FP8_corrected.v
hierarchy -check -top fp8_mlb_top_corrected
synth -top fp8_mlb_top_corrected
dfflibmap -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
abc -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
tee -o fp8_mlb_corrected_area.rpt stat -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
write_verilog fp8_mlb_corrected_synth.v
"""
        with open(os.path.join(corr_synth_dir, "synth_and_corrected_130.ys"), "w") as f:
            f.write(ys_corr)

        # 2. MLB BFP Original (copy existing)
        orig_verilog_dir = os.path.join(dst_n_dir, "MLB BFP Original", "Verilog codes")
        orig_synth_dir   = os.path.join(dst_n_dir, "MLB BFP Original", "Synth files")
        os.makedirs(orig_verilog_dir, exist_ok=True)
        os.makedirs(orig_synth_dir, exist_ok=True)
        
        for f in os.listdir(src_mlb_v):
            src_f = os.path.join(src_mlb_v, f)
            shutil.copy2(src_f, os.path.join(orig_verilog_dir, f))
            shutil.copy2(src_f, os.path.join(orig_synth_dir, f))
            
        for tf in tech_files:
            src_tf = os.path.join(TECH_LIB_DIR, tf)
            if os.path.exists(src_tf):
                shutil.copy2(src_tf, os.path.join(orig_synth_dir, tf))
                
        ys_orig = f"""design -reset
read_verilog bpf_aligner.v and_popcount_4_bit.v MLB_MAC_unit.v MLB_MAC_4_bit.v FP8.v
hierarchy -check -top fp8_mlb_top
synth -top fp8_mlb_top
dfflibmap -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
abc -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
tee -o fp8_mlb_top_area.rpt stat -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
write_verilog fp8_mlb_synth.v
"""
        with open(os.path.join(orig_synth_dir, "synth_all_130.ys"), "w") as f:
            f.write(ys_orig)

        # 3. INT MAC BFP (Baseline 1)
        int_verilog_dir = os.path.join(dst_n_dir, "INT MAC BFP", "Verilog codes")
        int_synth_dir   = os.path.join(dst_n_dir, "INT MAC BFP", "Synth files")
        os.makedirs(int_verilog_dir, exist_ok=True)
        os.makedirs(int_synth_dir, exist_ok=True)
        
        src_int_v = os.path.join(src_n_dir, "INT MAC BFP", "Verilog codes")
        for f in os.listdir(src_int_v):
            src_f = os.path.join(src_int_v, f)
            shutil.copy2(src_f, os.path.join(int_verilog_dir, f))
            shutil.copy2(src_f, os.path.join(int_synth_dir, f))
            
        for tf in tech_files:
            src_tf = os.path.join(TECH_LIB_DIR, tf)
            if os.path.exists(src_tf):
                shutil.copy2(src_tf, os.path.join(int_synth_dir, tf))
                
        # In INT MAC BFP, files to read
        # check which bfp aligner exists
        aligner_name = "bfp_aligner.v" if os.path.exists(os.path.join(int_synth_dir, "bfp_aligner.v")) else "bpf_aligner.v"
        ys_int = f"""design -reset
read_verilog {aligner_name} int_mac_{N}.v fp8_int_top.v
hierarchy -check -top fp8_int_top
synth -top fp8_int_top
dfflibmap -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
abc -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
tee -o fp8_int_top_area.rpt stat -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
write_verilog fp8_int_synth.v
"""
        with open(os.path.join(int_synth_dir, "synth_all_130.ys"), "w") as f:
            f.write(ys_int)

        # 4. Native INT MAC (Baseline 2: Native FP8 MAC)
        nat_verilog_dir = os.path.join(dst_n_dir, "Native INT MAC", "Verilog codes")
        nat_synth_dir   = os.path.join(dst_n_dir, "Native INT MAC", "Synth files")
        os.makedirs(nat_verilog_dir, exist_ok=True)
        os.makedirs(nat_synth_dir, exist_ok=True)
        
        src_nat_v = os.path.join(src_n_dir, "Native INT MAC", "Verilog codes")
        for f in os.listdir(src_nat_v):
            src_f = os.path.join(src_nat_v, f)
            shutil.copy2(src_f, os.path.join(nat_verilog_dir, f))
            shutil.copy2(src_f, os.path.join(nat_synth_dir, f))
            
        for tf in tech_files:
            src_tf = os.path.join(TECH_LIB_DIR, tf)
            if os.path.exists(src_tf):
                shutil.copy2(src_tf, os.path.join(nat_synth_dir, tf))
                
        ys_nat = f"""design -reset
read_verilog FP8_NATIVE_MAC_{N}.v
hierarchy -check -top FP8_NATIVE_MAC_{N}
synth -top FP8_NATIVE_MAC_{N}
dfflibmap -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
abc -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
tee -o fp8_native_mac_area.rpt stat -liberty sky130_fd_sc_hd__tt_025C_1v80.lib
write_verilog fp8_native_mac_synth.v
"""
        with open(os.path.join(nat_synth_dir, "synth_all_130.ys"), "w") as f:
            f.write(ys_nat)

    print("[SUCCESS] All directory structures and scripts created successfully!")

if __name__ == "__main__":
    setup_all()
