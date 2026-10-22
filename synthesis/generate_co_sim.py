import os
import sys
import torch

sys.path.insert(0, 'ResNet20_QAT_Package')
from quantize import _fp8_extract_fields, GLOBAL_FP8_CONFIG

torch.manual_seed(12345)
cfg = GLOBAL_FP8_CONFIG

test_cases = []
# Case 0: All zeros
test_cases.append(('All Zeros', torch.zeros(64), torch.zeros(64)))
# Case 1: All ones (1.0)
test_cases.append(('All Ones 1.0', torch.ones(64), torch.ones(64)))
# Case 2: Alternating +1.0 and -1.0
alt = torch.tensor([1.0, -1.0] * 32)
test_cases.append(('Alternating Signs', alt, alt))
# Case 3: Max finite values (448.0)
test_cases.append(('Max Finite 448.0', torch.full((64,), 448.0), torch.full((64,), 448.0)))
# Case 4: Subnormals (should flush to 0)
test_cases.append(('Subnormals', torch.full((64,), 0.001), torch.full((64,), 0.001)))
# Case 5..30: Random values with various exponents
for i in range(25):
    act = (torch.randn(64) * 10.0).clamp(-448.0, 448.0)
    wgt = (torch.randn(64) * 10.0).clamp(-448.0, 448.0)
    test_cases.append((f'Random Case {i+1}', act, wgt))

print(f'Total test cases generated: {len(test_cases)}')

def pack_fp8_vector(t):
    signs = (t < 0).long()
    E, M, is_norm = _fp8_extract_fields(t.abs(), cfg)
    fp8_bytes = (signs << 7) | (E << 3) | (M & 0x7)
    return fp8_bytes

tb_lines = [
    '`timescale 1ns/1ps',
    'module tb_co_verify;',
    '    reg clk, rst, valid_in;',
    '    reg [511:0] fp8_activations, fp8_weights;',
    '    wire signed [20:0] wide_integer_sum;',
    '    wire signed [8:0]  shared_exponent;',
    '    wire               mac_done;',
    '    integer mismatches = 0;',
    '    integer test_idx = 0;',
    '',
    '    fp8_mlb_top_corrected dut (',
    '        .clk(clk), .rst(rst), .valid_in(valid_in),',
    '        .fp8_activations(fp8_activations), .fp8_weights(fp8_weights),',
    '        .wide_integer_sum(wide_integer_sum), .shared_exponent(shared_exponent),',
    '        .mac_done(mac_done)',
    '    );',
    '',
    '    always #5 clk = ~clk;',
    '',
    '    task run_test(',
    '        input integer idx,',
    '        input [511:0] act,',
    '        input [511:0] wt,',
    '        input signed [20:0] exp_sum,',
    '        input signed [8:0] exp_exp',
    '    );',
    '    begin',
    '        @(posedge clk);',
    '        fp8_activations = act;',
    '        fp8_weights = wt;',
    '        valid_in = 1\'b1;',
    '        @(posedge clk);',
    '        valid_in = 1\'b0;',
    '        // Wait for pipeline latency (3 cycles)',
    '        repeat(3) @(posedge clk);',
    '        if (wide_integer_sum !== exp_sum || shared_exponent !== exp_exp) begin',
    '            $display("[MISMATCH] Test %0d: Expected sum=%0d exp=%0d | Got sum=%0d exp=%0d", idx, exp_sum, exp_exp, wide_integer_sum, shared_exponent);',
    '            mismatches = mismatches + 1;',
    '        end else begin',
    '            $display("[PASS] Test %0d: Sum=%0d, Exp=%0d matches bit-for-bit", idx, wide_integer_sum, shared_exponent);',
    '        end',
    '    end',
    '    endtask',
    '',
    '    initial begin',
    '        clk = 0; rst = 1; valid_in = 0;',
    '        fp8_activations = 0; fp8_weights = 0;',
    '        #20 rst = 0;',
    '        #10;'
]

for idx, (name, act, wgt) in enumerate(test_cases):
    act_bytes = pack_fp8_vector(act)
    wgt_bytes = pack_fp8_vector(wgt)
    
    act_hex = 0
    wgt_hex = 0
    for lane in range(64):
        act_hex |= (int(act_bytes[lane].item()) << (lane * 8))
        wgt_hex |= (int(wgt_bytes[lane].item()) << (lane * 8))
        
    E_a, M_a, is_n_a = _fp8_extract_fields(act.abs(), cfg)
    E_w, M_w, is_n_w = _fp8_extract_fields(wgt.abs(), cfg)
    max_ea = E_a.max().item()
    max_ew = E_w.max().item()
    exp_exp = max_ea + max_ew - 20
    
    mant_a = torch.where(is_n_a, 8 + M_a, torch.zeros_like(M_a))
    mant_w = torch.where(is_n_w, 8 + M_w, torch.zeros_like(M_w))
    shift_a = (max_ea - E_a).clamp(0, 15)
    shift_w = (max_ew - E_w).clamp(0, 15)
    smant_a = torch.div(mant_a, (2 ** shift_a), rounding_mode='trunc')
    smant_w = torch.div(mant_w, (2 ** shift_w), rounding_mode='trunc')
    
    sign_a = (act < 0).long()
    sign_w = (wgt < 0).long()
    s_prod = sign_a ^ sign_w
    
    pos_sum = 0
    neg_sum = 0
    for p in range(4):
        for q in range(4):
            pa = (smant_a >> p) & 1
            pw = (smant_w >> q) & 1
            and_bits = pa & pw
            pos_cnt = (and_bits & (1 - s_prod)).sum().item()
            neg_cnt = (and_bits & s_prod).sum().item()
            pos_sum += pos_cnt << (p + q)
            neg_sum += neg_cnt << (p + q)
            
    py_int_sum = pos_sum - neg_sum
    
    def format_signed(width, val):
        if val < 0:
            return f"-{width}'sd{abs(val)}"
        else:
            return f"{width}'sd{val}"

    s_sum = format_signed(21, py_int_sum)
    s_exp = format_signed(9, exp_exp)

    tb_lines.append(f"        run_test({idx}, 512'h{act_hex:0128x}, 512'h{wgt_hex:0128x}, {s_sum}, {s_exp});")

tb_lines.extend([
    '        #20;',
    '        if (mismatches == 0) $display("\\n*** ALL %0d TEST VECTORS PASSED WITH 0 MISMATCHES! ***\\n", ' + str(len(test_cases)) + ');',
    '        else $display("\\n*** FAILED WITH %0d MISMATCHES! ***\\n", mismatches);',
    '        $finish;',
    '    end',
    'endmodule'
])

with open('tb_co_verify.v', 'w') as f:
    f.write('\n'.join(tb_lines))

print('Wrote tb_co_verify.v successfully.')
