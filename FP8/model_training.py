"""
FP8-BFP MLB-MAC QAT on ResNet-20 / CIFAR-10 — Kaggle Notebook
================================================================

Self-contained script combining all modules:
  1. quantize.py  — FP8 quantization, BFP alignment, MLB-MAC dot product
  2. layers.py    — QAT layer wrappers (MLBConv2d, FP8OnlyConv2d, etc.)
  3. resnet20.py  — ResNet-20 architecture
  4. train.py     — Training pipeline (FP8 baseline + QAT sweep)

INSTRUCTIONS:
  1. Create a new Kaggle notebook
  2. Enable GPU accelerator (Settings -> Accelerator -> GPU T4 x2)
  3. Paste this entire file into a single code cell
  4. Run it

Expected runtime: ~1-2 hours on T4 GPU
"""

import math
import os
import csv
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Force unbuffered output so Kaggle shows progress in real-time
os.environ["PYTHONUNBUFFERED"] = "1"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ======================================================================
# CONFIG
# ======================================================================

# Training hyperparameters
FP8_BASELINE_EPOCHS = 30
QAT_EPOCHS          = 20
BATCH_SIZE          = 128
N_VALUES            = [9, 25, 32, 64, 128, 256, 512]
QAT_MAX_BATCHES     = 0    # 0 = full dataset on GPU (fast enough!)
BASELINE_MAX_BATCHES = 0   # 0 = full dataset

CHECKPOINT_DIR = "./checkpoints"
DATA_DIR       = "./data"
RESULTS_CSV    = "./results.csv"

# ======================================================================
# PART 1: quantize.py — FP8-BFP MLB-MAC Quantization Simulation
# ======================================================================

FP8_BIAS = 7
FP8_MAX_EXP = 15
FP8_MANT_BITS = 3
FP8_NUM_PLANES = 4

FP8_MAX_VAL = (1.0 + (2**FP8_MANT_BITS - 1) / 2**FP8_MANT_BITS) * (
    2.0 ** (FP8_MAX_EXP - FP8_BIAS)
)
FP8_MIN_NORMAL = 2.0 ** (1 - FP8_BIAS)
_SHARED_EXP_OFFSET = 2 * FP8_BIAS + 2 * FP8_MANT_BITS


def _fp8_extract_fields(abs_val):
    log2_val = torch.log2(abs_val.clamp(min=1e-45))
    exp = (torch.floor(log2_val) + FP8_BIAS).clamp(0, FP8_MAX_EXP).long()
    exp = torch.where(abs_val > 0, exp, torch.zeros_like(exp))
    is_normal = exp > 0
    power = (exp.float() - FP8_BIAS)
    scale = (2.0 ** power).clamp(min=1e-45)
    mant_frac = (abs_val / scale - 1.0).clamp(min=0.0)
    mant = torch.round(mant_frac * 8.0).clamp(0, 7).long()
    return exp, mant, is_normal


class FP8Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        sign = x.sign()
        abs_x = x.abs().clamp(max=FP8_MAX_VAL)
        exp, mant, is_normal = _fp8_extract_fields(abs_x)
        power = (exp.float() - FP8_BIAS)
        scale = 2.0 ** power
        x_q = sign * (1.0 + mant.float() / 8.0) * scale
        x_q = torch.where(is_normal, x_q, torch.zeros_like(x_q))
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # STE


def fp8_quantize(x):
    return FP8Quantize.apply(x)


def batched_bfp_align(vals_abs):
    exp, mant, is_normal = _fp8_extract_fields(vals_abs)
    max_exp = exp.max(dim=1).values
    full_mant = torch.where(is_normal, 8 + mant, torch.zeros_like(mant))
    shift_amt = (max_exp.unsqueeze(1) - exp).clamp(min=0, max=15)
    aligned_mant = torch.div(
        full_mant, (2 ** shift_amt.clamp(max=15)), rounding_mode="trunc"
    )
    return aligned_mant, max_exp


def batched_mlb_dot_product(act_fp8, wgt_fp8, N, sort_by_exp=True):
    batch, K = act_fp8.shape
    C_out = wgt_fp8.shape[0]
    assert wgt_fp8.shape[1] == K

    # Exponent-sorted permutation (SINGLE shared permutation)
    if sort_by_exp:
        wgt_exp, _, _ = _fp8_extract_fields(wgt_fp8.abs())
        mean_wgt_exp = wgt_exp.float().mean(dim=0)
        sort_idx = torch.argsort(mean_wgt_exp)
        act_fp8 = act_fp8[:, sort_idx]
        wgt_fp8 = wgt_fp8[:, sort_idx]

    num_chunks = math.ceil(K / N)#zero pads if K_padded is greater than K
    K_padded = num_chunks * N

    if K_padded > K:
        act_padded = F.pad(act_fp8, (0, K_padded - K))
        wgt_padded = F.pad(wgt_fp8, (0, K_padded - K))
    else:
        act_padded = act_fp8
        wgt_padded = wgt_fp8

    act_reshaped = act_padded.view(batch, num_chunks, N)
    wgt_reshaped = wgt_padded.view(C_out, num_chunks, N)

    sign_act = act_reshaped < 0#collects sign
    sign_wgt = wgt_reshaped < 0

    aligned_act, max_exp_x = batched_bfp_align(act_reshaped.abs().view(batch * num_chunks, N))#finds the max exponent
    aligned_wgt, max_exp_w = batched_bfp_align(wgt_reshaped.abs().view(C_out * num_chunks, N))

    aligned_act = aligned_act.view(batch, num_chunks, N)
    max_exp_x = max_exp_x.view(batch, num_chunks)
    aligned_wgt = aligned_wgt.view(C_out, num_chunks, N)
    max_exp_w = max_exp_w.view(C_out, num_chunks)

    sign_act_f = torch.where(sign_act, -1.0, 1.0)#converts sign to -1 and 1
    sign_wgt_f = torch.where(sign_wgt, -1.0, 1.0)

    signed_act = aligned_act.float() * sign_act_f#adding the sign
    signed_wgt = aligned_wgt.float() * sign_wgt_f

    act_perm = signed_act.permute(1, 0, 2)
    wgt_perm = signed_wgt.permute(1, 0, 2)

    raw_dot = torch.bmm(act_perm, wgt_perm.transpose(1, 2))#multiplication

    scale_x = (2.0 ** max_exp_x.float()).t().unsqueeze(2)
    scale_w = (2.0 ** max_exp_w.float()).t().unsqueeze(1)

    chunk_results = raw_dot * scale_x * scale_w * (2.0 ** -_SHARED_EXP_OFFSET)#converts from FP8 to integer
    return chunk_results.sum(dim=0)#sums and returns 


def batched_fp8_dot_product(act_fp8, wgt_fp8):
    return act_fp8 @ wgt_fp8.t()


# ======================================================================
# PART 2: layers.py — QAT Layer Wrappers
# ======================================================================

def _pair(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class _MLBConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, N):
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        with torch.no_grad():
            x_q = fp8_quantize(x)
            w_q = fp8_quantize(weight)

            B, C_in, H_in, W_in = x_q.shape
            C_out, C_in_g, kH, kW = w_q.shape
            K = C_in_g * kH * kW

            H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
            L = H_out * W_out

            x_unf = F.unfold(x_q, (kH, kW), dilation=dilation, padding=padding, stride=stride)

            act_flat = x_unf.permute(0, 2, 1).reshape(B * L, K)
            wgt_flat = w_q.reshape(C_out, K)

            out_flat = batched_mlb_dot_product(act_flat, wgt_flat, N)

            output = out_flat.reshape(B, L, C_out).permute(0, 2, 1)
            output = output.reshape(B, C_out, H_out, W_out)

            if bias is not None:
                output = output + bias.reshape(1, -1, 1, 1)

        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class MLBConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, N=64):
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.N = N

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _MLBConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups, self.N,
        )


class _MLBLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, N):
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None

        with torch.no_grad():
            x_q = fp8_quantize(x)
            w_q = fp8_quantize(weight)
            output = batched_mlb_dot_product(x_q, w_q, N)
            if bias is not None:
                output = output + bias.unsqueeze(0)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(x)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None


class MLBLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, N=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = N
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _MLBLinearFunction.apply(x, self.weight, self.bias, self.N)


# -- FP8-only layers (no BFP, no MLB) -- used for FP8 baseline --

class _FP8OnlyConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups):
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        with torch.no_grad():
            x_q = fp8_quantize(x)
            w_q = fp8_quantize(weight)
            B, C_in, H_in, W_in = x_q.shape
            C_out, C_in_g, kH, kW = w_q.shape
            K = C_in_g * kH * kW
            H_out = (H_in + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
            W_out = (W_in + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
            L = H_out * W_out
            x_unf = F.unfold(x_q, (kH, kW), dilation=dilation, padding=padding, stride=stride)
            act_flat = x_unf.permute(0, 2, 1).reshape(B * L, K)
            wgt_flat = w_q.reshape(C_out, K)
            out_flat = batched_fp8_dot_product(act_flat, wgt_flat)
            output = out_flat.reshape(B, L, C_out).permute(0, 2, 1)
            output = output.reshape(B, C_out, H_out, W_out)
            if bias is not None:
                output = output + bias.reshape(1, -1, 1, 1)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            bias = None
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                x.shape, weight, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                x, weight.shape, grad_output,
                ctx.stride, ctx.padding, ctx.dilation, ctx.groups,
            )
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None, None, None, None


class FP8OnlyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _FP8OnlyConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


class _FP8OnlyLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        with torch.no_grad():
            x_q = fp8_quantize(x)
            w_q = fp8_quantize(weight)
            output = batched_fp8_dot_product(x_q, w_q)
            if bias is not None:
                output = output + bias.unsqueeze(0)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x, weight, bias = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(x)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias


class FP8OnlyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _FP8OnlyLinearFunction.apply(x, self.weight, self.bias)


# ======================================================================
# PART 3: resnet20.py — ResNet-20 for CIFAR-10
# ======================================================================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, N=64, quantize=True):
        super().__init__()
        if quantize is True:
            self.conv1 = MLBConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, N=N)
            self.conv2 = MLBConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, N=N)
        elif quantize == 'fp8':
            self.conv1 = FP8OnlyConv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.conv2 = FP8OnlyConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10, N=64, quantize=True):
        super().__init__()
        if quantize is True:
            self.conv1 = MLBConv2d(3, 16, 3, stride=1, padding=1, bias=False, N=N)
        elif quantize == 'fp8':
            self.conv1 = FP8OnlyConv2d(3, 16, 3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 3, stride=1, N=N, quantize=quantize)
        self.layer2 = self._make_layer(16, 32, 3, stride=2, N=N, quantize=quantize)
        self.layer3 = self._make_layer(32, 64, 3, stride=2, N=N, quantize=quantize)

        if quantize is True:
            self.linear = MLBLinear(64, num_classes, N=N)
        elif quantize == 'fp8':
            self.linear = FP8OnlyLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, stride, N, quantize):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        ch_in = in_channels
        for s in strides:
            layers.append(BasicBlock(ch_in, out_channels, s, N=N, quantize=quantize))
            ch_in = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================
# PART 4: Training Pipeline
# ======================================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cifar10_loaders(batch_size=128, data_dir="./data", num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test,
    )

    persistent = num_workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, max_batches=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_batches = min(max_batches, len(loader)) if max_batches > 0 else len(loader)
            print(
                f"  [{epoch+1}/{total_epochs}] batch {batch_idx+1}/{n_batches}  "
                f"loss={running_loss/total:.4f}  acc={100.*correct/total:.2f}%  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_fp8_baseline(device, epochs, batch_size, checkpoint_dir, data_dir, max_batches=0):
    print("=" * 60)
    print("Phase 1 -- FP8 Baseline Training (FP8 quantize, standard matmul)")
    print("=" * 60)

    os.makedirs(checkpoint_dir, exist_ok=True)
    model = ResNet20(quantize='fp8').to(device)
    print(f"Model: ResNet-20 FP8-only ({count_parameters(model):,} parameters)")
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size, data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            max_batches=max_batches,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                       max_batches=max_batches)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.2f}%  "
            f"lr={lr_now:.6f}",
            flush=True,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(checkpoint_dir, "fp8_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  * New best: {best_acc:.2f}% -> saved {ckpt_path}")

    best_path = os.path.join(checkpoint_dir, "fp8_best.pth")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, final_acc = evaluate(model, test_loader, criterion, device, max_batches=max_batches)
    print(f"\nFP8 Baseline -- Best test accuracy: {final_acc:.2f}%\n")
    return final_acc


def train_qat(N, device, epochs, batch_size, checkpoint_dir, data_dir, max_batches=0):
    print(f"\n{'-' * 60}")
    print(f"QAT: N = {N}")
    print(f"{'-' * 60}")

    model = ResNet20(quantize=True, N=N).to(device)
    fp8_path = os.path.join(checkpoint_dir, "fp8_best.pth")
    if not os.path.exists(fp8_path):
        raise FileNotFoundError(f"FP8 baseline checkpoint not found: {fp8_path}")

    state_dict = torch.load(fp8_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded FP8 baseline weights from {fp8_path}")

    train_loader, test_loader = get_cifar10_loaders(batch_size, data_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs,
            max_batches=max_batches,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device,
                                       max_batches=max_batches)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  N={N:3d}  epoch {epoch+1:3d}/{epochs}  "
            f"train={train_acc:.2f}%  test={test_acc:.2f}%  "
            f"lr={lr_now:.6f}",
            flush=True,
        )

        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(checkpoint_dir, f"qat_N{N}_best.pth")
            torch.save(model.state_dict(), ckpt_path)

    best_path = os.path.join(checkpoint_dir, f"qat_N{N}_best.pth")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, final_acc = evaluate(model, test_loader, criterion, device, max_batches=max_batches)
    print(f"  N={N}: Best test accuracy = {final_acc:.2f}%")
    return final_acc


# ======================================================================
# MAIN — Run everything
# ======================================================================

if __name__ == "__main__":
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"N values: {N_VALUES}")
    print(f"FP8 Baseline: {FP8_BASELINE_EPOCHS} epochs, full dataset")
    print(f"QAT: {QAT_EPOCHS} epochs per N, "
          f"{'full dataset' if QAT_MAX_BATCHES == 0 else f'{QAT_MAX_BATCHES} batches/epoch'}")
    print()

    # -- Phase 1: FP8 Baseline --
    fp8_acc = train_fp8_baseline(
        device=device,
        epochs=FP8_BASELINE_EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_dir=CHECKPOINT_DIR,
        data_dir=DATA_DIR,
        max_batches=BASELINE_MAX_BATCHES,
    )

    # -- Phase 2: QAT Sweep --
    results = [("fp8_baseline", fp8_acc)]

    for N in N_VALUES:
        try:
            acc = train_qat(
                N=N,
                device=device,
                epochs=QAT_EPOCHS,
                batch_size=BATCH_SIZE,
                checkpoint_dir=CHECKPOINT_DIR,
                data_dir=DATA_DIR,
                max_batches=QAT_MAX_BATCHES,
            )
            results.append((str(N), acc))
        except Exception as e:
            print(f"  X N={N} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((str(N), -1.0))

    # -- Write results.csv --
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "test_top1_accuracy"])
        for n_str, acc in results:
            writer.writerow([n_str, f"{acc:.2f}"])

    # -- Print summary table --
    print("\n" + "=" * 50)
    print("  Results Summary")
    print("=" * 50)
    for n_str, acc in results:
        tag = ""
        if n_str == "64":
            tag = "  <-- original hardware block size"
        if acc < 0:
            print(f"  N={n_str:>5s} : FAILED")
        else:
            print(f"  N={n_str:>5s} : {acc:.2f}%{tag}")
    print("=" * 50)
    print(f"\nResults saved to {RESULTS_CSV}")

    # -- Commentary --
    print("\nCommentary on accuracy vs N:")
    print("-" * 50)
    qat_results = [(n, a) for n, a in results if n != "fp8_baseline" and a >= 0]
    if qat_results:
        best_n, best_a = max(qat_results, key=lambda x: x[1])
        worst_n, worst_a = min(qat_results, key=lambda x: x[1])
        print(f"  Best QAT accuracy:   N={best_n} -> {best_a:.2f}%")
        print(f"  Worst QAT accuracy:  N={worst_n} -> {worst_a:.2f}%")
        print(f"  FP8 baseline:        {fp8_acc:.2f}%  (FP8 quantize, standard matmul)")
        if fp8_acc > 0:
            print(f"  Best gap to FP8:  {fp8_acc - best_a:.2f}pp")
            print(f"  Worst gap to FP8: {fp8_acc - worst_a:.2f}pp")
        print()
        print("  Exponent-sorted BFP (enabled by default):")
        print("    Elements within each block of size N are sorted by FP8")
        print("    exponent before BFP alignment, minimising the max right-")
        print("    shift within each block and preserving more mantissa bits.")
        print()
        print("  Smaller N -> more blocks, finer exponent grouping,")
        print("  less precision loss per block (approaching FP8 baseline).")
        print("  Larger N -> coarser blocks, small values in same block as")
        print("  large values may be shifted to zero (accuracy drops).")

    print("\n\nDONE!")
*