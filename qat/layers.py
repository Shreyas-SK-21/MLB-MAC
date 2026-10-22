"""
layers.py
=========
PyTorch neural network layers with Quantization-Aware Training (QAT) autograd functions.

Provides:
- MLBConv2d: Convolutional layer using BFP-aligned Multi-Level Binary dot products.
- MLBLinear: Fully-connected layer using BFP-aligned Multi-Level Binary dot products.
- FP8OnlyConv2d: Native FP8 Convolution baseline (per-tensor/standard matmul, no BFP).
- FP8OnlyLinear: Native FP8 Linear baseline (per-tensor/standard matmul, no BFP).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize import (
    fp8_quantize,
    batched_mlb_dot_product,
    batched_native_fp8_dot_product,
)


def _pair(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


# ==============================================================================
# 1. BFP-Aligned Multi-Level Binary (MLB) Layers
# ==============================================================================

class _MLBConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, N, sort_by_exp):
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

            out_flat = batched_mlb_dot_product(act_flat, wgt_flat, N=N, sort_by_exp=sort_by_exp)

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

        # Straight-Through Estimator (STE) using floating-point backward convolution
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

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class MLBConv2d(nn.Module):
    """
    2D Convolution using BFP-aligned Multi-Level Binary dot products.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        N: int = 64,
        sort_by_exp: bool = True,
    ):
        super().__init__()
        assert groups == 1, "Groups > 1 currently not supported in MLBConv2d"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.N = N
        self.sort_by_exp = sort_by_exp

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _MLBConv2dFunction.apply(
            x, self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
            self.N, self.sort_by_exp,
        )


class _MLBLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, N, sort_by_exp):
        if bias is not None:
            ctx.save_for_backward(x, weight, bias)
        else:
            ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None

        with torch.no_grad():
            x_q = fp8_quantize(x)
            w_q = fp8_quantize(weight)
            output = batched_mlb_dot_product(x_q, w_q, N=N, sort_by_exp=sort_by_exp)
            if bias is not None:
                output = output + bias.unsqueeze(0)
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
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(x)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias, None, None


class MLBLinear(nn.Module):
    """
    Linear layer using BFP-aligned Multi-Level Binary dot products.
    """
    def __init__(self, in_features: int, out_features: int, bias=True, N: int = 64, sort_by_exp: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = N
        self.sort_by_exp = sort_by_exp
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _MLBLinearFunction.apply(x, self.weight, self.bias, self.N, self.sort_by_exp)


# ==============================================================================
# 2. Native FP8 Baseline Layers (No BFP, No MLB)
# ==============================================================================

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
            output = F.conv2d(x_q, w_q, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        assert groups == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            output = F.linear(x_q, w_q, bias)
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
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(x)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=0)
        return grad_input, grad_weight, grad_bias


class FP8OnlyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _FP8OnlyLinearFunction.apply(x, self.weight, self.bias)
