"""
quantization.py
===============
Multi-Level Binary (MLB) Quantization core implementation.

Mathematical formulation
------------------------
    W ~= W_MLB = sum_{i=1}^{M}  alpha_i * B_i

where:
    B_i  in {-1, +1}^{shape(W)}   -- binary basis
    alpha_i  in R_{>0}             -- per-level scalar

Iterative residual decomposition (Algorithm 1):
    R_0 = W
    for i = 1 ... M:
        B_i = sign(R_{i-1})            (sign(0) -> +1)
        alpha_i = mean(|R_{i-1}|)
        R_i = R_{i-1} - alpha_i * B_i

Hierarchical M=8 decomposition (new in v2):
    Stage 1: standard M=4 MLB  ->  W4, residual R4 = W - W4
    Stage 2: standard M=4 MLB on R4  ->  R4_MLB
    Final:   W_hierarchical = W4 + R4_MLB  (equivalent to M=8)

Gradient flow (Straight-Through Estimator):
    dL/dW ~= dL/dW_MLB   (identity through quantization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Low-level MLB decomposition helpers
# ---------------------------------------------------------------------------

def mlb_decompose(
    W: torch.Tensor,
    M: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Perform iterative residual decomposition of weight tensor W into M
    binary bases and their scalar coefficients.

    Parameters
    ----------
    W : torch.Tensor
        Full-precision weight tensor of arbitrary shape.
    M : int
        Number of binary levels (must be ≥ 1).

    Returns
    -------
    alphas : list of M scalar Tensors (on same device as W)
        α_i = mean(|R_{i-1}|)
    bases  : list of M binary Tensors ∈ {-1, +1} (same shape as W)
        B_i = sign(R_{i-1})

    Notes
    -----
    sign(0) is treated as +1 via the `torch.sign` + clamp idiom.
    """
    assert M >= 1, "M must be at least 1."
    alphas: List[torch.Tensor] = []
    bases: List[torch.Tensor] = []

    R = W.detach().clone().float()  # work in float32 always
    for _ in range(M):
        # B_i = sign(R_{i-1}), mapping 0 → +1
        B = torch.sign(R)
        B[B == 0] = 1.0

        # α_i = mean(|R_{i-1}|)  — scalar
        alpha = R.abs().mean()

        # R_i = R_{i-1} − α_i · B_i
        R = R - alpha * B

        alphas.append(alpha)
        bases.append(B)

    return alphas, bases


def mlb_reconstruct(
    alphas: List[torch.Tensor],
    bases: List[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstruct the approximated weight tensor from M binary levels.

    W_MLB = Σ_{i=1}^{M} α_i · B_i

    Parameters
    ----------
    alphas : list of scalar Tensors
    bases  : list of binary Tensors ∈ {-1, +1}

    Returns
    -------
    W_MLB : torch.Tensor (same shape and device as bases[0])
    """
    W_mlb = torch.zeros_like(bases[0])
    for alpha, B in zip(alphas, bases):
        W_mlb = W_mlb + alpha * B
    return W_mlb


def quantization_error(W: torch.Tensor, W_mlb: torch.Tensor) -> torch.Tensor:
    """
    Frobenius norm of quantization error  ‖W − W_MLB‖_F.
    """
    return torch.norm(W.float() - W_mlb.float())


# ---------------------------------------------------------------------------
# Straight-Through Estimator (STE) autograd function
# ---------------------------------------------------------------------------

class _MLBQuantizerSTE(torch.autograd.Function):
    """
    Applies MLB quantization in the forward pass and passes gradients
    straight through in the backward pass (STE approximation).

    This allows the *latent* FP32 weights to be updated by back-propagation
    while the *effective* weights used in the forward pass are MLB-quantized.
    """

    @staticmethod
    def forward(ctx, W: torch.Tensor, M: int) -> torch.Tensor:  # type: ignore[override]
        """Quantize W using M-level MLB decomposition."""
        alphas, bases = mlb_decompose(W, M)
        W_mlb = mlb_reconstruct(alphas, bases)
        return W_mlb.to(W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """STE: gradient passes through the quantizer unchanged."""
        # Second return value is for M (not a tensor, so None)
        return grad_output, None


def mlb_quantize_ste(W: torch.Tensor, M: int) -> torch.Tensor:
    """
    Quantize weight tensor W using M-level MLB via STE.

    During *training* gradients flow through unchanged (STE).
    During *inference* (.no_grad context) this is equivalent to a plain
    MLB decomposition + reconstruction.

    Parameters
    ----------
    W : torch.Tensor  -- latent FP32 weight
    M : int           -- number of binary levels

    Returns
    -------
    W_mlb : torch.Tensor (same shape and device as W)
    """
    return _MLBQuantizerSTE.apply(W, M)


# ---------------------------------------------------------------------------
# MLB-quantized Conv2d layer
# ---------------------------------------------------------------------------

class MLBConv2d(nn.Module):
    """
    A drop-in replacement for nn.Conv2d that applies MLB quantization to
    its weight tensor.

    In QAT mode  (qat=True):
        - Latent FP32 weights are stored as normal parameters.
        - During forward(), weights are quantized via STE so gradients
          flow back to update the latent weights.

    In post-training quantization mode (qat=False):
        - Call `quantize_weights()` once after FP32 training to bake
          MLB-quantized weights into the layer; no STE is used.
        - During forward() the stored quantized weights are used directly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = False,
        M: int = 4,
        qat: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.M = M
        self.qat = qat

        # Latent FP32 weight (always a learnable parameter)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias_param = None

        # Buffer to store baked PTQ weights (populated by quantize_weights())
        self.register_buffer("_quantized_weight", None)
        self._ptq_applied = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize_weights(self) -> Dict:
        """
        Perform post-training MLB quantization.

        Decomposes the current FP32 weight into M binary levels, stores the
        reconstructed tensor in `_quantized_weight`, and returns per-level
        metadata (alphas, bases).

        Returns
        -------
        info : dict with keys 'alphas', 'bases', 'quantization_error'
        """
        with torch.no_grad():
            W = self.weight.data
            alphas, bases = mlb_decompose(W, self.M)
            W_mlb = mlb_reconstruct(alphas, bases).to(W.dtype)
            self._quantized_weight = W_mlb
            self._ptq_applied = True
            q_err = quantization_error(W, W_mlb).item()
        return {
            "alphas": [a.item() for a in alphas],
            "bases": bases,
            "quantization_error": q_err,
        }

    def set_qat(self, qat: bool):
        """Toggle QAT mode on or off."""
        self.qat = qat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ptq_applied and not self.qat:
            # Post-training quantization path: use baked weights
            W_eff = self._quantized_weight
        elif self.qat:
            # QAT path: quantize on-the-fly via STE
            W_eff = mlb_quantize_ste(self.weight, self.M)
        else:
            # FP32 baseline path (no quantization)
            W_eff = self.weight

        return F.conv2d(
            x,
            W_eff,
            bias=self.bias_param,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, M={self.M}, qat={self.qat}"
        )


# ---------------------------------------------------------------------------
# MLB-quantized Linear layer
# ---------------------------------------------------------------------------

class MLBLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear with MLB quantization.

    See MLBConv2d docstring for QAT vs PTQ semantics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        M: int = 4,
        qat: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.M = M
        self.qat = qat

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias_param = None

        self.register_buffer("_quantized_weight", None)
        self._ptq_applied = False

    def quantize_weights(self) -> Dict:
        """Post-training MLB quantization (same semantics as MLBConv2d)."""
        with torch.no_grad():
            W = self.weight.data
            alphas, bases = mlb_decompose(W, self.M)
            W_mlb = mlb_reconstruct(alphas, bases).to(W.dtype)
            self._quantized_weight = W_mlb
            self._ptq_applied = True
            q_err = quantization_error(W, W_mlb).item()
        return {
            "alphas": [a.item() for a in alphas],
            "bases": bases,
            "quantization_error": q_err,
        }

    def set_qat(self, qat: bool):
        """Toggle QAT mode on or off."""
        self.qat = qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ptq_applied and not self.qat:
            W_eff = self._quantized_weight
        elif self.qat:
            W_eff = mlb_quantize_ste(self.weight, self.M)
        else:
            W_eff = self.weight

        return F.linear(x, W_eff, self.bias_param)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"M={self.M}, qat={self.qat}"
        )


# ---------------------------------------------------------------------------
# Hierarchical M=8 quantizer
# ---------------------------------------------------------------------------

def hierarchical_mlb_decompose(
    W: torch.Tensor,
    M_per_stage: int = 4,
) -> Tuple[Dict, Dict, torch.Tensor]:
    """
    Two-stage hierarchical MLB decomposition.

    Stage 1: M_per_stage-level MLB on W            -> W4, R4
    Stage 2: M_per_stage-level MLB on R4           -> R4_MLB
    Final  : W_hierarchical = W4 + R4_MLB

    Parameters
    ----------
    W            : full-precision weight tensor
    M_per_stage  : number of binary levels per stage (default 4, total=8)

    Returns
    -------
    stage1_info  : dict with 'alphas', 'bases', 'error'
    stage2_info  : dict with 'alphas', 'bases', 'error'
    W_hier       : reconstructed weight tensor
    """
    W_fp = W.detach().clone().float()

    # Stage 1
    alphas1, bases1 = mlb_decompose(W_fp, M_per_stage)
    W4 = mlb_reconstruct(alphas1, bases1)
    R4 = W_fp - W4
    stage1_err = quantization_error(W_fp, W4).item()

    # Stage 2 on residual
    alphas2, bases2 = mlb_decompose(R4, M_per_stage)
    R4_mlb = mlb_reconstruct(alphas2, bases2)
    stage2_err = quantization_error(R4, R4_mlb).item()

    # Combined reconstruction
    W_hier = W4 + R4_mlb
    combined_err = quantization_error(W_fp, W_hier).item()

    stage1_info = {
        "alphas": [a.item() for a in alphas1],
        "bases": bases1,
        "error": stage1_err,
    }
    stage2_info = {
        "alphas": [a.item() for a in alphas2],
        "bases": bases2,
        "error": stage2_err,
    }
    return stage1_info, stage2_info, W_hier.to(W.dtype)


class _HierarchicalMLBQuantizerSTE(torch.autograd.Function):
    """
    STE wrapper for hierarchical M=8 MLB quantization.
    Forward: two-stage M=4+M=4 decomposition.
    Backward: straight-through (identity gradient).
    """

    @staticmethod
    def forward(ctx, W: torch.Tensor, M_per_stage: int) -> torch.Tensor:  # type: ignore[override]
        _, _, W_hier = hierarchical_mlb_decompose(W, M_per_stage)
        return W_hier.to(W.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return grad_output, None


def hierarchical_mlb_quantize_ste(W: torch.Tensor, M_per_stage: int = 4) -> torch.Tensor:
    """
    Quantize W using two-stage hierarchical MLB (total effective M = 2*M_per_stage)
    with STE gradient flow.
    """
    return _HierarchicalMLBQuantizerSTE.apply(W, M_per_stage)


# ---------------------------------------------------------------------------
# Hierarchical MLB Conv2d layer
# ---------------------------------------------------------------------------

class HierarchicalMLBConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d using hierarchical M=8 quantization
    (two consecutive M=4 decompositions).

    Parameters
    ----------
    M_per_stage : int
        Number of binary levels per stage.  Effective total M = 2 * M_per_stage.
        Default is 4 (total effective M = 8).
    qat : bool
        If True, uses STE for gradient flow (QAT).  Otherwise, PTQ after call
        to `quantize_weights()`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        M_per_stage: int = 4,
        qat: bool = False,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.M_per_stage  = M_per_stage
        self.M            = 2 * M_per_stage   # effective total M (for metrics compat)
        self.qat          = qat

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.register_buffer("_quantized_weight", None)
        self._ptq_applied = False

        # Hierarchical stage info populated after PTQ
        self._stage1_info: Optional[Dict] = None
        self._stage2_info: Optional[Dict] = None

    def quantize_weights(self) -> Dict:
        """Post-training hierarchical MLB quantization."""
        with torch.no_grad():
            W = self.weight.data
            s1, s2, W_hier = hierarchical_mlb_decompose(W, self.M_per_stage)
            self._quantized_weight = W_hier
            self._ptq_applied  = True
            self._stage1_info  = s1
            self._stage2_info  = s2
            q_err = quantization_error(W.float(), W_hier.float()).item()
        return {
            "quantization_error": q_err,
            "stage1_error": s1["error"],
            "stage2_error": s2["error"],
        }

    def set_qat(self, qat: bool):
        self.qat = qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ptq_applied and not self.qat:
            W_eff = self._quantized_weight
        elif self.qat:
            W_eff = hierarchical_mlb_quantize_ste(self.weight, self.M_per_stage)
        else:
            W_eff = self.weight
        return F.conv2d(x, W_eff, bias=self.bias_param,
                        stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, M_per_stage={self.M_per_stage} "
            f"(effective_M={self.M}), qat={self.qat}"
        )


# ---------------------------------------------------------------------------
# Hierarchical MLB Linear layer
# ---------------------------------------------------------------------------

class HierarchicalMLBLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using hierarchical M=8 quantization.
    See HierarchicalMLBConv2d for full documentation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        M_per_stage: int = 4,
        qat: bool = False,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.M_per_stage  = M_per_stage
        self.M            = 2 * M_per_stage
        self.qat          = qat

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.register_buffer("_quantized_weight", None)
        self._ptq_applied  = False
        self._stage1_info: Optional[Dict] = None
        self._stage2_info: Optional[Dict] = None

    def quantize_weights(self) -> Dict:
        """Post-training hierarchical MLB quantization."""
        with torch.no_grad():
            W = self.weight.data
            s1, s2, W_hier = hierarchical_mlb_decompose(W, self.M_per_stage)
            self._quantized_weight = W_hier
            self._ptq_applied  = True
            self._stage1_info  = s1
            self._stage2_info  = s2
            q_err = quantization_error(W.float(), W_hier.float()).item()
        return {
            "quantization_error": q_err,
            "stage1_error": s1["error"],
            "stage2_error": s2["error"],
        }

    def set_qat(self, qat: bool):
        self.qat = qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ptq_applied and not self.qat:
            W_eff = self._quantized_weight
        elif self.qat:
            W_eff = hierarchical_mlb_quantize_ste(self.weight, self.M_per_stage)
        else:
            W_eff = self.weight
        return F.linear(x, W_eff, self.bias_param)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"M_per_stage={self.M_per_stage} (effective_M={self.M}), qat={self.qat}"
        )


# ---------------------------------------------------------------------------
# Utility: apply PTQ to all quantized layers in a model
# ---------------------------------------------------------------------------

# Canonical tuple of all quantized layer types -- used throughout the codebase
_ALL_MLB_TYPES = (MLBConv2d, MLBLinear, HierarchicalMLBConv2d, HierarchicalMLBLinear)


def apply_ptq(model: nn.Module) -> Dict[str, Dict]:
    """
    Walk the model and call `quantize_weights()` on every quantized layer
    (MLBConv2d, MLBLinear, HierarchicalMLBConv2d, HierarchicalMLBLinear).

    Returns
    -------
    layer_info : dict  {layer_name: quantize_weights() return value}
    """
    layer_info: Dict[str, Dict] = {}
    for name, module in model.named_modules():
        if isinstance(module, _ALL_MLB_TYPES):
            info = module.quantize_weights()
            layer_info[name] = info
    return layer_info


def set_qat_mode(model: nn.Module, qat: bool):
    """Enable or disable QAT mode on all quantized layers in *model*."""
    for module in model.modules():
        if isinstance(module, _ALL_MLB_TYPES):
            module.set_qat(qat)


def get_layer_M_map(model: nn.Module) -> Dict[str, int]:
    """
    Return a dict mapping layer names to their effective M value.
    Hierarchical layers report M = 2 * M_per_stage.
    """
    return {
        name: module.M
        for name, module in model.named_modules()
        if isinstance(module, _ALL_MLB_TYPES)
    }
