"""
resnet20.py
===========
ResNet-20 for Fashion-MNIST implemented from scratch in PyTorch.

Architecture (He et al., 2016 -- CIFAR variant):
  - Input: (B, 1, 28, 28)   grayscale Fashion-MNIST images
  - conv1: 3×3, 16 filters, stride 1, padding 1
  - layer1: 3 × BasicBlock, 16 channels
  - layer2: 3 × BasicBlock, 32 channels, first block stride 2
  - layer3: 3 × BasicBlock, 64 channels, first block stride 2
  - global average pooling
  - fc: 64 → 10

The model supports three weight implementations:
  1. nn.Conv2d  / nn.Linear  (FP32 baseline)
  2. MLBConv2d / MLBLinear   (uniform M for all layers)
  3. MLBConv2d / MLBLinear   (per-layer M via mixed_precision_config)

mixed_precision_config example
-------------------------------
    {
        "conv1":  5,
        "layer1": 5,
        "layer2": 4,
        "layer3": 3,
        "fc":     2,
    }

Keys map to the *group* names used internally.  Every sublayer inside the
group inherits the group's M value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Callable

from quantization import MLBConv2d, MLBLinear, HierarchicalMLBConv2d, HierarchicalMLBLinear


# ---------------------------------------------------------------------------
# Helper: choose the right Conv / Linear factory given mode + M
# ---------------------------------------------------------------------------

def _make_conv(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = False,
    use_mlb: bool = False,
    M: int = 4,
    qat: bool = False,
    use_hierarchical: bool = False,
    M_per_stage: int = 4,
) -> nn.Module:
    """Return the appropriate Conv2d variant based on mode flags."""
    if use_hierarchical:
        return HierarchicalMLBConv2d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            M_per_stage=M_per_stage,
            qat=qat,
        )
    if use_mlb:
        return MLBConv2d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            M=M,
            qat=qat,
        )
    return nn.Conv2d(
        in_ch, out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )


def _make_linear(
    in_f: int,
    out_f: int,
    bias: bool = True,
    use_mlb: bool = False,
    M: int = 4,
    qat: bool = False,
    use_hierarchical: bool = False,
    M_per_stage: int = 4,
) -> nn.Module:
    """Return the appropriate Linear variant based on mode flags."""
    if use_hierarchical:
        return HierarchicalMLBLinear(in_f, out_f, bias=bias,
                                     M_per_stage=M_per_stage, qat=qat)
    if use_mlb:
        return MLBLinear(in_f, out_f, bias=bias, M=M, qat=qat)
    return nn.Linear(in_f, out_f, bias=bias)


# ---------------------------------------------------------------------------
# Basic Residual Block
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """
    Standard ResNet basic block with two 3×3 convolutions.

    Shortcut connection:
        - Identity shortcut when in_channels == out_channels and stride == 1.
        - 1×1 convolution (projection) otherwise.
    """

    expansion: int = 1  # kept for API compatibility

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_mlb: bool = False,
        M: int = 4,
        qat: bool = False,
        use_hierarchical: bool = False,
        M_per_stage: int = 4,
    ):
        super().__init__()
        self.use_mlb = use_mlb
        self.M = M

        conv_kwargs = dict(
            use_mlb=use_mlb, M=M, qat=qat,
            use_hierarchical=use_hierarchical, M_per_stage=M_per_stage,
        )

        # --- First conv ---
        self.conv1 = _make_conv(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
            **conv_kwargs,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # --- Second conv ---
        self.conv2 = _make_conv(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
            **conv_kwargs,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- Shortcut ---
        self.shortcut = nn.Sequential()  # identity by default
        if stride != 1 or in_channels != out_channels:
            # Projection shortcut (1×1 conv) – not quantized even in MLB
            # modes (standard practice; also keeps compression analysis clean)
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------------
# ResNet-20
# ---------------------------------------------------------------------------

class ResNet20(nn.Module):
    """
    ResNet-20 for Fashion-MNIST (1-channel, 28x28, 10 classes).

    Parameters
    ----------
    use_mlb : bool
        If True, replace Conv2d/Linear layers with MLB equivalents.
    M : int
        Uniform number of binary levels (used when mixed_precision_config
        is None and use_hierarchical is False).
    qat : bool
        If True, enable Quantization-Aware Training (STE gradients).
    mixed_precision_config : dict | None
        Per-group M allocation.  When provided, `M` is ignored.
        Example: {"conv1": 2, "layer1": 3, "layer2": 4, "layer3": 5, "fc": 5}
    use_hierarchical : bool
        If True, replace all quantized layers with HierarchicalMLBConv2d /
        HierarchicalMLBLinear (M_per_stage=4, effective M=8).
    M_per_stage : int
        Binary levels per stage for hierarchical modes (default 4).
    num_classes : int
        Number of output classes (default 10 for Fashion-MNIST).
    """

    def __init__(
        self,
        use_mlb: bool = False,
        M: int = 4,
        qat: bool = False,
        mixed_precision_config: Optional[Dict[str, int]] = None,
        use_hierarchical: bool = False,
        M_per_stage: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        self.use_mlb          = use_mlb
        self.qat              = qat
        self.use_hierarchical = use_hierarchical
        self.M_per_stage      = M_per_stage
        self.num_classes      = num_classes

        # ------------------------------------------------------------------
        # Resolve per-group M values
        # ------------------------------------------------------------------
        def _M(group: str) -> int:
            """Look up M for this group; fall back to uniform M."""
            if mixed_precision_config is not None:
                return mixed_precision_config.get(group, M)
            return M

        self._mixed_precision_config = mixed_precision_config
        self._uniform_M = M

        # Shared kwargs for _make_conv / _make_linear
        def conv_kw(group: str) -> dict:
            return dict(
                use_mlb=use_mlb,
                M=_M(group),
                qat=qat,
                use_hierarchical=use_hierarchical,
                M_per_stage=M_per_stage,
            )

        def lin_kw() -> dict:
            return dict(
                use_mlb=use_mlb,
                M=_M("fc"),
                qat=qat,
                use_hierarchical=use_hierarchical,
                M_per_stage=M_per_stage,
            )

        # ------------------------------------------------------------------
        # Stem convolution  (conv1)
        #   Fashion-MNIST is 28×28; we use stride=1 so we don't lose too
        #   much spatial resolution before the residual blocks.
        # ------------------------------------------------------------------
        self.conv1 = _make_conv(
            in_ch=1, out_ch=16,
            kernel_size=3, stride=1, padding=1, bias=False,
            **conv_kw("conv1"),
        )
        self.bn1 = nn.BatchNorm2d(16)

        # ------------------------------------------------------------------
        # Residual layers
        # ------------------------------------------------------------------
        # layer1 — 3 blocks, 16 channels, no downsampling
        self.layer1 = self._make_layer(
            in_channels=16, out_channels=16,
            blocks=3, stride=1, **conv_kw("layer1"),
        )
        # layer2 — 3 blocks, 32 channels, stride 2 (14×14)
        self.layer2 = self._make_layer(
            in_channels=16, out_channels=32,
            blocks=3, stride=2, **conv_kw("layer2"),
        )
        # layer3 — 3 blocks, 64 channels, stride 2 (7×7)
        self.layer3 = self._make_layer(
            in_channels=32, out_channels=64,
            blocks=3, stride=2, **conv_kw("layer3"),
        )

        # ------------------------------------------------------------------
        # Classifier head
        # ------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # → (B, 64, 1, 1)
        self.fc = _make_linear(
            in_f=64, out_f=num_classes, bias=True,
            **lin_kw(),
        )

        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        self._initialize_weights()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        use_mlb: bool,
        M: int,
        qat: bool,
        use_hierarchical: bool = False,
        M_per_stage: int = 4,
    ) -> nn.Sequential:
        """Stack `blocks` BasicBlocks; first block uses `stride`."""
        layers = []
        layers.append(
            BasicBlock(
                in_channels, out_channels,
                stride=stride,
                use_mlb=use_mlb, M=M, qat=qat,
                use_hierarchical=use_hierarchical,
                M_per_stage=M_per_stage,
            )
        )
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    out_channels, out_channels,
                    stride=1,
                    use_mlb=use_mlb, M=M, qat=qat,
                    use_hierarchical=use_hierarchical,
                    M_per_stage=M_per_stage,
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Kaiming-normal init for Conv2d and nn.Linear."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, MLBConv2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, MLBLinear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias_param is not None if hasattr(m, "bias_param") else m.bias is not None:
                    bias_attr = m.bias_param if hasattr(m, "bias_param") else m.bias
                    if bias_attr is not None:
                        nn.init.zeros_(bias_attr)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # Residual stages
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Head
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_mixed_precision_config(self) -> Optional[Dict[str, int]]:
        """Return the mixed-precision config used at construction."""
        return self._mixed_precision_config

    def model_size_mb(self) -> float:
        """Model size in MB (FP32 parameters only)."""
        total = sum(p.numel() for p in self.parameters())
        return total * 4 / (1024 ** 2)  # 4 bytes per float32


# ---------------------------------------------------------------------------
# Convenience factory functions (mirrors config.py Mode enum)
# ---------------------------------------------------------------------------

def build_fp32_resnet20(num_classes: int = 10) -> ResNet20:
    """Mode A -- plain FP32 ResNet-20."""
    return ResNet20(use_mlb=False, num_classes=num_classes)


def build_uniform_mlb_resnet20(
    M: int = 4,
    qat: bool = False,
    num_classes: int = 10,
) -> ResNet20:
    """
    Mode B (qat=False) or Mode C (qat=True) -- uniform MLB ResNet-20.
    """
    return ResNet20(use_mlb=True, M=M, qat=qat, num_classes=num_classes)


def build_mixed_precision_resnet20(
    mixed_precision_config: Dict[str, int],
    qat: bool = False,
    num_classes: int = 10,
) -> ResNet20:
    """
    Mode D (qat=False) or Mode E (qat=True) -- mixed-precision MLB ResNet-20.
    """
    return ResNet20(
        use_mlb=True,
        qat=qat,
        mixed_precision_config=mixed_precision_config,
        num_classes=num_classes,
    )


def build_hierarchical_resnet20(
    M_per_stage: int = 4,
    qat: bool = False,
    num_classes: int = 10,
) -> ResNet20:
    """
    Mode F (qat=False) or Mode G (qat=True) -- hierarchical M=8 MLB ResNet-20.

    Uses two consecutive M=4 decompositions per layer for an effective
    total of 8 binary bases per weight tensor.

    Parameters
    ----------
    M_per_stage : int
        Binary levels per stage (default 4, so total effective M = 8).
    qat         : bool
        Enable quantization-aware training.
    """
    return ResNet20(
        use_mlb=False,            # standard MLB layers not used
        use_hierarchical=True,    # hierarchical layers are used
        M_per_stage=M_per_stage,
        qat=qat,
        num_classes=num_classes,
    )
