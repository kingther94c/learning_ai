"""Residual learning in its smallest algebraic and convolutional forms.

``ResidualLinearStack`` is a controlled optimisation microscope: it composes
``x + W_i x`` so the identity path is impossible to miss. ``BasicResidualBlock``
and ``TinyResNet`` then implement the two-3x3-convolution block from the original
ResNet paper at laptop scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainLinearStack(nn.Module):
    """Compose equally sized linear maps: ``W_L ... W_2 W_1 x``."""

    def __init__(self, dimension: int, depth: int, *, init_scale: float = 0.05):
        super().__init__()
        _validate_linear_stack(dimension, depth, init_scale)
        self.layers = nn.ModuleList(
            [nn.Linear(dimension, dimension, bias=False) for _ in range(depth)]
        )
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualLinearStack(nn.Module):
    """Compose residual maps: ``(I + W_L) ... (I + W_1) x``."""

    def __init__(self, dimension: int, depth: int, *, init_scale: float = 0.05):
        super().__init__()
        _validate_linear_stack(dimension, depth, init_scale)
        self.layers = nn.ModuleList(
            [nn.Linear(dimension, dimension, bias=False) for _ in range(depth)]
        )
        self.reset_parameters(init_scale)

    def reset_parameters(self, init_scale: float) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


def _validate_linear_stack(dimension: int, depth: int, init_scale: float) -> None:
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if init_scale <= 0:
        raise ValueError(f"init_scale must be positive, got {init_scale}")


class BasicResidualBlock(nn.Module):
    """The original ResNet basic block for 2D feature maps.

    Main path: ``3x3 conv -> BN -> ReLU -> 3x3 conv -> BN``.
    Shortcut: identity when shape permits, otherwise a 1x1 projection.
    A final ReLU follows the elementwise addition, as in the 2015 paper.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.bn1(self.conv1(x))
        residual = F.relu(residual, inplace=False)
        residual = self.bn2(self.conv2(residual))
        return F.relu(residual + self.shortcut(x), inplace=False)


class TinyResNet(nn.Module):
    """A small-image ResNet built from :class:`BasicResidualBlock`.

    The 3x3 stem is appropriate for CIFAR-sized inputs.  Each later stage halves
    spatial resolution and doubles width; global average pooling makes the
    classifier independent of the input image size.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        widths: Sequence[int] = (16, 32, 64),
        in_channels: int = 3,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if len(blocks_per_stage) == 0 or len(blocks_per_stage) != len(widths):
            raise ValueError("blocks_per_stage and widths must have the same non-zero length")
        if any(blocks <= 0 for blocks in blocks_per_stage) or any(width <= 0 for width in widths):
            raise ValueError("every stage must have positive block and width counts")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=False),
        )

        stages: list[nn.Module] = []
        current_width = widths[0]
        for stage_index, (block_count, width) in enumerate(
            zip(blocks_per_stage, widths, strict=True)
        ):
            stride = 1 if stage_index == 0 else 2
            stages.append(BasicResidualBlock(current_width, width, stride=stride))
            stages.extend(BasicResidualBlock(width, width) for _ in range(block_count - 1))
            current_width = width
        self.stages = nn.Sequential(*stages)
        self.classifier = nn.Linear(current_width, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Blocks initialise themselves; initialise the stem and classifier here.
        stem_conv = self.stem[0]
        stem_bn = self.stem[1]
        nn.init.kaiming_normal_(stem_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.ones_(stem_bn.weight)
        nn.init.zeros_(stem_bn.bias)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return self.classifier(x)
