from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for NCHW tensors.
    Matches the official torchvision ConvNeXt implementation behavior.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N,C,H,W
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlockV1(nn.Module):
    """
    Official-style ConvNeXt v1 block (torchvision-like):
    DWConv 7x7 -> LayerNorm -> Linear(4C) -> GELU -> Linear(C) -> LayerScale -> StochasticDepth -> residual
    """

    def __init__(
        self,
        dim: int,
        *,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value and layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)))
        else:
            self.gamma = None

        self.drop_path = StochasticDepth(drop_path, mode="row") if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # N H W C
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # N C H W
        x = shortcut + self.drop_path(x)
        return x


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt v2).
    Operates in channels-last (NHWC) convention.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: N H W C
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # N 1 1 C
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtBlockV2(nn.Module):
    """
    ConvNeXt v2-style block core idea:
    DWConv 7x7 -> LayerNorm -> Linear(4C) -> GELU -> GRN -> Linear(C) -> StochasticDepth -> residual

    Note: v2 typically removes LayerScale; GRN is the key addition.
    """

    def __init__(self, dim: int, *, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = StochasticDepth(drop_path, mode="row") if drop_path and drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # N H W C
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # N C H W
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """
    Channel-changing adapter + repeated ConvNeXt blocks.
    Intended as a drop-in replacement for DoubleConv inside U-Net stages.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        variant: str = "v1",
        num_blocks: int = 2,
        layer_scale_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        variant = (variant or "v1").lower()
        if variant in {"v1", "convnext", "convnextv1"}:
            blocks = [
                ConvNeXtBlockV1(out_ch, layer_scale_init_value=layer_scale_init_value, drop_path=drop_path)
                for _ in range(int(num_blocks))
            ]
        elif variant in {"v2", "convnextv2"}:
            blocks = [ConvNeXtBlockV2(out_ch, drop_path=drop_path) for _ in range(int(num_blocks))]
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)

