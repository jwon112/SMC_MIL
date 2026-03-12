from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini.module.convnext import ConvNeXtStage
from mini.module.kernelconv import KernelConvStage


def norm_layer(kind: str, num_channels: int) -> nn.Module:
    kind = (kind or "bn").lower()
    if kind in {"bn", "batchnorm", "batch_norm"}:
        return nn.BatchNorm2d(num_channels)
    if kind in {"gn", "groupnorm", "group_norm"}:
        groups = 32
        groups = min(groups, num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    if kind in {"in", "instancenorm", "instance_norm"}:
        return nn.InstanceNorm2d(num_channels, affine=True)
    if kind in {"ln", "layernorm", "layer_norm"}:
        return nn.GroupNorm(1, num_channels)
    if kind in {"none", "identity", ""}:
        return nn.Identity()
    raise ValueError(f"Unknown norm kind: {kind}")


class ConvActNorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        norm: str = "bn",
        act: str = "relu",
        bias: Optional[bool] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        if bias is None:
            bias = norm in {"none", "identity", ""}

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.norm = norm_layer(norm, out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        act = (act or "relu").lower()
        if act in {"relu"}:
            self.act = nn.ReLU(inplace=True)
        elif act in {"silu", "swish"}:
            self.act = nn.SiLU(inplace=True)
        elif act in {"gelu"}:
            self.act = nn.GELU()
        elif act in {"leaky_relu", "lrelu"}:
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act in {"none", "identity", ""}:
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {act}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        mid_ch: Optional[int] = None,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.block = nn.Sequential(
            ConvActNorm(in_ch, mid_ch, norm=norm, act=act, dropout=dropout),
            ConvActNorm(mid_ch, out_ch, norm=norm, act=act, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        pool: str = "max",
    ):
        super().__init__()
        pool = (pool or "max").lower()
        if pool in {"max", "maxpool"}:
            self.pool = nn.MaxPool2d(2)
        elif pool in {"avg", "avgpool"}:
            self.pool = nn.AvgPool2d(2)
        elif pool in {"conv"}:
            self.pool = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, groups=in_ch, bias=False)
        else:
            raise ValueError(f"Unknown pool kind: {pool}")

        self.conv = DoubleConv(in_ch, out_ch, norm=norm, act=act, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
        up: str = "bilinear",
    ):
        super().__init__()
        up = (up or "bilinear").lower()
        if up in {"bilinear", "nearest"}:
            mode = "bilinear" if up == "bilinear" else "nearest"
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=False if mode == "bilinear" else None)
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            conv_in = out_ch * 2
        elif up in {"deconv", "transposed", "transpose"}:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
            conv_in = out_ch * 2
        else:
            raise ValueError(f"Unknown up kind: {up}")

        self.conv = DoubleConv(conv_in, out_ch, norm=norm, act=act, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.reduce(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SegHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass(frozen=True)
class UNetSpec:
    in_channels: int = 3
    num_classes: int = 21
    base_channels: int = 64
    depth: int = 4
    norm: str = "bn"
    act: str = "relu"
    dropout: float = 0.0
    pool: str = "max"
    up: str = "bilinear"

    # block selection
    block: str = "conv"  # conv | convnext | convnextv2
    convnext_num_blocks: int = 2
    convnext_layer_scale: float = 1e-6
    convnext_drop_path: float = 0.0

    # kernel sweep blocks
    kernel_size: int = 3


def build_stage(spec: UNetSpec, in_ch: int, out_ch: int) -> nn.Module:
    """
    Registry/redirect for U-Net stages.
    This is the indirection layer so new blocks can live in separate files.
    """
    block = (spec.block or "conv").lower()
    if block == "conv":
        return DoubleConv(in_ch, out_ch, norm=spec.norm, act=spec.act, dropout=spec.dropout)
    if block in {"convnext", "convnextv1"}:
        return ConvNeXtStage(
            in_ch,
            out_ch,
            variant="v1",
            num_blocks=spec.convnext_num_blocks,
            layer_scale_init_value=spec.convnext_layer_scale,
            drop_path=spec.convnext_drop_path,
        )
    if block in {"convnextv2", "v2"}:
        return ConvNeXtStage(
            in_ch,
            out_ch,
            variant="v2",
            num_blocks=spec.convnext_num_blocks,
            layer_scale_init_value=0.0,
            drop_path=spec.convnext_drop_path,
        )
    if block in {"kconv", "kernelconv"}:
        return KernelConvStage(
            in_ch,
            out_ch,
            kernel_size=spec.kernel_size,
            depthwise_separable=False,
            norm=spec.norm,
            act=spec.act,
            dropout=spec.dropout,
        )
    if block in {"kdwsep", "kdw", "dwsep", "kernel_dwsep"}:
        return KernelConvStage(
            in_ch,
            out_ch,
            kernel_size=spec.kernel_size,
            depthwise_separable=True,
            norm=spec.norm,
            act=spec.act,
            dropout=spec.dropout,
        )
    raise ValueError(f"Unknown block: {spec.block}")

