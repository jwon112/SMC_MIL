from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mini.module.blocks import norm_layer


def _odd_kernel(k: int) -> int:
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"kernel_size must be positive odd, got {k}")
    return k


class KernelConvStage(nn.Module):
    """
    Simple stage for controlled experiments:
    - normal conv: Conv(kxk) -> Norm -> Act  (x2)
    - depthwise-separable: DWConv(kxk) -> PW(1x1) -> Norm -> Act (x2)

    Intended as a drop-in replacement for DoubleConv inside U-Net stages.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        depthwise_separable: bool = False,
        norm: str = "bn",
        act: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        k = _odd_kernel(kernel_size)
        p = k // 2
        self.depthwise_separable = bool(depthwise_separable)

        act = (act or "relu").lower()
        if act == "relu":
            activation = nn.ReLU(inplace=True)
        elif act in {"silu", "swish"}:
            activation = nn.SiLU(inplace=True)
        elif act == "gelu":
            activation = nn.GELU()
        elif act in {"leaky_relu", "lrelu"}:
            activation = nn.LeakyReLU(0.1, inplace=True)
        elif act in {"none", "identity", ""}:
            activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {act}")

        drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

        def _normal(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=k, padding=p, bias=(norm in {"none", "identity", ""})),
                norm_layer(norm, out_c),
                activation,
                drop,
            )

        def _dwsep(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=k, padding=p, groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=(norm in {"none", "identity", ""})),
                norm_layer(norm, out_c),
                activation,
                drop,
            )

        block = _dwsep if self.depthwise_separable else _normal
        self.net = nn.Sequential(
            block(in_ch, out_ch),
            block(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

