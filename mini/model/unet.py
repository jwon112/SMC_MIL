from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from mini.module.blocks import DoubleConv, Down, SegHead, UNetSpec, Up


class UNet(nn.Module):
    """
    Minimal U-Net skeleton intended for module swapping.
    - Replace `DoubleConv` / `Down` / `Up` / `SegHead` with your own modules.
    - Keep the forward shape contract: BxCxHxW -> BxKxHxW.
    """

    def __init__(self, spec: UNetSpec):
        super().__init__()
        if spec.depth < 2:
            raise ValueError("depth must be >= 2")

        chs: List[int] = [spec.base_channels * (2**i) for i in range(spec.depth)]

        self.stem = DoubleConv(spec.in_channels, chs[0], norm=spec.norm, act=spec.act, dropout=spec.dropout)
        self.downs = nn.ModuleList(
            [
                Down(
                    chs[i],
                    chs[i + 1],
                    norm=spec.norm,
                    act=spec.act,
                    dropout=spec.dropout,
                    pool=spec.pool,
                )
                for i in range(spec.depth - 1)
            ]
        )

        self.ups = nn.ModuleList(
            [
                Up(
                    chs[i + 1],
                    chs[i],
                    norm=spec.norm,
                    act=spec.act,
                    dropout=spec.dropout,
                    up=spec.up,
                )
                for i in reversed(range(spec.depth - 1))
            ]
        )

        self.head = SegHead(chs[0], spec.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []

        x = self.stem(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips.pop()  # deepest
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        return self.head(x)

