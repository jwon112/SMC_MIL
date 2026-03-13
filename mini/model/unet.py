from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from mini.module.blocks import Down, SegHead, UNetSpec, Up, build_stage


class UNet(nn.Module):
    """
    Minimal U-Net skeleton intended for module swapping.
    - Replace `DoubleConv` / `Down` / `Up` / `SegHead` with your own modules.
    - Keep the forward shape contract: BxCxHxW -> BxKxHxW.
    """

    def __init__(self, spec: UNetSpec):
        super().__init__()
        encoder_type = getattr(spec, "encoder_type", "plain") or "plain"

        if encoder_type == "plain":
            if spec.depth < 2:
                raise ValueError("depth must be >= 2")

            chs: List[int] = [spec.base_channels * (2**i) for i in range(spec.depth)]

            self.encoder = None
            self.stem = build_stage(spec, spec.in_channels, chs[0])
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

            for i, down in enumerate(self.downs):
                down.conv = build_stage(spec, chs[i], chs[i + 1])
            for i, up in enumerate(self.ups):
                # Up concatenates skip + upsampled => 2*out_ch input to stage
                out_ch = chs[len(chs) - 2 - i]
                up.conv = build_stage(spec, out_ch * 2, out_ch)

            self.head = SegHead(chs[0], spec.num_classes)

        elif encoder_type == "convnext_tiny":
            # Use timm ConvNeXt-Tiny encoder with ImageNet pretraining.
            import timm

            self.encoder = timm.create_model(
                "convnext_tiny",
                pretrained=bool(getattr(spec, "encoder_pretrained", False)),
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
            chs_enc: List[int] = list(self.encoder.feature_info.channels())  # [c1, c2, c3, c4]
            if len(chs_enc) != 4:
                raise ValueError(f"Expected 4 feature stages from convnext_tiny, got {len(chs_enc)}")

            c1, c2, c3, c4 = chs_enc

            # Decoder mirrors the encoder stages: c4 -> c3 -> c2 -> c1
            self.downs = None  # encoder handles downsampling
            self.stem = None

            self.ups = nn.ModuleList(
                [
                    Up(c4, c3, norm=spec.norm, act=spec.act, dropout=spec.dropout, up=spec.up),
                    Up(c3, c2, norm=spec.norm, act=spec.act, dropout=spec.dropout, up=spec.up),
                    Up(c2, c1, norm=spec.norm, act=spec.act, dropout=spec.dropout, up=spec.up),
                ]
            )

            self.head = SegHead(c1, spec.num_classes)

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # plain UNet encoder
        if self.encoder is None:
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

        # ConvNeXt-Tiny encoder path
        feats = self.encoder(x)
        if not isinstance(feats, (list, tuple)) or len(feats) != 4:
            raise RuntimeError("ConvNeXt encoder did not return 4 feature maps as expected")

        f1, f2, f3, f4 = feats  # low -> high level
        x = f4
        skips = [f3, f2, f1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        return self.head(x)

