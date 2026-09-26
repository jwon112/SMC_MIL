from __future__ import annotations

import torch.nn as nn


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

