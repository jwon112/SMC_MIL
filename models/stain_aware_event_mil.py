"""Hierarchical stain-aware MIL for biopsy-event classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


STAIN_GROUPS = ("HE", "IHC", "other")
AGNOSTIC_GROUP = "all"


class GatedAttentionPool(nn.Module):
    """Pool a variable-length sequence with gated attention."""

    def __init__(self, dimension: int, attention_dimension: int, dropout: float) -> None:
        super().__init__()
        self.attention_a = nn.Sequential(nn.Linear(dimension, attention_dimension), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(dimension, attention_dimension), nn.Sigmoid())
        self.attention_c = nn.Linear(attention_dimension, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or not len(features):
            raise ValueError("Expected a non-empty [instances, features] tensor")
        attention = self.attention_c(self.dropout(self.attention_a(features) * self.attention_b(features)))
        weights = F.softmax(attention.squeeze(1), dim=0)
        return torch.sum(weights.unsqueeze(1) * features, dim=0), weights


class StainAwareEventMIL(nn.Module):
    """Patch-to-slide-to-event MIL with H&E, IHC, and other-stain branches.

    Slide patch encoders are shared to preserve statistical efficiency. Each
    stain group has its own slide projection and event-level slide attention.
    Missing groups are represented by zero embeddings plus explicit masks.
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        n_classes: int = 2,
        use_stain_branches: bool = True,
        include_presence_masks: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.use_stain_branches = use_stain_branches
        self.include_presence_masks = include_presence_masks
        self.branch_groups = STAIN_GROUPS if use_stain_branches else (AGNOSTIC_GROUP,)
        self.patch_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.patch_pool = GatedAttentionPool(hidden_dim, max(hidden_dim // 2, 16), dropout)
        self.branch_projection = nn.ModuleDict({
            group: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            for group in self.branch_groups
        })
        self.branch_pool = nn.ModuleDict({
            group: GatedAttentionPool(hidden_dim, max(hidden_dim // 2, 16), dropout)
            for group in self.branch_groups
        })
        classifier_dimension = hidden_dim * len(self.branch_groups)
        if include_presence_masks:
            classifier_dimension += len(self.branch_groups)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dimension, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, slides: list[tuple[str, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, object]]:
        grouped: dict[str, list[torch.Tensor]] = {group: [] for group in self.branch_groups}
        patch_attention: list[torch.Tensor] = []
        for stain_group, patch_features in slides:
            if self.use_stain_branches and stain_group not in grouped:
                raise ValueError(f"Unsupported stain group: {stain_group}")
            branch_group = stain_group if self.use_stain_branches else AGNOSTIC_GROUP
            encoded = self.patch_encoder(patch_features)
            slide_embedding, weights = self.patch_pool(encoded)
            grouped[branch_group].append(self.branch_projection[branch_group](slide_embedding))
            patch_attention.append(weights)

        branch_embeddings: list[torch.Tensor] = []
        branch_masks: list[torch.Tensor] = []
        slide_attention: dict[str, torch.Tensor | None] = {}
        reference = next(self.parameters())
        for group in self.branch_groups:
            if grouped[group]:
                stacked = torch.stack(grouped[group], dim=0)
                embedding, weights = self.branch_pool[group](stacked)
                branch_embeddings.append(embedding)
                branch_masks.append(torch.ones(1, device=embedding.device))
                slide_attention[group] = weights
            else:
                branch_embeddings.append(torch.zeros(self.hidden_dim, device=reference.device))
                branch_masks.append(torch.zeros(1, device=reference.device))
                slide_attention[group] = None

        components = [*branch_embeddings]
        if self.include_presence_masks:
            components.extend(branch_masks)
        event_features = torch.cat(components, dim=0).unsqueeze(0)
        logits = self.classifier(event_features)
        return logits, {"patch_attention": patch_attention, "slide_attention": slide_attention}
