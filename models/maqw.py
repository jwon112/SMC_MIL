"""
M-AQW: Meta-Parametric Asymmetric Quality-Aware Weighting.
Per-patch weights from Laplacian (quality) via double-sigmoid; parameters predicted by Meta-MLP from slide-level stats + histogram.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_q(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-slide normalize q to [0, 1]. q: [N]."""
    q_min = q.min()
    q_max = q.max()
    span = q_max - q_min + eps
    return (q - q_min) / span


def _slide_stats_and_histogram(q_norm: torch.Tensor, n_bins: int = 10, eps: float = 1e-8) -> torch.Tensor:
    """
    From q_norm [N] build 16-dim input: 6 stats + 10-bin histogram (density = count_b / N).
    Returns [16] on same device as q_norm.
    """
    n = q_norm.numel()
    device = q_norm.device
    q_flat = q_norm.view(-1)

    # Statistics (6): mean, std, min, max, p25, p75
    mean_q = q_flat.mean()
    std_q = q_flat.std() + eps
    min_q = q_flat.min()
    max_q = q_flat.max()
    p25 = torch.quantile(q_flat.float(), 0.25)
    p75 = torch.quantile(q_flat.float(), 0.75)
    stats = torch.stack([mean_q, std_q, min_q, max_q, p25, p75])

    # Histogram: 10 bins over [0, 1]; density = count_b / N
    hist = torch.histc(q_flat, bins=n_bins, min=0.0, max=1.0)
    if n > 0:
        hist = hist / n  # density: count_b / N
    else:
        hist = torch.zeros(n_bins, device=device, dtype=q_flat.dtype)

    x = torch.cat([stats, hist], dim=0)
    return x  # [6 + 10] = [16]


class M_AQW(nn.Module):
    """
    Meta-Parametric Asymmetric Quality-Aware Weighting.
    Input: h [N, D], q [N] (raw Laplacian scores).
    Output: h_mod [N, D] = h * W(q), with W from double-sigmoid parameterized by Meta-MLP.
    """

    def __init__(self, meta_input_dim: int = 16, meta_hidden: int = 32):
        super().__init__()
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_input_dim, meta_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(meta_hidden, 4),
        )

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        h: [N, D], q: [N].
        Returns h_out [N, D] = h * W(q).unsqueeze(1).
        """
        if q is None or q.numel() == 0:
            return h
        q_norm = _normalize_q(q)  # [N]
        x = _slide_stats_and_histogram(q_norm).unsqueeze(0)  # [1, 16]
        raw = self.meta_mlp(x).squeeze(0)  # [4] -> (tau_L, k_L, tau_R, k_R)
        tau_L = torch.sigmoid(raw[0])
        k_L = F.softplus(raw[1]) + 0.1
        tau_R = torch.sigmoid(raw[2])
        k_R = F.softplus(raw[3]) + 0.1

        # W_left(q) = sigmoid(k_L * (q_norm - tau_L))
        w_left = torch.sigmoid(k_L * (q_norm - tau_L))
        # W_right(q) = sigmoid(k_R * (tau_R - q_norm))
        w_right = torch.sigmoid(k_R * (tau_R - q_norm))
        w = w_left * w_right  # [N]
        return h * w.unsqueeze(1)
