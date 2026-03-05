"""
Feature-space DDPM for denoising patch features. Identity-on-clean is supported via L_id.
Schedule and forward noising are included; reverse process yields denoised h_0.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def get_ddpm_schedule(T: int, beta_min: float = 1e-4, beta_max: float = 0.02, device=None):
    """
    Linear beta schedule. Returns tensors of shape [T+1] (index 0 unused for t=1..T).
    """
    betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)  # [T]
    # prepend 1.0 for t=0 so alpha_bar[0] = 1, alpha_bar[1] = alpha_1, ...
    alpha_bar = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar], dim=0)  # [T+1]
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
    if device is not None:
        betas = betas.to(device)
        alphas = alphas.to(device)
        alpha_bar = alpha_bar.to(device)
        sqrt_alpha_bar = sqrt_alpha_bar.to(device)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.to(device)
    return {
        "betas": betas.float(),
        "alphas": alphas.float(),
        "alpha_bar": alpha_bar.float(),
        "sqrt_alpha_bar": sqrt_alpha_bar.float(),
        "sqrt_one_minus_alpha_bar": sqrt_one_minus_alpha_bar.float(),
        "T": T,
    }


def noisify(h_clean: torch.Tensor, t: torch.Tensor, schedule: dict, rng=None) -> tuple:
    """
    h_clean: [B, D], t: [B] long in 0..T-1 (0 = clean, 1..T = noised).
    Returns (h_t [B, D], epsilon [B, D]).
    For t=0, h_t = h_clean and epsilon = 0 (no noise added).
    """
    B, D = h_clean.shape
    device = h_clean.device
    sqrt_ab = schedule["sqrt_alpha_bar"]   # [T+1]
    sqrt_omb = schedule["sqrt_one_minus_alpha_bar"]

    # t is 0-indexed; schedule index: t=0 -> sqrt_alpha_bar[0]=1, sqrt_omb[0]=0
    # t in [0, T-1] -> indices in [0, T-1]; for t=0 we want no noise
    idx = t.clamp(0, schedule["T"] - 1)
    s_ab = sqrt_ab[idx].to(device).unsqueeze(1)   # [B, 1]
    s_omb = sqrt_omb[idx].to(device).unsqueeze(1)

    if rng is None:
        rng = torch.Generator(device=device)
    epsilon = torch.randn(B, D, device=device, dtype=h_clean.dtype, generator=rng)
    # t=0: s_ab=1, s_omb=0 -> h_t = h_clean
    h_t = s_ab * h_clean + s_omb * epsilon
    # For t=0, return epsilon=0 so target is 0 for L_id
    epsilon_zero = torch.where(t.unsqueeze(1) == 0, torch.zeros_like(epsilon), epsilon)
    return h_t, epsilon_zero


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t [B] long -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ---------------------------------------------------------------------------
# Noise predictor and denoiser module
# ---------------------------------------------------------------------------

class EpsilonPredictor(nn.Module):
    """Predicts epsilon from (h_t, t). Input h_t [B, D], t [B] -> output [B, D]."""

    def __init__(self, D: int, time_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.D = D
        self.time_dim = time_dim
        self.proj_in = nn.Linear(D + time_dim, hidden_dim)
        self.proj_h = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Linear(hidden_dim, D)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.zeros_(self.proj_in.bias)
        for m in self.proj_h:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, h_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: [B] long, 0..T-1
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        x = torch.cat([h_t, t_emb], dim=1)
        x = F.relu(self.proj_in(x))
        x = x + self.proj_h(x)
        return self.proj_out(x)


class DDPMFeatureDenoiser(nn.Module):
    """
    Wraps schedule + epsilon predictor. Provides denoise(h, t_start, num_steps).
    """

    def __init__(self, embed_dim: int, T: int = 1000, time_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.T = T
        schedule = get_ddpm_schedule(T, device=None)
        self.eps_theta = EpsilonPredictor(embed_dim, time_dim=time_dim, hidden_dim=hidden_dim)
        for k, v in schedule.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer("schedule_" + k, v)
        self._schedule_keys = [k for k in schedule if isinstance(schedule[k], torch.Tensor)]

    def _get_schedule(self, device):
        return {k: getattr(self, "schedule_" + k) for k in self._schedule_keys}

    def predict_epsilon(self, h_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.eps_theta(h_t, t)

    def reverse_step(self, h_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """
        Single DDPM reverse step: h_t, t (0-indexed) -> h_{t-1}.
        Posterior mean: (1/sqrt(alpha_t)) * (h_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_pred).
        """
        device = h_t.device
        schedule = self._get_schedule(device)
        T = self.T
        alpha_bar = schedule["alpha_bar"]
        sqrt_one_minus_alpha_bar = schedule["sqrt_one_minus_alpha_bar"]
        betas = schedule["betas"]
        alphas = schedule["alphas"]

        t_idx = t.clamp(0, T - 1)
        ab_t = alpha_bar[t_idx + 1].to(device).unsqueeze(1)
        sqrt_omb_t = sqrt_one_minus_alpha_bar[t_idx + 1].to(device).unsqueeze(1)
        beta_t = betas[t_idx].to(device).unsqueeze(1)
        alpha_t = alphas[t_idx].to(device).unsqueeze(1)

        mean = (1.0 / (torch.sqrt(alpha_t) + 1e-8)) * (h_t - (beta_t / (sqrt_omb_t + 1e-8)) * eps_pred)
        return mean

    def denoise(self, h: torch.Tensor, t_start: int = 20, num_steps: int = None) -> torch.Tensor:
        """
        Run reverse process from t_start down to 0. h [N, D] is treated as h_{t_start}.
        Returns h_0 [N, D]. t_start and steps use 1-indexed logic (t_start=20 -> 20 steps down).
        """
        if num_steps is None:
            num_steps = t_start
        num_steps = min(num_steps, t_start)
        device = h.device
        N, D = h.shape
        assert D == self.embed_dim
        h_cur = h
        for step in range(num_steps):
            # Current noised level (1-indexed): t_start - step. 0-indexed: t_start - step - 1
            t_1idx = t_start - step
            if t_1idx <= 0:
                break
            t_0idx = t_1idx - 1
            t_batch = torch.full((N,), t_0idx, device=device, dtype=torch.long)
            eps = self.eps_theta(h_cur, t_batch)
            h_cur = self.reverse_step(h_cur, t_batch, eps)
        return h_cur
