import argparse
import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from artidiffuser.dataloader import SynthEvalDataset
from artidiffuser.train_coad_ddpm import SinusoidalTimeEmbedding, make_beta_schedule


class CondUNet(nn.Module):
    """
    조건부 UNet (x_t + cond(inpainted)) 를 채널 concat으로 입력.
    입력: x_t [B,3,H,W], cond [B,3,H,W], t [B]
    출력: eps_hat [B,3,H,W]
    """

    def __init__(self, time_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim, max_steps=1000),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
        )

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
            )

        # x_t(3) + cond(3) = 6
        self.down1 = conv_block(6, base_channels)
        self.down2 = conv_block(base_channels, base_channels * 2)
        self.down3 = conv_block(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)
        self.mid = conv_block(base_channels * 4, base_channels * 4)

        # concat 구조 기준
        self.up3 = conv_block(base_channels * 6, base_channels * 2)
        self.up2 = conv_block(base_channels * 3, base_channels)
        self.up1 = conv_block(base_channels * 2, base_channels)

        self.time_proj = nn.Linear(time_dim, base_channels * 4)
        self.final = nn.Conv2d(base_channels, 3, 1)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        x = torch.cat([x_t, cond], dim=1)  # [B,6,H,W]

        t_emb = self.time_mlp(t)
        t_proj = self.time_proj(t_emb)[:, :, None, None]

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))

        mid = self.mid(d3 + t_proj)

        u3 = F.interpolate(mid, scale_factor=2, mode="nearest")
        u3 = self.up3(torch.cat([u3, d2], dim=1))

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, d1], dim=1))

        out = self.up1(torch.cat([u2, d1], dim=1))
        return self.final(out)


def prepare_diffusion(T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def q_sample(x0: torch.Tensor, t: torch.LongTensor, alphas_cumprod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise
    return x_t, noise


def _stable_hash_int(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def split_indices_by_hash(paths: list[str], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """
    파일 경로 문자열을 해시로 나눠 고정 split.
    """
    assert 0.0 < val_ratio < 1.0
    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, p in enumerate(paths):
        x = _stable_hash_int(f"{seed}:{p}") % 10_000
        if x < int(val_ratio * 10_000):
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx


@dataclass
class TrainConfig:
    timesteps: int = 1000
    lr: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 50
    device: str = "cuda"
    eval_every: int = 1
    eval_t_start: int = 100
    eval_max_samples: int = 200
    val_ratio: float = 0.1
    seed: int = 42


def _compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


def _compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return float("nan")
    vals = []
    for i in range(pred.shape[0]):
        vals.append(ssim(pred[i], gt[i], channel_axis=2, data_range=1.0))
    return float(np.mean(vals))


def _to_01(x: torch.Tensor) -> np.ndarray:
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def p_sample_step(
    x_t: torch.Tensor,
    cond: torch.Tensor,
    t: int,
    model: nn.Module,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    B = x_t.size(0)
    t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
    eps = model(x_t, cond, t_tensor)

    beta_t = betas[t]
    alpha_t = 1.0 - beta_t
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)

    coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    x_prev = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps)

    if t > 1:
        sigma2_t = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
        x_prev = x_prev + torch.sqrt(sigma2_t) * torch.randn_like(x_t)
    return x_prev


def restore_conditional(
    cond: torch.Tensor,
    model: nn.Module,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    t_start: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    conditional generation: x_T ~ N(0,1)에서 시작하지 않고,
    cond를 기반으로 빠른 비교를 위해 x_{t_start}를 N(0,1)로 시작.
    """
    if seed is not None:
        torch.manual_seed(seed)
    x_t = torch.randn_like(cond, device=device)
    for t in range(t_start, 0, -1):
        x_t = p_sample_step(x_t, cond, t, model, betas, alphas_cumprod, device)
    return x_t


def eval_on_val(
    model: nn.Module,
    loader: DataLoader,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    t_start: int,
    max_samples: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    n_done = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val Synth", leave=False):
            cond = batch["inpainted"].to(device)
            gt = batch["ori"].to(device)

            pred = restore_conditional(
                cond=cond,
                model=model,
                betas=betas,
                alphas_cumprod=alphas_cumprod,
                t_start=t_start,
                device=device,
                seed=seed,
            )

            pred_np = _to_01(pred)
            gt_np = _to_01(gt)
            for i in range(pred_np.shape[0]):
                psnr_list.append(_compute_psnr(pred_np[i : i + 1], gt_np[i : i + 1]))
                ssim_list.append(_compute_ssim(pred_np[i : i + 1], gt_np[i : i + 1]))
                n_done += 1
                if n_done >= max_samples:
                    break
            if n_done >= max_samples:
                break

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def train_synth_pair_ddpm(
    synth_root: str,
    save_dir: str,
    result_dir: Optional[str],
    cfg: TrainConfig,
):
    out_dir = result_dir if result_dir is not None else save_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    cls_map = {
        "normal": 0,
        "marking": 1,
        "out_of_focus": 2,
        "tattoo": 3,
        "tissue_folding": 4,
    }
    ds = SynthEvalDataset(synth_root, cls_map)
    # ds.items: (imp_path, ori_path, label)
    paths = [imp for (imp, _ori, _lbl) in ds.items]
    train_idx, val_idx = split_indices_by_hash(paths, val_ratio=cfg.val_ratio, seed=cfg.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = CondUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    betas, alphas_cumprod = prepare_diffusion(cfg.timesteps, device=device)

    best_psnr = -1.0
    best_epoch = -1

    log_path = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_psnr,val_ssim,is_best\n")

    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        epoch_loss = 0.0
        n_seen = 0

        for batch in pbar:
            cond = batch["inpainted"].to(device)
            gt = batch["ori"].to(device)  # target clean

            B = gt.size(0)
            t = torch.randint(0, cfg.timesteps, (B,), device=device, dtype=torch.long)
            x_t, eps = q_sample(gt, t, alphas_cumprod)
            eps_hat = model(x_t, cond, t)
            loss = F.mse_loss(eps_hat, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            n_seen += B
            epoch_loss += float(loss.item()) * B
            pbar.set_postfix({"loss": float(loss.item())})

        epoch_loss /= max(1, n_seen)
        print(f"Epoch {epoch+1} avg loss: {epoch_loss:.6f}")

        val_psnr = None
        val_ssim = None
        is_best = False

        if (epoch + 1) % cfg.eval_every == 0:
            model.eval()
            val_psnr, val_ssim = eval_on_val(
                model=model,
                loader=val_loader,
                betas=betas,
                alphas_cumprod=alphas_cumprod,
                t_start=cfg.eval_t_start,
                max_samples=cfg.eval_max_samples,
                device=device,
                seed=cfg.seed,
            )
            print(f"Epoch {epoch+1} val PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch + 1
                is_best = True

        with open(log_path, "a") as f:
            psnr_str = f"{val_psnr:.4f}" if val_psnr is not None else ""
            ssim_str = f"{val_ssim:.4f}" if val_ssim is not None else ""
            f.write(f"{epoch+1},{epoch_loss},{psnr_str},{ssim_str},{int(is_best)}\n")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "config": {
                "timesteps": cfg.timesteps,
                "conditional": True,
                "val_ratio": cfg.val_ratio,
                "seed": cfg.seed,
            },
        }

        if is_best:
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))

        if epoch + 1 == cfg.num_epochs:
            torch.save(ckpt, os.path.join(out_dir, "final.pt"))
            results_path = os.path.join(out_dir, "results.csv")
            with open(results_path, "w") as f:
                f.write("best_epoch,best_psnr,final_epoch,final_train_loss\n")
                f.write(f"{best_epoch},{best_psnr:.4f},{epoch+1},{epoch_loss:.6f}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional DDPM on ArtiDiffuser-Synth pairs (maskless).")
    p.add_argument("--synth_root", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--result_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--eval_t_start", type=int, default=100)
    p.add_argument("--eval_max_samples", type=int, default=200)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        timesteps=args.timesteps,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
        eval_every=args.eval_every,
        eval_t_start=args.eval_t_start,
        eval_max_samples=args.eval_max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_synth_pair_ddpm(
        synth_root=args.synth_root,
        save_dir=args.save_dir,
        result_dir=args.result_dir,
        cfg=cfg,
    )

