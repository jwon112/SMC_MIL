import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from artidiffuser.dataloader import build_dataloader


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_steps: int):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("freqs", emb, persistent=False)

    def forward(self, t: torch.LongTensor) -> torch.Tensor:
        # t: [B]
        t = t.float().unsqueeze(1)  # [B,1]
        freqs = self.freqs.unsqueeze(0)  # [1,half_dim]
        angles = t * freqs  # [B,half_dim]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] < self.dim:
            pad = self.dim - emb.shape[1]
            emb = F.pad(emb, (0, pad))
        return emb  # [B,dim]


class SimpleUNet(nn.Module):
    """
    간단한 UNet 구조 + time embedding (채널에 주입).
    입력: x_t [B,3,H,W], t [B], cls [B] (현재는 cls는 사용하지 않음, 필요시 cond에 추가 가능).
    출력: noise 예측 [B,3,H,W]
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

        self.down1 = conv_block(3, base_channels)
        self.down2 = conv_block(base_channels, base_channels * 2)
        self.down3 = conv_block(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(2)

        self.mid = conv_block(base_channels * 4, base_channels * 4)

        # up path: 채널 수는 forward의 concat 구조에 맞게 조정
        # u3 입력: concat([u3(4C), d2(2C)]) → 6C
        self.up3 = conv_block(base_channels * 6, base_channels * 2)
        # u2 입력: concat([u2(2C), d1(C)]) → 3C
        self.up2 = conv_block(base_channels * 3, base_channels)
        self.up1 = conv_block(base_channels * 2, base_channels)

        self.time_proj = nn.Linear(time_dim, base_channels * 4)

        self.final = nn.Conv2d(base_channels, 3, 1)

    def forward(self, x: torch.Tensor, t: torch.LongTensor) -> torch.Tensor:
        # x: [B,3,H,W], t: [B]
        t_emb = self.time_mlp(t)  # [B,time_dim]
        t_proj = self.time_proj(t_emb)  # [B, Cmid]
        t_proj = t_proj[:, :, None, None]  # [B,Cmid,1,1]

        d1 = self.down1(x)         # [B,C,H,W]
        d2 = self.down2(self.pool(d1))   # [B,2C,H/2,W/2]
        d3 = self.down3(self.pool(d2))   # [B,4C,H/4,W/4]

        mid = self.mid(d3 + t_proj)      # time embedding 주입

        u3 = F.interpolate(mid, scale_factor=2, mode="nearest")
        u3 = self.up3(torch.cat([u3, d2], dim=1))

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, d1], dim=1))

        # 마지막 업샘플은 입력 크기와 맞는다고 가정 (H,W가 4의 배수일 것)
        out = self.up1(torch.cat([u2, d1], dim=1))
        out = self.final(out)
        return out


def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def prepare_diffusion(T: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def forward_diffusion_sample(x0: torch.Tensor, t: torch.LongTensor, alphas_cumprod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x0: [B,3,H,W], t: [B], alphas_cumprod: [T]
    return: x_t, noise eps
    """
    device = x0.device
    noise = torch.randn_like(x0)
    a_bar = alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
    return x_t, noise


def train_coad_ddpm(
    coad_root: str,
    save_dir: str,
    batch_size: int = 8,
    num_epochs: int = 50,
    timesteps: int = 1000,
    lr: float = 1e-4,
    device: str = "cuda",
    w_mask: float = 10.0,
    synth_root: Optional[str] = None,
    eval_every: int = 5,
    eval_t_start: int = 100,
    eval_max_samples: int = 200,
    result_dir: Optional[str] = None,
):
    os.makedirs(save_dir, exist_ok=True)
    out_dir = result_dir if result_dir is not None else save_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    dataloader: DataLoader = build_dataloader(
        coad_root=coad_root,
        synth_root=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        coad_only=True,
    )

    model = SimpleUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    betas, alphas_cumprod = prepare_diffusion(timesteps, device=device)

    global_step = 0
    best_psnr: float = -1.0
    best_ssim: float = -1.0
    best_epoch: int = -1

    # 로그 파일 (모든 epoch 기록)
    log_path = os.path.join(out_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,psnr,ssim,is_best\n")
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch in pbar:
            img = batch["img"].to(device)          # [B,3,H,W] in [-1,1]
            cls = batch["cls"].to(device)          # [B]

            B = img.size(0)
            t = torch.randint(0, timesteps, (B,), device=device, dtype=torch.long)

            x_t, noise = forward_diffusion_sample(img, t, alphas_cumprod)

            noise_pred = model(x_t, t)

            # 마스크 없이 순수 per-pixel MSE
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item() * B
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch {epoch+1} avg loss: {epoch_loss:.6f}")

        # 현재 epoch 상태 dict
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "config": {
                "timesteps": timesteps,
                "w_mask": w_mask,
            },
        }

        # Synth validation (synth_root 지정 시 eval_every epoch마다)
        psnr = None
        ssim = None
        if synth_root and (epoch + 1) % eval_every == 0:
            from artidiffuser.eval_synth_ddpm import run_eval

            # 임시 체크포인트 저장 후 eval
            tmp_ckpt_path = os.path.join(out_dir, "_tmp_eval.pt")
            torch.save(ckpt, tmp_ckpt_path)

            psnr, ssim = run_eval(
                ckpt_path=tmp_ckpt_path,
                synth_root=synth_root,
                t_start=eval_t_start,
                batch_size=4,
                max_samples=eval_max_samples,
                device=str(device),
                seed=42,
            )
            print(f"Epoch {epoch+1} Synth val PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

            # best 갱신 시 best.pt로 저장
            if psnr is not None and psnr > best_psnr:
                best_psnr = psnr
                best_ssim = ssim if ssim is not None else -1.0
                best_epoch = epoch + 1
                best_path = os.path.join(out_dir, "best.pt")
                torch.save(ckpt, best_path)
                is_best = True
            else:
                is_best = False
        else:
            is_best = False

        # 로그 파일에 한 줄 추가
        with open(log_path, "a") as f:
            psnr_str = f"{psnr:.4f}" if psnr is not None else ""
            ssim_str = f"{ssim:.4f}" if ssim is not None else ""
            f.write(f"{epoch+1},{epoch_loss},{psnr_str},{ssim_str},{int(is_best)}\n")

        # 마지막 epoch이면 final.pt 저장
        if epoch + 1 == num_epochs:
            final_path = os.path.join(out_dir, "final.pt")
            torch.save(ckpt, final_path)
            print(f"Saved final checkpoint to {final_path}")

            # results.csv: 최종 요약만 한 줄
            results_path = os.path.join(out_dir, "results.csv")
            with open(results_path, "w") as f:
                f.write("best_epoch,best_psnr,best_ssim,final_epoch,final_train_loss\n")
                f.write(
                    f"{best_epoch},{best_psnr:.4f if best_psnr>=0 else -1},{best_ssim:.4f if best_ssim>=0 else -1},{epoch+1},{epoch_loss}\n"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple DDPM on COAD-Artifact (COAD-only experiment).")
    parser.add_argument(
        "--coad_root",
        type=str,
        required=True,
        help="Root directory of COAD-Artifact, e.g. /home/jupyter/data/image_team/COAD-Artifact",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w_mask", type=float, default=10.0)
    parser.add_argument(
        "--synth_root",
        type=str,
        default=None,
        help="ArtiDiffuser-Synth root for validation (PSNR/SSIM). If set, eval every --eval_every epochs.",
    )
    parser.add_argument("--eval_every", type=int, default=5, help="Run Synth validation every N epochs")
    parser.add_argument("--eval_t_start", type=int, default=100, help="t_start for restoration eval")
    parser.add_argument("--eval_max_samples", type=int, default=200, help="Max Synth samples for eval")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Directory to save logs and best/final checkpoints (default: save_dir).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_coad_ddpm(
        coad_root=args.coad_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        timesteps=args.timesteps,
        lr=args.lr,
        device=args.device,
        w_mask=args.w_mask,
        synth_root=args.synth_root,
        eval_every=args.eval_every,
        eval_t_start=args.eval_t_start,
        eval_max_samples=args.eval_max_samples,
        result_dir=args.result_dir,
    )

