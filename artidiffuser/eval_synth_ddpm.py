"""
ArtiDiffuser-Synth (inpainted, ori) 페어로 DDPM 복원 지표(PSNR, SSIM) 평가.
inpainted를 t_start에서 노이즈 주입 후 역방향 샘플링하여 복원, ori와 비교.
"""
import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from artidiffuser.train_coad_ddpm import (
    SimpleUNet,
    forward_diffusion_sample,
    make_beta_schedule,
    prepare_diffusion,
)


def _build_synth_eval_loader(synth_root: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    from artidiffuser.dataloader import SynthEvalDataset

    cls_map = {
        "normal": 0,
        "marking": 1,
        "out_of_focus": 2,
        "tattoo": 3,
        "tissue_folding": 4,
    }
    ds = SynthEvalDataset(synth_root, cls_map)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def ddpm_reverse_step(
    x_t: torch.Tensor,
    t: int,
    model: torch.nn.Module,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    DDPM 역방향 한 스텝: x_t -> x_{t-1}
    x_t: [B,3,H,W], t: scalar (1..T)
    """
    B = x_t.size(0)
    t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

    eps_pred = model(x_t, t_tensor)

    beta_t = betas[t]
    alpha_t = 1.0 - beta_t
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)

    # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps) + sigma_t * z
    coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    x_prev = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps_pred)

    # t=1 -> x_0로 갈 때는 노이즈 추가 안 함
    if t > 1:
        sigma2_t = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
        sigma_t = torch.sqrt(sigma2_t)
        z = torch.randn_like(x_t, device=device)
        x_prev = x_prev + sigma_t * z

    return x_prev


def restore_from_inpainted(
    inpainted: torch.Tensor,
    model: torch.nn.Module,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    t_start: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    inpainted [B,3,H,W] in [-1,1] -> x_t_start 생성 후 t_start 스텝 역방향 -> x0_hat
    """
    if seed is not None:
        torch.manual_seed(seed)

    x_t = inpainted.to(device)
    # t_start에서 시작: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
    noise = torch.randn_like(x_t, device=device)
    a_bar = alphas_cumprod[t_start].view(1, 1, 1, 1)
    x_t = torch.sqrt(a_bar) * x_t + torch.sqrt(1.0 - a_bar) * noise

    for t in range(t_start, 0, -1):
        x_t = ddpm_reverse_step(x_t, t, model, betas, alphas_cumprod, device)

    return x_t  # x_0


def tensor_to_01(x: torch.Tensor) -> np.ndarray:
    """[B,3,H,W] in [-1,1] -> [B,H,W,3] in [0,1] numpy"""
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    return x


def compute_psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """pred, gt: [B,H,W,3] in [0,1]. returns mean PSNR over batch."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(max_val**2 / mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: [B,H,W,3] in [0,1]. returns mean SSIM over batch."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")

    vals = []
    for i in range(pred.shape[0]):
        p = pred[i]
        g = gt[i]
        v = ssim(p, g, channel_axis=2, data_range=1.0)
        vals.append(v)
    return float(np.mean(vals))


def run_eval(
    ckpt_path: str,
    synth_root: str,
    t_start: int = 100,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[float, float]:
    """
    체크포인트 로드 -> Synth 데이터로 복원 -> PSNR, SSIM 반환.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    betas = ckpt["betas"].to(device)
    alphas_cumprod = ckpt["alphas_cumprod"].to(device)
    timesteps = betas.size(0)

    if t_start >= timesteps:
        t_start = timesteps - 1

    model = SimpleUNet().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = _build_synth_eval_loader(synth_root, batch_size=batch_size)
    if max_samples is not None:
        total = min(len(loader.dataset), max_samples)
    else:
        total = len(loader.dataset)

    psnr_list: List[float] = []
    ssim_list: List[float] = []

    n_done = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Synth"):
            inpainted = batch["inpainted"]
            ori = batch["ori"]

            restored = restore_from_inpainted(
                inpainted, model, betas, alphas_cumprod, t_start, device, seed=seed
            )

            pred_np = tensor_to_01(restored)
            gt_np = tensor_to_01(ori)

            for i in range(pred_np.shape[0]):
                if max_samples is not None and n_done >= max_samples:
                    break
                p = compute_psnr(pred_np[i : i + 1], gt_np[i : i + 1])
                s = compute_ssim(pred_np[i : i + 1], gt_np[i : i + 1])
                psnr_list.append(p)
                ssim_list.append(s)
                n_done += 1

            if max_samples is not None and n_done >= max_samples:
                break

    psnr_mean = float(np.mean(psnr_list))
    ssim_mean = float(np.mean(ssim_list))
    return psnr_mean, ssim_mean


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DDPM restoration on ArtiDiffuser-Synth (PSNR, SSIM)."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--synth_root",
        type=str,
        required=True,
        help="Root of ArtiDiffuser-Synth, e.g. /home/jupyter/data/image_team/ArtiDiffuser-Synth",
    )
    parser.add_argument(
        "--t_start",
        type=int,
        default=100,
        help="Start timestep for restoration (higher = more denoising steps, default 100)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    psnr, ssim = run_eval(
        ckpt_path=args.ckpt,
        synth_root=args.synth_root,
        t_start=args.t_start,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
    )

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
