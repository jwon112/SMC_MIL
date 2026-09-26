import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from artidiffuser.dataloader import SynthEvalDataset
from artidiffuser.train_synth_pair_ddpm import CondUNet, p_sample_step


def _to_01(x: torch.Tensor) -> np.ndarray:
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    return x.permute(0, 2, 3, 1).cpu().numpy()


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


def _save_triplet(inpainted: np.ndarray, restored: np.ndarray, ori: np.ndarray, out_path: str) -> None:
    def u8(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return (x * 255).astype(np.uint8)

    h, w, _ = inpainted.shape
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, 0:w] = u8(inpainted)
    canvas[:, w : 2 * w] = u8(restored)
    canvas[:, 2 * w : 3 * w] = u8(ori)
    Image.fromarray(canvas).save(out_path)


def restore(
    cond: torch.Tensor,
    model: torch.nn.Module,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    t_start: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    x_t = torch.randn_like(cond, device=device)
    for t in range(t_start, 0, -1):
        x_t = p_sample_step(x_t, cond, t, model, betas, alphas_cumprod, device)
    return x_t


def run_eval(
    ckpt_path: str,
    synth_root: str,
    t_start: int = 100,
    batch_size: int = 4,
    max_samples: Optional[int] = 200,
    device: str = "cuda",
    seed: int = 42,
    result_dir: Optional[str] = None,
    max_triplet_images: int = 16,
) -> Tuple[float, float]:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device_t)
    betas = ckpt["betas"].to(device_t)
    alphas_cumprod = ckpt["alphas_cumprod"].to(device_t)

    model = CondUNet().to(device_t)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cls_map = {
        "normal": 0,
        "marking": 1,
        "out_of_focus": 2,
        "tattoo": 3,
        "tissue_folding": 4,
    }
    ds = SynthEvalDataset(synth_root, cls_map)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)

    psnr_list = []
    ssim_list = []
    n_done = 0
    triplet_saved = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Synth Pair"):
            cond = batch["inpainted"].to(device_t)
            gt = batch["ori"].to(device_t)

            pred = restore(
                cond=cond,
                model=model,
                betas=betas,
                alphas_cumprod=alphas_cumprod,
                t_start=t_start,
                device=device_t,
                seed=seed,
            )

            pred_np = _to_01(pred)
            gt_np = _to_01(gt)
            in_np = _to_01(cond)

            for i in range(pred_np.shape[0]):
                psnr_list.append(_compute_psnr(pred_np[i : i + 1], gt_np[i : i + 1]))
                ssim_list.append(_compute_ssim(pred_np[i : i + 1], gt_np[i : i + 1]))

                if result_dir is not None and triplet_saved < max_triplet_images:
                    _save_triplet(
                        inpainted=in_np[i],
                        restored=pred_np[i],
                        ori=gt_np[i],
                        out_path=os.path.join(result_dir, f"triplet_{triplet_saved:03d}.png"),
                    )
                    triplet_saved += 1

                n_done += 1
                if max_samples is not None and n_done >= max_samples:
                    break
            if max_samples is not None and n_done >= max_samples:
                break

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def main():
    p = argparse.ArgumentParser(description="Evaluate conditional DDPM trained on Synth pairs.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--synth_root", type=str, required=True)
    p.add_argument("--t_start", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result_dir", type=str, default=None)
    args = p.parse_args()

    psnr, ssim = run_eval(
        ckpt_path=args.ckpt,
        synth_root=args.synth_root,
        t_start=args.t_start,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        seed=args.seed,
        result_dir=args.result_dir,
    )
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()

