from __future__ import annotations

import sys
import argparse
import contextlib
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Allow running either:
# - from repo root:  python -m mini.train ...
# - from inside mini: cd mini && python train.py ...
_MINI_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _MINI_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from mini.data.dataloader import DataSpec, build_loader, build_dataset
from mini.model.unet import UNet
from mini.module.blocks import UNetSpec


def seed_all(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def confusion_matrix(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    pred = logits.argmax(dim=1)  # BxHxW
    pred = pred.view(-1)
    target = target.view(-1)
    keep = target != ignore_index
    pred = pred[keep]
    target = target[keep]
    k = (target * num_classes + pred).to(torch.int64)
    cm = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


@torch.no_grad()
def miou_from_cm(cm: torch.Tensor, eps: float = 1e-6) -> float:
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    iou = tp / (tp + fp + fn + eps)
    return float(iou.mean().item())


@torch.no_grad()
def mdice_from_cm(cm: torch.Tensor, *, ignore_background: bool = True, eps: float = 1e-6) -> float:
    """
    Multi-class Dice computed from confusion matrix.
    nnU-Net-style reporting typically uses mean foreground Dice (exclude background).
    - Background(class 0) is excluded when ignore_background=True.
    - Classes that never appear in GT (tp+fn == 0) are excluded from the mean so that
      "correctly predicting all zeros" for those classes does not drag the mean down.
    """
    cm = cm.to(torch.float32)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    present = (tp + fn) > 0
    if ignore_background and present.numel() > 0:
        present[0] = False  # class 0 is background in VOC

    valid = present.nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return 0.0

    return float(dice[valid].mean().item())


def _voc_color_map(num_classes: int = 21) -> torch.Tensor:
    """
    Standard PASCAL VOC color map (21 classes).
    Returns tensor of shape [num_classes, 3] with uint8 RGB values.
    """
    import numpy as np

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= (bitget(c, 0) << (7 - j))
            g |= (bitget(c, 1) << (7 - j))
            b |= (bitget(c, 2) << (7 - j))
            c >>= 3
        cmap[i] = np.array([r, g, b], dtype=np.uint8)
    return torch.from_numpy(cmap)


@torch.no_grad()
def _save_sample_visuals(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    num_samples: int,
    num_classes: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> None:
    """
    Save a few (image, GT, prediction) triplets for qualitative inspection.
    """
    import numpy as np
    from PIL import Image

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = _voc_color_map(num_classes=num_classes)

    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    saved = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
        preds = logits.argmax(dim=1)

        imgs_denorm = imgs.detach().cpu() * std_t + mean_t
        imgs_denorm = imgs_denorm.clamp(0.0, 1.0)
        preds = preds.detach().cpu()
        masks = masks.detach().cpu()

        bsz = imgs_denorm.size(0)
        for b in range(bsz):
            if saved >= num_samples:
                return

            img = imgs_denorm[b]  # 3xHxW
            img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            gt = masks[b].numpy().astype(np.int64)
            pr = preds[b].numpy().astype(np.int64)

            # 255는 VOC ignore index이므로 색맵 인덱싱 전에 안전한 값(배경=0)으로 내려준다.
            gt_vis = gt.copy()
            gt_vis[gt_vis == 255] = 0
            pr_vis = pr.copy()
            pr_vis[pr_vis == 255] = 0

            gt_color = cmap[gt_vis].numpy()
            pr_color = cmap[pr_vis].numpy()

            # concatenate horizontally: [input | GT | pred]
            vis = np.concatenate([img_np, gt_color, pr_color], axis=1)
            Image.fromarray(vis).save(out_dir / f"sample_{saved:03d}.png")
            saved += 1

        if saved >= num_samples:
            return


def _open_log(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    f = open(log_path, "a", encoding="utf-8")
    f.write("\n" + "=" * 80 + "\n")
    f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.flush()
    return f


def _log_print(log_f, msg: str) -> None:
    print(msg, flush=True)
    if log_f is not None:
        log_f.write(msg + "\n")
        log_f.flush()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = 0.0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")
    n = 0

    for imgs, masks in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            loss_meter += float(loss.item()) * imgs.size(0)
            cm += confusion_matrix(logits.detach().cpu(), masks.detach().cpu(), num_classes=num_classes, ignore_index=ignore_index)
            n += imgs.size(0)

    return {
        "loss": loss_meter / max(n, 1),
        "dice": mdice_from_cm(cm, ignore_background=True),
        "miou": miou_from_cm(cm),
    }


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, float]:
    model.eval()
    loss_meter = 0.0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")
    n = 0

    for imgs, masks in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
        loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)

        loss_meter += float(loss.item()) * imgs.size(0)
        cm += confusion_matrix(logits.detach().cpu(), masks.detach().cpu(), num_classes=num_classes, ignore_index=ignore_index)
        n += imgs.size(0)

    return {
        "loss": loss_meter / max(n, 1),
        "dice": mdice_from_cm(cm, ignore_background=True),
        "miou": miou_from_cm(cm),
    }


def _save_metric_plot(history, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train"]["loss"] for h in history]
    va_loss = [h["val"]["loss"] for h in history]
    tr_dice = [h["train"]["dice"] for h in history]
    va_dice = [h["val"]["dice"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)

    ax = axes[0]
    ax.plot(epochs, tr_loss, label="train")
    ax.plot(epochs, va_loss, label="val")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, tr_dice, label="train")
    ax.plot(epochs, va_dice, label="val")
    ax.set_title("Mean foreground Dice")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _estimate_flops(model: nn.Module, x: torch.Tensor) -> Optional[float]:
    """
    Return FLOPs for a single forward (not MACs), if a supported profiler is available.
    - fvcore: returns total ops count (we treat as FLOPs)
    - thop: returns MACs; we convert to FLOPs by *2
    """
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        return float(FlopCountAnalysis(model, x).total())
    except Exception:
        pass
    try:
        from thop import profile  # type: ignore

        macs, _params = profile(model, inputs=(x,), verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None


@torch.no_grad()
def _measure_latency_ms(model: nn.Module, x: torch.Tensor, *, iters: int = 50, warmup: int = 10) -> float:
    model.eval()
    device = x.device

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(iters):
            _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        return float(starter.elapsed_time(ender) / iters)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    t1 = time.perf_counter()
    return float((t1 - t0) * 1000.0 / iters)


def _append_results_row(results_csv: Path, row: Dict[str, object]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="voc")
    p.add_argument("--data_root", type=str, default="./_data")
    p.add_argument("--run_dir", type=str, default="./mini/runs/exp")
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    # scheduler
    p.add_argument("--scheduler", type=str, default="poly", choices=["none", "poly", "cosine"])
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--poly_power", type=float, default=0.9)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--do_test", action="store_true", default=True)
    p.add_argument("--test_set", type=str, default="val", choices=["train", "val"])

    # model knobs (swappable)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--norm", type=str, default="bn")
    p.add_argument("--act", type=str, default="relu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--pool", type=str, default="max")
    p.add_argument("--up", type=str, default="bilinear")
    p.add_argument("--encoder_type", type=str, default="plain", choices=["plain", "convnext_tiny"])
    p.add_argument("--encoder_pretrained", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--encoder_lr_scale", type=float, default=0.1, help="Relative LR factor for encoder params")
    p.add_argument("--block", type=str, default="conv", choices=["conv", "convnext", "convnextv2", "kconv", "kdwsep"])
    p.add_argument("--convnext_num_blocks", type=int, default=2)
    p.add_argument("--convnext_layer_scale", type=float, default=1e-6)
    p.add_argument("--convnext_drop_path", type=float, default=0.0)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--overfit_n", type=int, default=0)
    p.add_argument("--num_vis", type=int, default=0, help="Number of qualitative (img, gt, pred) samples to save from test_set")

    # data aug knobs
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--scale_min", type=float, default=0.5)
    p.add_argument("--scale_max", type=float, default=2.0)
    p.add_argument("--hflip", type=float, default=0.5)
    p.add_argument("--color_jitter", type=float, default=0.2)
    p.add_argument("--gaussian_blur_p", type=float, default=0.0)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_f = _open_log(run_dir)

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset.lower() == "voc":
        num_classes = 21
        ignore_index = 255
    else:
        raise ValueError("Only --dataset voc is supported for now")

    data_spec = DataSpec(
        dataset=args.dataset,
        data_root=args.data_root,
        image_set="train",
        download=bool(args.download),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),
        crop_size=args.crop_size,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        gaussian_blur_p=args.gaussian_blur_p,
    )
    val_spec = DataSpec(
        dataset=args.dataset,
        data_root=args.data_root,
        image_set="val",
        download=bool(args.download),
        num_workers=max(1, args.num_workers // 2),
        batch_size=args.batch_size,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),
        crop_size=args.crop_size,
        scale_min=1.0,
        scale_max=1.0,
        hflip=0.0,
        color_jitter=0.0,
        gaussian_blur_p=0.0,
    )

    if args.overfit_n and args.overfit_n > 0:
        base_ds = build_dataset(data_spec)
        n = min(int(args.overfit_n), len(base_ds))
        # deterministic small subset: first n indices
        indices = list(range(n))
        subset = Subset(base_ds, indices)
        train_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            drop_last=False,
        )
        val_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            drop_last=False,
        )
    else:
        train_loader = build_loader(data_spec, shuffle=True)
        val_loader = build_loader(val_spec, shuffle=False)

    if args.test_set == "train":
        test_loader = build_loader(data_spec, shuffle=False)
    else:
        test_loader = val_loader

    model_spec = UNetSpec(
        in_channels=3,
        num_classes=num_classes,
        base_channels=args.base_channels,
        depth=args.depth,
        norm=args.norm,
        act=args.act,
        dropout=args.dropout,
        pool=args.pool,
        up=args.up,
        encoder_type=args.encoder_type,
        encoder_pretrained=bool(args.encoder_pretrained),
        block=args.block,
        convnext_num_blocks=args.convnext_num_blocks,
        convnext_layer_scale=args.convnext_layer_scale,
        convnext_drop_path=args.convnext_drop_path,
        kernel_size=args.kernel_size,
    )
    model = UNet(model_spec).to(device)

    # Separate encoder/decoder param groups when using a pretrained encoder so we can
    # use a smaller LR on the backbone.
    if getattr(model, "encoder", None) is not None:
        enc_params = list(model.encoder.parameters())
        enc_param_ids = {id(p) for p in enc_params}
        dec_params = [p for p in model.parameters() if id(p) not in enc_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": dec_params, "lr": args.lr},
                {"params": enc_params, "lr": args.lr * float(args.encoder_lr_scale)},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp))
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    else:
        scaler = None

    # iteration-based LR scheduler (seg-friendly)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = int(args.epochs * steps_per_epoch)
    warmup_steps = int(float(args.warmup_epochs) * steps_per_epoch)
    warmup_steps = max(0, min(warmup_steps, total_steps))
    base_lr = float(optimizer.param_groups[0]["lr"])
    min_lr = float(args.min_lr)
    min_factor = (min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            warm = float(step + 1) / float(warmup_steps)
            return min_factor + (1.0 - min_factor) * warm
        t = step - warmup_steps
        T = max(1, total_steps - warmup_steps)
        if args.scheduler == "poly":
            s = float((1.0 - t / T) ** float(args.poly_power))
            return min_factor + (1.0 - min_factor) * s
        if args.scheduler == "cosine":
            import math

            s = float(0.5 * (1.0 + math.cos(math.pi * t / T)))
            return min_factor + (1.0 - min_factor) * s
        return min_factor + (1.0 - min_factor) * 1.0

    if args.scheduler == "none":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "data_spec": asdict(data_spec),
        "val_spec": asdict(val_spec),
        "model_spec": asdict(model_spec),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_val = {"epoch": -1, "dice": -1.0}
    history = []

    _log_print(log_f, f"run_dir={run_dir.resolve()}")
    _log_print(log_f, f"device={device} amp={bool(args.amp and device.type == 'cuda')}")
    _log_print(log_f, f"dataset={args.dataset} data_root={os.path.abspath(args.data_root)}")
    _log_print(log_f, f"model=UNet base={args.base_channels} depth={args.depth} norm={args.norm} act={args.act} up={args.up}")
    _log_print(
        log_f,
        f"scheduler={args.scheduler} warmup_epochs={args.warmup_epochs} min_lr={args.min_lr} "
        f"steps/epoch={steps_per_epoch} total_steps={total_steps}",
    )
    if args.overfit_n and args.overfit_n > 0:
        _log_print(log_f, f"overfit_n={args.overfit_n} (using a fixed small subset of the training data)")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, num_classes, ignore_index)
        va = eval_one_epoch(model, val_loader, device, num_classes, ignore_index)
        lr_now = float(optimizer.param_groups[0]["lr"])
        row = {"epoch": epoch, "lr": lr_now, "train": tr, "val": va}
        history.append(row)

        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _save_metric_plot(history, run_dir / "metrics.png")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_spec": asdict(model_spec),
        }
        torch.save(ckpt, run_dir / "last.pt")

        if va["dice"] > best_val["dice"]:
            best_val = {"epoch": epoch, "dice": va["dice"]}
            torch.save(ckpt, run_dir / "best.pt")

        _log_print(
            log_f,
            f"[{epoch:03d}/{args.epochs}] "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"train loss={tr['loss']:.4f} dice={tr['dice']:.4f} | "
            f"val loss={va['loss']:.4f} dice={va['dice']:.4f} | "
            f"best(val dice)={best_val['dice']:.4f}@{best_val['epoch']}"
        )

    te = None
    if args.do_test:
        te = eval_one_epoch(model, test_loader, device, num_classes, ignore_index)
        _log_print(log_f, f"[test:{args.test_set}] loss={te['loss']:.4f} dice={te['dice']:.4f} miou={te['miou']:.4f}")

        # qualitative samples
        if args.num_vis and args.num_vis > 0:
            vis_dir = run_dir / "vis"
            # use training data normalization (val/test share same mean/std except jitter)
            _save_sample_visuals(
                model,
                test_loader,
                device,
                vis_dir,
                num_samples=int(args.num_vis),
                num_classes=num_classes,
                mean=data_spec.mean,
                std=data_spec.std,
            )
            _log_print(log_f, f"saved {args.num_vis} qualitative samples to {vis_dir}")

    # Summary row (results.csv)
    params = _count_params(model)
    dummy = torch.zeros(
        (1, 3, int(args.crop_size), int(args.crop_size)),
        dtype=torch.float32,
        device=device,
    )
    flops = _estimate_flops(model, dummy)
    latency_ms = _measure_latency_ms(model, dummy, iters=50, warmup=10)

    results_row = {
        "run_dir": str(run_dir.resolve()),
        "time": meta["time"],
        "dataset": args.dataset,
        "test_set": args.test_set,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_epoch": best_val["epoch"],
        "best_val_dice": best_val["dice"],
        "test_loss": (te["loss"] if te is not None else None),
        "test_dice": (te["dice"] if te is not None else None),
        "test_miou": (te["miou"] if te is not None else None),
        "params": params,
        "flops": flops,
        "latency_ms": latency_ms,
        "model_base_channels": args.base_channels,
        "model_depth": args.depth,
        "model_norm": args.norm,
        "model_act": args.act,
        "model_up": args.up,
    }
    results_csv = run_dir / "results" / "results.csv"
    _append_results_row(results_csv, results_row)
    _log_print(log_f, f"wrote {results_csv}")

    with contextlib.suppress(Exception):
        log_f.close()


if __name__ == "__main__":
    main()

