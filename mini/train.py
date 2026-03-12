from __future__ import annotations

import sys
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Allow running either:
# - from repo root:  python -m mini.train ...
# - from inside mini: cd mini && python train.py ...
_MINI_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _MINI_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from mini.data.dataloader import DataSpec, build_loader
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


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
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
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(imgs)
                loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_meter += float(loss.item()) * imgs.size(0)
            cm += confusion_matrix(logits.detach().cpu(), masks.detach().cpu(), num_classes=num_classes, ignore_index=ignore_index)
            n += imgs.size(0)

    return {"loss": loss_meter / max(n, 1), "miou": miou_from_cm(cm)}


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
        loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)

        loss_meter += float(loss.item()) * imgs.size(0)
        cm += confusion_matrix(logits.detach().cpu(), masks.detach().cpu(), num_classes=num_classes, ignore_index=ignore_index)
        n += imgs.size(0)

    return {"loss": loss_meter / max(n, 1), "miou": miou_from_cm(cm)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="voc")
    p.add_argument("--data_root", type=str, default="./_data")
    p.add_argument("--run_dir", type=str, default="./mini/runs/exp")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")

    # model knobs (swappable)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--norm", type=str, default="bn")
    p.add_argument("--act", type=str, default="relu")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--pool", type=str, default="max")
    p.add_argument("--up", type=str, default="bilinear")

    # data aug knobs
    p.add_argument("--resize", type=int, default=384)
    p.add_argument("--crop_size", type=int, default=352)
    p.add_argument("--hflip", type=float, default=0.5)

    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

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
        download=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        resize=args.resize if args.resize > 0 else None,
        crop_size=args.crop_size if args.crop_size > 0 else None,
        hflip=args.hflip,
    )
    val_spec = DataSpec(
        dataset=args.dataset,
        data_root=args.data_root,
        image_set="val",
        download=True,
        num_workers=max(1, args.num_workers // 2),
        batch_size=args.batch_size,
        resize=args.resize if args.resize > 0 else None,
        crop_size=args.crop_size if args.crop_size > 0 else None,
        hflip=0.0,
    )

    train_loader = build_loader(data_spec, shuffle=True)
    val_loader = build_loader(val_spec, shuffle=False)

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
    )
    model = UNet(model_spec).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "args": vars(args),
        "data_spec": asdict(data_spec),
        "val_spec": asdict(val_spec),
        "model_spec": asdict(model_spec),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best = {"epoch": -1, "miou": -1.0}
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, num_classes, ignore_index)
        va = eval_one_epoch(model, val_loader, device, num_classes, ignore_index)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)

        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_spec": asdict(model_spec),
        }
        torch.save(ckpt, run_dir / "last.pt")

        if va["miou"] > best["miou"]:
            best = {"epoch": epoch, "miou": va["miou"]}
            torch.save(ckpt, run_dir / "best.pt")

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train loss={tr['loss']:.4f} miou={tr['miou']:.4f} | "
            f"val loss={va['loss']:.4f} miou={va['miou']:.4f} | "
            f"best={best['miou']:.4f}@{best['epoch']}"
        )


if __name__ == "__main__":
    main()

