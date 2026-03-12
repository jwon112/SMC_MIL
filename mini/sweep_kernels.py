from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


# 3..21 odd (2D seg can get memory-heavy for larger kernels)
KERNEL_SIZES = list(range(3, 22, 2))
BLOCKS = ["kconv", "kdwsep"]  # normal conv vs depthwise-separable


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--runs_dir", type=str, default="./runs/kernel_sweep")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--download", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--scale_min", type=float, default=0.5)
    p.add_argument("--scale_max", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    train_py = Path(__file__).resolve().parent / "train.py"
    if not train_py.exists():
        raise SystemExit(f"train.py not found: {train_py}")

    for block in BLOCKS:
        for k in KERNEL_SIZES:
            exp_name = f"{block}_k{k:02d}"
            run_dir = runs_dir / exp_name

            cmd = [
                sys.executable,
                str(train_py),
                "--dataset",
                "voc",
                "--data_root",
                args.data_root,
                "--run_dir",
                str(run_dir),
                "--no-download" if not args.download else "--download",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--no-pin_memory" if not args.pin_memory else "--pin_memory",
                "--persistent_workers" if args.persistent_workers else "--no-persistent_workers",
                "--prefetch_factor",
                str(args.prefetch_factor),
                "--crop_size",
                str(args.crop_size),
                "--scale_min",
                str(args.scale_min),
                "--scale_max",
                str(args.scale_max),
                "--seed",
                str(args.seed),
                "--block",
                block,
                "--kernel_size",
                str(k),
                "--do_test",
            ]
            if args.amp:
                cmd.append("--amp")

            if args.dry_run:
                print("[dry_run]", " ".join(cmd))
            else:
                _run(cmd)


if __name__ == "__main__":
    main()

