from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


KERNEL_SIZES = list(range(3, 32, 2))  # 3..31 odd
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
    p.add_argument("--num_workers", type=int, default=4)
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
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
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

