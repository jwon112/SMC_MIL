#!/usr/bin/env python3
"""Create repeated patient-grouped gold-only CV splits for the primary SMC tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PRIMARY_TASKS = (
    "task_smc_acr_binary_0r1r_vs_2r3r",
    "task_smc_amr_binary_pamr0_vs_positive",
    "task_smc_significant_rejection_binary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 21, 31, 41))
    parser.add_argument("--folds", type=int, choices=(3, 5), default=5)
    parser.add_argument("--tasks", nargs="+", choices=PRIMARY_TASKS, default=PRIMARY_TASKS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")

    root = Path(__file__).resolve().parents[1]
    split_script = root / "create_smc_cv_splits.py"
    for seed in args.seeds:
        for task in args.tasks:
            command = [
                sys.executable,
                str(split_script),
                "--task",
                task,
                "--folds",
                str(args.folds),
                "--seed",
                str(seed),
            ]
            subprocess.run(command, cwd=root, check=True)
    print(
        f"[DONE] repeated splits: folds={args.folds}, "
        f"seeds={','.join(map(str, args.seeds))}, tasks={len(args.tasks)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
