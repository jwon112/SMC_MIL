#!/usr/bin/env python3
"""Collect future-pair baseline results across magnifications."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SCALE_PATTERN = re.compile(r"_(l[0-3])_([0-9p]+)mpp_(40x|20x|10x|5x)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/smc_future_pair_summary"))
    parser.add_argument("--folds", type=int, choices=(3, 4, 5), default=3)
    return parser.parse_args()


def experiment_metadata(path: Path) -> dict[str, object]:
    match = SCALE_PATTERN.search(path.parent.name)
    if not match:
        raise ValueError(f"Cannot parse scale from {path.parent.name}")
    level, mpp_text, magnification = match.groups()
    return {
        "experiment": path.parent.name,
        "level": level,
        "mpp": float(mpp_text.replace("p", ".")),
        "magnification": magnification,
        "source_path": str(path),
    }


def collect(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        frame = pd.read_csv(path)
        metadata = experiment_metadata(path)
        for key, value in reversed(list(metadata.items())):
            frame.insert(0, key, value)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    args = parse_args()
    pattern = f"smc_future_pair_*_gold{args.folds}cv_s1"
    experiment_dirs = sorted(path for path in args.results_root.glob(pattern) if path.is_dir())
    pooled_paths = [path / "pooled_metrics.csv" for path in experiment_dirs if (path / "pooled_metrics.csv").is_file()]
    horizon_paths = [path / "horizon_metrics.csv" for path in experiment_dirs if (path / "horizon_metrics.csv").is_file()]
    if not pooled_paths:
        raise FileNotFoundError(f"No completed experiments matched {args.results_root / pattern}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pooled = collect(pooled_paths).sort_values(
        ["evaluation_unit", "model_variant", "mpp"]
    )
    horizon = collect(horizon_paths)
    if not horizon.empty:
        horizon = horizon.sort_values(["model_variant", "horizon", "mpp"])
    pooled.to_csv(args.output_dir / "combined_pooled_metrics.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(args.output_dir / "combined_horizon_metrics.csv", index=False, encoding="utf-8-sig")

    columns = [
        "magnification", "model_variant", "evaluation_unit", "patients",
        "positive_patients", "pairs", "positive_pairs", "target_events",
        "positive_target_events", "auroc", "pr_auc", "sensitivity",
        "specificity", "balanced_accuracy",
    ]
    print(pooled[[column for column in columns if column in pooled]].to_string(index=False))
    print(f"[OK] Combined future-pair results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
