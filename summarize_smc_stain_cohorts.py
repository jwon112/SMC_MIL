#!/usr/bin/env python3
"""Summarize class balance in stain-restricted SMC CV cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASKS = {
    "smc_acr_binary_0r_vs_1r2r3r": "ACR 0R vs 1R/2R/3R",
    "smc_acr_binary_0r1r_vs_2r3r": "ACR 0R/1R vs 2R/3R",
    "smc_amr_binary_pamr0_vs_positive": "AMR pAMR0 vs positive",
    "smc_any_rejection_binary": "Any rejection",
    "smc_significant_rejection_binary": "Significant rejection",
}
SPLITS = {
    "smc_acr_binary_0r_vs_1r2r3r": "smc_cv_acr_0r_vs_1r2r3r_standard3",
    "smc_acr_binary_0r1r_vs_2r3r": "smc_cv_acr_0r1r_vs_2r3r_standard3",
    "smc_amr_binary_pamr0_vs_positive": "smc_cv_amr_pamr0_vs_positive_standard3",
    "smc_any_rejection_binary": "smc_cv_any_rejection_standard3",
    "smc_significant_rejection_binary": "smc_cv_significant_rejection_standard3",
}
COHORTS = ("mixed_known", "he_only", "non_he", "ihc_only")
SUFFIX = "weak_unique_0to3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.cohort_root / "balance_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for task, task_display in TASKS.items():
        for cohort in COHORTS:
            summary_path = args.cohort_root / "splits" / f"{SPLITS[task]}_{SUFFIX}_{cohort}" / "fold_summary.csv"
            if not summary_path.is_file():
                print(f"[SKIP] missing: {summary_path}")
                continue
            frame = pd.read_csv(summary_path)
            frame.insert(0, "task", task)
            frame.insert(1, "task_display", task_display)
            frame.insert(2, "cohort", cohort)
            frame["train_negative_bags"] = frame.train_bags - frame.train_positive_bags
            frame["val_negative_bags"] = frame.val_gold_bags - frame.val_positive_bags
            frame["train_positive_pct"] = frame.train_positive_bags / frame.train_bags * 100
            frame["val_positive_pct"] = frame.val_positive_bags / frame.val_gold_bags * 100
            rows.append(frame)
    if not rows:
        raise ValueError("No fold_summary.csv files found")

    folds = pd.concat(rows, ignore_index=True)
    folds.to_csv(output_dir / "stain_cohort_fold_balance.csv", index=False)
    summary = (
        folds.groupby(["task", "task_display", "cohort"], as_index=False)
        .agg(
            train_bags_mean=("train_bags", "mean"),
            train_positive_min=("train_positive_bags", "min"),
            train_positive_max=("train_positive_bags", "max"),
            train_positive_pct_mean=("train_positive_pct", "mean"),
            val_bags_mean=("val_gold_bags", "mean"),
            val_positive_min=("val_positive_bags", "min"),
            val_positive_max=("val_positive_bags", "max"),
            val_positive_pct_mean=("val_positive_pct", "mean"),
        )
    )
    summary.to_csv(output_dir / "stain_cohort_balance_summary.csv", index=False)

    display = summary[[
        "task_display", "cohort", "train_bags_mean", "train_positive_min",
        "train_positive_max", "train_positive_pct_mean", "val_bags_mean",
        "val_positive_min", "val_positive_max", "val_positive_pct_mean",
    ]].copy()
    display["train_bags_mean"] = display.train_bags_mean.round().astype(int)
    display["train_positive_pct_mean"] = display.train_positive_pct_mean.map("{:.1f}%".format)
    display["val_bags_mean"] = display.val_bags_mean.round().astype(int)
    display["val_positive_pct_mean"] = display.val_positive_pct_mean.map("{:.1f}%".format)
    print(display.to_string(index=False))
    print(f"Fold detail: {output_dir / 'stain_cohort_fold_balance.csv'}")
    print(f"Summary: {output_dir / 'stain_cohort_balance_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
