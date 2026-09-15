#!/usr/bin/env python3
"""Summarize pooled OOF metrics across repeated SMC 5-fold experiments."""

from __future__ import annotations

import argparse
import pickle
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


PATTERN = re.compile(
    r"^smc_(?P<task>acr_high_grade|amr_positive|significant_rejection)_"
    r"(?P<level>l[0-3])_(?P<mpp>0p25|0p50|1p00|2p00)mpp_"
    r"(?P<magnification>40x|20x|10x|5x)_uni2_clamsb_"
    r"cv(?P<folds>[35])val_s(?P<seed>[0-9]+)$"
)
TASK_NAMES = {
    "acr_high_grade": "acr_high",
    "amr_positive": "amr_positive",
    "significant_rejection": "significant_rejection",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/smc_repeated_cv_summary"))
    parser.add_argument("--folds", type=int, choices=(3, 5), default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1, 11, 21, 31, 41))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def positive_probability(value: object) -> float:
    probabilities = np.asarray(value, dtype=float).reshape(-1)
    if probabilities.size != 2:
        raise ValueError(f"Expected two class probabilities, got {probabilities.shape}")
    return float(probabilities[1])


def read_oof(experiment_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(experiment_dir.glob("split_*_results.pkl")):
        fold = int(result_path.stem.split("_")[1])
        with result_path.open("rb") as handle:
            result = pickle.load(handle)
        for slide_id, item in result.items():
            rows.append(
                {
                    "fold": fold,
                    "slide_id": str(slide_id),
                    "label": int(item["label"]),
                    "probability": positive_probability(item["prob"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No split result files in {experiment_dir}")
    if frame["slide_id"].duplicated().any():
        raise RuntimeError(f"A slide occurs in multiple held-out folds: {experiment_dir}")
    return frame


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": len(frame),
        "positive_n": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def main() -> int:
    args = parse_args()
    requested_seeds = set(args.seeds)
    rows: list[dict[str, object]] = []
    for experiment_dir in sorted(args.results_root.iterdir()):
        if not experiment_dir.is_dir() or not (experiment_dir / "summary.csv").is_file():
            continue
        match = PATTERN.match(experiment_dir.name)
        if match is None:
            continue
        info = match.groupdict()
        seed = int(info["seed"])
        if int(info["folds"]) != args.folds or seed not in requested_seeds:
            continue
        oof = read_oof(experiment_dir)
        rows.append(
            {
                "experiment": experiment_dir.name,
                "task": TASK_NAMES[info["task"]],
                "magnification": info["magnification"],
                "mpp": info["mpp"].replace("p", "."),
                "seed": seed,
                "folds": args.folds,
                "threshold": args.threshold,
                **metrics(oof, args.threshold),
            }
        )

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise FileNotFoundError("No completed repeated-CV experiments matched the requested folds and seeds")

    expected = pd.MultiIndex.from_product(
        [TASK_NAMES.values(), ("40x", "20x", "10x", "5x"), sorted(requested_seeds)],
        names=("task", "magnification", "seed"),
    )
    actual = pd.MultiIndex.from_frame(per_seed[["task", "magnification", "seed"]])
    missing = expected.difference(actual)

    metric_columns = ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy")
    aggregate = per_seed.groupby(["task", "magnification", "mpp"], as_index=False).agg(
        seeds_completed=("seed", "nunique"),
        n=("n", "first"),
        positive_n=("positive_n", "first"),
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in metric_columns
            for stat in ("mean", "std", "min", "max")
        },
    )

    paired_rows: list[dict[str, object]] = []
    for task, task_frame in per_seed.groupby("task"):
        for left, right in combinations(("40x", "20x", "10x", "5x"), 2):
            left_frame = task_frame[task_frame.magnification.eq(left)].set_index("seed")
            right_frame = task_frame[task_frame.magnification.eq(right)].set_index("seed")
            common = left_frame.index.intersection(right_frame.index)
            for metric in ("auroc", "pr_auc", "balanced_accuracy"):
                differences = left_frame.loc[common, metric] - right_frame.loc[common, metric]
                paired_rows.append(
                    {
                        "task": task,
                        "metric": metric,
                        "comparison": f"{left}-{right}",
                        "seeds_compared": len(common),
                        "mean_difference": differences.mean(),
                        "std_difference": differences.std(ddof=1),
                        "left_wins": int((differences > 0).sum()),
                        "ties": int((differences == 0).sum()),
                        "right_wins": int((differences < 0).sum()),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed.sort_values(["task", "magnification", "seed"]).to_csv(
        args.output_dir / "per_seed_oof_metrics.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.sort_values(["task", "magnification"]).to_csv(
        args.output_dir / "repeated_cv_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "paired_scale_differences.csv", index=False, encoding="utf-8-sig"
    )

    print(f"[OK] repeated-CV runs collected: {len(per_seed)}")
    print(f"[OK] summary: {args.output_dir / 'repeated_cv_summary.csv'}")
    if len(missing):
        print(f"[WARN] incomplete experiment combinations: {len(missing)}")
        print(missing.to_frame(index=False).to_string(index=False))
        return 1
    print("[OK] all requested task/scale/seed combinations are complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
