#!/usr/bin/env python3
"""Measure how repeated-seed ensemble performance changes with ensemble size."""

from __future__ import annotations

import argparse
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


SEED_METHODS = {
    "single_seed": "single_scale",
    "scale_fusion_per_seed": "multiscale",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("results/smc_prediction_ensembles/ensemble_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smc_prediction_ensembles/ensemble_size"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(1, 11, 21, 31, 41))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n": len(frame),
        "positive_n": int(labels.sum()),
        "negative_n": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.sum() else np.nan,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": threshold,
    }


def validate(frame: pd.DataFrame, seeds: tuple[int, ...]) -> None:
    required = {
        "evaluation_scope", "task", "cohort", "method", "scales", "seed",
        "unit_id", "label", "probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {sorted(missing)}")
    expected = set(seeds)
    for values, group in frame.groupby(
        ["evaluation_scope", "task", "cohort", "method", "scales"], sort=False
    ):
        found = set(group["seed"].dropna().astype(int))
        if found != expected:
            identity = "/".join(map(str, values))
            raise RuntimeError(f"Seed mismatch for {identity}: expected={sorted(expected)}, found={sorted(found)}")
        unit_counts = group.groupby("seed")["unit_id"].apply(lambda value: set(value))
        if len({frozenset(value) for value in unit_counts}) != 1:
            raise RuntimeError(f"Unit mismatch across seeds for {'/'.join(map(str, values))}")
        label_counts = group.groupby("unit_id")["label"].nunique()
        if (label_counts != 1).any():
            raise RuntimeError(f"Label disagreement across seeds for {'/'.join(map(str, values))}")


def ensemble_prediction(group: pd.DataFrame, seed_subset: tuple[int, ...]) -> pd.DataFrame:
    subset = group[group["seed"].astype(int).isin(seed_subset)]
    averaged = subset.groupby("unit_id", as_index=False).agg(
        label=("label", "first"),
        labels_seen=("label", "nunique"),
        probability=("probability", "mean"),
    )
    if (averaged["labels_seen"] != 1).any():
        raise RuntimeError("Label disagreement during seed averaging")
    return averaged.drop(columns="labels_seen")


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    if not args.predictions_csv.is_file():
        raise FileNotFoundError(args.predictions_csv)

    seeds = tuple(sorted(args.seeds))
    raw = pd.read_csv(args.predictions_csv, dtype={"unit_id": str})
    frame = raw[raw["method"].isin(SEED_METHODS)].copy()
    if frame.empty:
        raise ValueError("No single-seed prediction rows found")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["strategy"] = frame["method"].map(SEED_METHODS)
    validate(frame, seeds)

    combination_rows: list[dict[str, object]] = []
    keys = ["evaluation_scope", "task", "cohort", "method", "strategy", "scales"]
    for values, group in frame.groupby(keys, sort=False):
        metadata = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        for size in range(1, len(seeds) + 1):
            for seed_subset in combinations(seeds, size):
                prediction = ensemble_prediction(group, seed_subset)
                combination_rows.append({
                    **metadata,
                    "ensemble_size": size,
                    "seed_set": "+".join(map(str, seed_subset)),
                    **metrics(prediction, args.threshold),
                })

    combinations_frame = pd.DataFrame(combination_rows)
    metric_columns = ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy")
    summary = combinations_frame.groupby(
        ["evaluation_scope", "task", "cohort", "strategy", "scales", "ensemble_size"],
        as_index=False,
    ).agg(
        seed_combinations=("seed_set", "size"),
        n=("n", "first"),
        positive_n=("positive_n", "first"),
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in metric_columns
            for stat in ("mean", "std", "min", "max")
        },
    )

    baseline = summary[summary["ensemble_size"].eq(1)].set_index(
        ["evaluation_scope", "task", "cohort", "strategy", "scales"]
    )
    final = summary[summary["ensemble_size"].eq(len(seeds))].copy()
    for index, row in final.iterrows():
        key = tuple(row[column] for column in ("evaluation_scope", "task", "cohort", "strategy", "scales"))
        reference = baseline.loc[key]
        for metric in metric_columns:
            final.loc[index, f"delta_{metric}_vs_one_seed_mean"] = (
                row[f"{metric}_mean"] - reference[f"{metric}_mean"]
            )

    previous = summary[summary["ensemble_size"].eq(len(seeds) - 1)].set_index(
        ["evaluation_scope", "task", "cohort", "strategy", "scales"]
    )
    for index, row in final.iterrows():
        key = tuple(row[column] for column in ("evaluation_scope", "task", "cohort", "strategy", "scales"))
        reference = previous.loc[key]
        for metric in metric_columns:
            final.loc[index, f"delta_{metric}_vs_{len(seeds) - 1}_seed_mean"] = (
                row[f"{metric}_mean"] - reference[f"{metric}_mean"]
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combinations_frame.sort_values(
        ["evaluation_scope", "task", "cohort", "strategy", "scales", "ensemble_size", "seed_set"]
    ).to_csv(args.output_dir / "ensemble_size_combinations.csv", index=False, encoding="utf-8-sig")
    summary.sort_values(
        ["evaluation_scope", "task", "cohort", "strategy", "scales", "ensemble_size"]
    ).to_csv(args.output_dir / "ensemble_size_summary.csv", index=False, encoding="utf-8-sig")
    final.sort_values(
        ["evaluation_scope", "task", "cohort", "strategy", "scales"]
    ).to_csv(args.output_dir / "five_seed_gain_summary.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] source predictions: {len(frame):,}")
    print(f"[OK] seed combinations evaluated: {len(combinations_frame):,}")
    print(f"[OK] size summary: {args.output_dir / 'ensemble_size_summary.csv'}")
    print(f"[OK] five-seed gains: {args.output_dir / 'five_seed_gain_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
