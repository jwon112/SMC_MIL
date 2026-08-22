"""Summarize binary CLAM cross-validation predictions beyond AUROC.

The CLAM training loop saves one ``split_<fold>_results.pkl`` file per outer
fold.  This utility turns those bag-level probabilities into reproducible
fold-level and pooled out-of-fold (OOF) metrics.  It deliberately evaluates
the fixed 0.5 decision threshold: choosing a threshold on held-out test bags
would leak test information.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


TASK_LABELS = {
    "acr_0r_vs_rest": "ACR 0R vs 1R/2R/3R",
    "acr_high_grade": "ACR 0R/1R vs 2R/3R",
    "amr_positive": "AMR pAMR0 vs positive",
    "any_rejection": "Any rejection",
}


def experiment_metadata(name: str) -> tuple[str, str, str]:
    task_key = next((key for key in TASK_LABELS if key in name), "unknown")
    scale_match = re.search(r"(l[0-3]_\d+p\d+mpp_\d+x)", name)
    scale = scale_match.group(1) if scale_match else "unknown"
    return task_key, TASK_LABELS.get(task_key, "Unknown"), scale


def positive_probability(value: object) -> float:
    probs = np.asarray(value, dtype=float).reshape(-1)
    if probs.size != 2:
        raise ValueError(f"Expected two class probabilities, got shape {np.asarray(value).shape}")
    return float(probs[1])


def read_split(path: Path, experiment: str) -> list[dict[str, object]]:
    with path.open("rb") as handle:
        results = pickle.load(handle)

    task_key, task, scale = experiment_metadata(experiment)
    fold_match = re.search(r"split_(\d+)_results\.pkl$", path.name)
    fold = int(fold_match.group(1)) if fold_match else -1
    rows = []
    for slide_id, item in results.items():
        rows.append(
            {
                "experiment": experiment,
                "task_key": task_key,
                "task": task,
                "scale": scale,
                "fold": fold,
                "slide_id": str(slide_id),
                "label": int(item["label"]),
                "prob_positive": positive_probability(item["prob"]),
            }
        )
    return rows


def metric_row(frame: pd.DataFrame, threshold: float, scope: str, fold: object) -> dict[str, object]:
    labels = frame["label"].to_numpy(dtype=int)
    probs = frame["prob_positive"].to_numpy(dtype=float)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()

    return {
        "experiment": frame["experiment"].iloc[0],
        "task_key": frame["task_key"].iloc[0],
        "task": frame["task"].iloc[0],
        "scale": frame["scale"].iloc[0],
        "scope": scope,
        "fold": fold,
        "n_bags": len(frame),
        "positive_bags": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "threshold": threshold,
        "auroc": roc_auc_score(labels, probs),
        "average_precision": average_precision_score(labels, probs),
        "balanced_accuracy": balanced_accuracy_score(labels, pred),
        "sensitivity": recall_score(labels, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": precision_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "mcc": matthews_corrcoef(labels, pred) if len(np.unique(pred)) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute imbalance-aware metrics from CLAM outer-fold predictions.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/smc_cv_comparison"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be strictly between 0 and 1")

    split_paths = sorted(args.results_root.glob("*/split_*_results.pkl"))
    if not split_paths:
        raise FileNotFoundError(f"No split_*_results.pkl files under {args.results_root}")

    rows: list[dict[str, object]] = []
    for path in split_paths:
        rows.extend(read_split(path, path.parent.name))
    predictions = pd.DataFrame(rows).sort_values(["experiment", "fold", "slide_id"])

    metrics: list[dict[str, object]] = []
    for (experiment, fold), frame in predictions.groupby(["experiment", "fold"], sort=True):
        metrics.append(metric_row(frame, args.threshold, "outer_fold", int(fold)))
    for _, frame in predictions.groupby("experiment", sort=True):
        metrics.append(metric_row(frame, args.threshold, "pooled_oof", "all"))
    metrics_df = pd.DataFrame(metrics)

    fold_metrics = metrics_df.loc[metrics_df["scope"] == "outer_fold"].copy()
    metric_cols = [
        "auroc", "average_precision", "balanced_accuracy", "sensitivity",
        "specificity", "precision", "f1", "mcc",
    ]
    group_cols = ["experiment", "task_key", "task", "scale"]
    aggregate = fold_metrics.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    aggregate.columns = [
        "_".join(part for part in column if part).rstrip("_") if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    aggregate = aggregate.merge(
        metrics_df.loc[metrics_df["scope"] == "pooled_oof", group_cols + ["n_bags", "positive_bags", "positive_prevalence"] + metric_cols],
        on=group_cols,
        how="left",
        suffixes=("", "_pooled_oof"),
    ).sort_values(["task", "scale"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "oof_bag_predictions.csv", index=False)
    metrics_df.to_csv(args.output_dir / "fold_imbalance_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "averaged_imbalance_metrics.csv", index=False)

    display_cols = ["task", "scale", "auroc_mean", "average_precision_mean", "balanced_accuracy_mean", "sensitivity_mean", "specificity_mean", "f1_mean", "mcc_mean"]
    print(aggregate[display_cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved predictions and metrics to: {args.output_dir}")
    print("Metrics are bag-level. Splits are patient-grouped, but this does not make the metrics patient-level.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
