#!/usr/bin/env python3
"""Aggregate held-out slide-level CLAM predictions into event-level baselines."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


TASK_PREFIXES = {
    "acr_high": "acr_high_grade",
    "amr_positive": "amr_positive",
    "significant_rejection": "significant_rejection",
}
SCALES = {
    "40x": "l0_0p25mpp_40x",
    "20x": "l1_0p50mpp_20x",
    "10x": "l2_1p00mpp_10x",
    "5x": "l3_2p00mpp_5x",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASK_PREFIXES, default=list(TASK_PREFIXES))
    parser.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 11, 21, 31, 41])
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def positive_probability(value: object) -> float:
    probabilities = np.asarray(value, dtype=float).reshape(-1)
    if probabilities.size != 2:
        raise ValueError(f"Expected two class probabilities, got {probabilities.shape}")
    return float(probabilities[1])


def read_slide_oof(experiment_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result_path in sorted(experiment_dir.glob("split_*_results.pkl")):
        fold = int(result_path.stem.split("_")[1])
        with result_path.open("rb") as handle:
            result = pickle.load(handle)
        for slide_id, item in result.items():
            rows.append({
                "slide_id": str(slide_id),
                "fold": fold,
                "slide_label": int(item["label"]),
                "slide_probability": positive_probability(item["prob"]),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No split_*_results.pkl files in {experiment_dir}")
    if frame.slide_id.duplicated().any():
        raise ValueError(f"Slide occurs in multiple held-out folds: {experiment_dir}")
    return frame


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame.probability.to_numpy(dtype=float)
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    prevalence = float(labels.mean())
    pr_auc = float(average_precision_score(labels, probabilities))
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return {
        "events": len(frame),
        "positive_events": int(labels.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": pr_auc,
        "pr_auc_lift": float(pr_auc / prevalence) if prevalence else float("nan"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }


def experiment_dir(results_root: Path, task: str, scale: str, seed: int) -> Path:
    return results_root / (
        f"smc_{TASK_PREFIXES[task]}_{SCALES[scale]}_"
        f"uni2_clamsb_cv5val_s{seed}"
    )


def aggregate_events(
    slide_predictions: pd.DataFrame,
    event_slides: pd.DataFrame,
    events: pd.DataFrame,
    aggregation: str,
) -> pd.DataFrame:
    merged = event_slides.merge(slide_predictions, on="slide_id", how="inner", validate="many_to_one")
    if merged.empty:
        raise ValueError("No slide predictions matched the event manifest")
    probability_aggregation = "mean" if aggregation == "mean" else "max"
    event_predictions = merged.groupby("event_id", as_index=False).agg(
        probability=("slide_probability", probability_aggregation),
        slide_count=("slide_id", "nunique"),
        folds=("fold", "nunique"),
    )
    if not event_predictions.folds.eq(1).all():
        bad = event_predictions.loc[~event_predictions.folds.eq(1), "event_id"].tolist()
        raise ValueError(f"Slides from one event crossed held-out folds: {bad[:5]}")
    event_predictions = event_predictions.merge(
        events[["event_id", "case_id", "label"]], on="event_id", how="left", validate="one_to_one"
    )
    if event_predictions.label.isna().any():
        raise ValueError("Aggregated predictions include unknown events")
    return event_predictions.drop(columns="folds")


def main() -> int:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []

    for task in args.tasks:
        task_root = args.manifest_root / task
        events = pd.read_csv(task_root / "events.csv", dtype={"event_id": str, "case_id": str})
        event_slides = pd.read_csv(task_root / "event_slides.csv", dtype=str)
        for scale in args.scales:
            seed_predictions: list[pd.DataFrame] = []
            for seed in args.seeds:
                source = experiment_dir(args.results_root, task, scale, seed)
                event_oof = aggregate_events(
                    read_slide_oof(source), event_slides, events, args.aggregation
                )
                event_oof["seed"] = seed
                event_oof.to_csv(
                    args.output_dir / f"{task}_{scale}_seed{seed}_event_predictions.csv",
                    index=False,
                )
                seed_predictions.append(event_oof)
                per_seed_rows.append({
                    "task": task,
                    "scale": scale,
                    "seed": seed,
                    "aggregation": args.aggregation,
                    **metrics(event_oof, args.threshold),
                })

            combined = pd.concat(seed_predictions, ignore_index=True)
            if (combined.groupby("event_id").label.nunique() > 1).any():
                raise ValueError(f"{task}/{scale}: inconsistent event labels across seeds")
            ensemble = combined.groupby("event_id", as_index=False).agg(
                case_id=("case_id", "first"),
                label=("label", "first"),
                probability=("probability", "mean"),
                slide_count=("slide_count", "first"),
                seed_predictions=("seed", "nunique"),
            )
            if not ensemble.seed_predictions.eq(len(args.seeds)).all():
                raise ValueError(f"{task}/{scale}: incomplete seed coverage")
            ensemble["prediction"] = (ensemble.probability >= args.threshold).astype(int)
            ensemble.to_csv(
                args.output_dir / f"{task}_{scale}_seed_ensemble_event_predictions.csv",
                index=False,
            )
            seed_metrics = pd.DataFrame([
                row for row in per_seed_rows
                if row["task"] == task and row["scale"] == scale
            ])
            summary = {
                "task": task,
                "scale": scale,
                "seeds": len(args.seeds),
                "aggregation": args.aggregation,
                **metrics(ensemble, args.threshold),
            }
            for column in ("auroc", "pr_auc", "pr_auc_lift", "sensitivity", "specificity", "balanced_accuracy"):
                summary[f"seed_mean_{column}"] = float(seed_metrics[column].mean())
                summary[f"seed_std_{column}"] = float(seed_metrics[column].std(ddof=1))
            ensemble_rows.append(summary)

    pd.DataFrame(per_seed_rows).to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    summary = pd.DataFrame(ensemble_rows)
    summary.to_csv(args.output_dir / "slide_to_event_summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
