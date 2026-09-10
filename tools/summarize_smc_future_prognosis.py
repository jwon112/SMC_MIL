#!/usr/bin/env python3
"""Aggregate future-rejection slide predictions into patient-level OOF metrics."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def positive_probability(value: object) -> float:
    probabilities = np.asarray(value, dtype=float).reshape(-1)
    if probabilities.size != 2:
        raise ValueError(f"Expected binary probabilities, got {probabilities.shape}")
    return float(probabilities[1])


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame.probability.to_numpy(dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "patients": len(frame), "positive_patients": int(labels.sum()),
        "negative_patients": int((labels == 0).sum()),
        "auroc": roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 else np.nan,
        "pr_auc": average_precision_score(labels, probabilities) if labels.sum() else np.nan,
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "threshold": threshold, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def bootstrap(frame: pd.DataFrame, threshold: float, repetitions: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy")}
    for _ in range(repetitions):
        sampled = frame.iloc[rng.integers(0, len(frame), len(frame))]
        result = metrics(sampled, threshold)
        for name in values:
            value = result[name]
            if np.isfinite(value):
                values[name].append(float(value))
    intervals = {}
    for name, samples in values.items():
        if samples:
            intervals[f"{name}_ci_low"] = float(np.quantile(samples, 0.025))
            intervals[f"{name}_ci_high"] = float(np.quantile(samples, 0.975))
    return intervals


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")
    manifest = pd.read_csv(args.manifest)
    required = {"case_id", "slide_id", "label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    slide_rows = []
    for path in sorted(args.result_dir.glob("split_*_results.pkl")):
        fold = int(path.stem.split("_")[1])
        with path.open("rb") as handle:
            predictions = pickle.load(handle)
        for slide_id, result in predictions.items():
            slide_rows.append({
                "fold": fold, "slide_id": str(slide_id),
                "result_label": int(result["label"]),
                "probability": positive_probability(result["prob"]),
            })
    if not slide_rows:
        raise FileNotFoundError(f"No split_*_results.pkl files in {args.result_dir}")
    slides = pd.DataFrame(slide_rows).merge(
        manifest[["case_id", "slide_id", "label"]], on="slide_id", how="left", validate="one_to_one"
    )
    if slides[["case_id", "label"]].isna().any().any():
        raise ValueError("At least one prediction could not be joined to the task manifest")
    if not slides.result_label.eq(slides.label).all():
        raise ValueError("Prediction labels disagree with the task manifest")
    patients = slides.groupby(["case_id", "label"], as_index=False).agg(
        fold=("fold", "first"), probability=("probability", "mean"), slide_bags=("slide_id", "size")
    )
    if slides.groupby("case_id").fold.nunique().max() != 1:
        raise RuntimeError("A patient appears in more than one held-out fold")

    fold_rows = []
    for fold, frame in patients.groupby("fold"):
        fold_rows.append({"fold": int(fold), **metrics(frame, args.threshold)})
    pooled = {"scope": "pooled_oof", **metrics(patients, args.threshold)}
    pooled.update(bootstrap(patients, args.threshold, args.bootstrap, args.seed))

    output = args.output_dir or args.result_dir / "patient_level_oof"
    output.mkdir(parents=True, exist_ok=True)
    slides.to_csv(output / "slide_oof_predictions.csv", index=False, encoding="utf-8-sig")
    patients.to_csv(output / "patient_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([pooled]).to_csv(output / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([pooled]).to_string(index=False))
    print(f"[OK] Patient-level OOF results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
