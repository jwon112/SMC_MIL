#!/usr/bin/env python3
"""Select internal-OOF operating thresholds and transfer them to GSE290577."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score


PRIMARY_CANDIDATES = {
    ("acr_high", "seed_ensemble", "40x"),
    ("amr_positive", "seed_scale_ensemble", "40x+20x+10x+5x"),
    ("significant_rejection", "seed_ensemble", "40x"),
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
        default=Path("results/smc_prediction_ensembles/threshold_calibration"),
    )
    parser.add_argument("--target-sensitivity", type=float, nargs="+", default=(0.80, 0.90))
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
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
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "threshold": float(threshold),
    }


def candidate_thresholds(probabilities: np.ndarray) -> np.ndarray:
    unique = np.unique(probabilities)
    # The extra value represents the all-negative classifier.
    return np.r_[unique, np.nextafter(unique.max(), np.inf)]


def choose_youden(frame: pd.DataFrame) -> float:
    options = []
    for threshold in candidate_thresholds(frame["probability"].to_numpy(float)):
        result = metrics(frame, float(threshold))
        options.append((result["sensitivity"] + result["specificity"] - 1, threshold))
    # Prefer a higher threshold when operating points tie, to reduce false positives.
    return float(max(options, key=lambda item: (item[0], item[1]))[1])


def choose_target_sensitivity(frame: pd.DataFrame, target: float) -> float:
    valid = []
    for threshold in candidate_thresholds(frame["probability"].to_numpy(float)):
        result = metrics(frame, float(threshold))
        if result["sensitivity"] >= target:
            valid.append(threshold)
    if not valid:
        return float(frame["probability"].min())
    return float(max(valid))


def calibration_rules(frame: pd.DataFrame, baseline: float, targets: tuple[float, ...]) -> list[tuple[str, float]]:
    rules = [("fixed_0p5", baseline), ("youden_internal_oof", choose_youden(frame))]
    for target in targets:
        label = f"target_sensitivity_{target:.0%}".replace("%", "pct")
        rules.append((label, choose_target_sensitivity(frame, target)))
    return rules


def validate(frame: pd.DataFrame) -> None:
    required = {"evaluation_scope", "task", "method", "scales", "unit_id", "label", "probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for values, group in frame.groupby(["evaluation_scope", "task", "method", "scales"], sort=False):
        if group["unit_id"].duplicated().any():
            raise RuntimeError(f"Duplicate units in {'/'.join(map(str, values))}")
        if group["label"].nunique() != 2:
            raise RuntimeError(f"Both labels are required in {'/'.join(map(str, values))}")


def main() -> int:
    args = parse_args()
    if not args.predictions_csv.is_file():
        raise FileNotFoundError(args.predictions_csv)
    if not 0 <= args.baseline_threshold <= 1:
        raise ValueError("--baseline-threshold must be between zero and one")
    if any(not 0 < value <= 1 for value in args.target_sensitivity):
        raise ValueError("--target-sensitivity values must be in (0, 1]")

    raw = pd.read_csv(args.predictions_csv, dtype={"unit_id": str})
    frame = raw[raw["method"].isin(("seed_ensemble", "seed_scale_ensemble"))].copy()
    if frame.empty:
        raise ValueError("No ensemble prediction rows found")
    validate(frame)

    internal = frame[frame["evaluation_scope"].eq("internal_oof")]
    external = frame[frame["evaluation_scope"].eq("external_test")]
    calibration_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    for (task, method, scales), internal_group in internal.groupby(["task", "method", "scales"], sort=False):
        primary = (task, method, scales) in PRIMARY_CANDIDATES
        for rule, threshold in calibration_rules(
            internal_group, args.baseline_threshold, tuple(args.target_sensitivity)
        ):
            calibration_rows.append({
                "task": task,
                "method": method,
                "scales": scales,
                "primary_candidate": primary,
                "selection_rule": rule,
                "selection_data": "internal_oof",
                **metrics(internal_group, threshold),
            })
            matched_external = external[
                external["task"].eq(task)
                & external["method"].eq(method)
                & external["scales"].eq(scales)
            ]
            for cohort, external_group in matched_external.groupby("cohort", sort=False):
                transfer_rows.append({
                    "task": task,
                    "method": method,
                    "scales": scales,
                    "cohort": cohort,
                    "primary_candidate": primary,
                    "selection_rule": rule,
                    "selection_data": "internal_oof",
                    **metrics(external_group, threshold),
                })

    calibration = pd.DataFrame(calibration_rows)
    transfer = pd.DataFrame(transfer_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration.sort_values(["primary_candidate", "task", "method", "scales", "selection_rule"], ascending=[False, True, True, True, True]).to_csv(
        args.output_dir / "internal_threshold_calibration.csv", index=False, encoding="utf-8-sig"
    )
    transfer.sort_values(["primary_candidate", "task", "method", "scales", "cohort", "selection_rule"], ascending=[False, True, True, True, True, True]).to_csv(
        args.output_dir / "external_threshold_transfer.csv", index=False, encoding="utf-8-sig"
    )
    calibration[calibration["primary_candidate"]].to_csv(
        args.output_dir / "primary_candidate_thresholds.csv", index=False, encoding="utf-8-sig"
    )
    transfer[transfer["primary_candidate"]].to_csv(
        args.output_dir / "primary_candidate_external_transfer.csv", index=False, encoding="utf-8-sig"
    )

    print(f"[OK] ensemble configurations calibrated: {len(calibration_rows) // (2 + len(args.target_sensitivity))}")
    print(f"[OK] internal calibration: {args.output_dir / 'internal_threshold_calibration.csv'}")
    print(f"[OK] external transfer: {args.output_dir / 'external_threshold_transfer.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
