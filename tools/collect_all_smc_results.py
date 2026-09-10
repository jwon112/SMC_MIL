#!/usr/bin/env python3
"""Collect internal SMC CV and GSE290577 evaluation metrics into one table."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


EXPERIMENT_PATTERN = re.compile(
    r"^smc_(?P<task>acr_0r_vs_rest|acr_high_grade|amr_positive|any_rejection|significant_rejection|future_significant)_"
    r"(?P<level>l[0-3])_(?P<mpp>0p25|0p50|1p00|2p00)mpp_(?P<magnification>40x|20x|10x|5x)_"
    r"uni2_clamsb(?P<variant>.+)_s1$"
)
TASK_NAMES = {
    "acr_0r_vs_rest": "acr_any",
    "acr_high_grade": "acr_high",
    "amr_positive": "amr_positive",
    "any_rejection": "any_rejection",
    "significant_rejection": "significant_rejection",
    "future_significant": "future_significant_rejection",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smc_all_results"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def metadata(experiment: str) -> dict[str, object]:
    match = EXPERIMENT_PATTERN.match(experiment)
    if not match:
        return {
            "experiment": experiment,
            "task": "",
            "folds": "",
            "training_set": "",
            "stain_cohort": "",
            "magnification": "",
            "mpp": "",
            "variant": "",
        }
    values = match.groupdict()
    variant = values["variant"].lstrip("_")
    fold_match = re.search(r"cv(?P<folds>[345])val", variant)
    if fold_match is None:
        fold_match = re.search(r"gold(?P<folds>[345])cv", variant)
    stain_cohort = next(
        (name for name in ("mixed_known", "he_only", "non_he", "ihc_only") if name in variant),
        "all_stains",
    )
    return {
        "experiment": experiment,
        "task": TASK_NAMES[values["task"]],
        "folds": int(fold_match.group("folds")) if fold_match else "",
        "training_set": "gold+weak" if "weakunique" in variant else "gold_only",
        "stain_cohort": stain_cohort,
        "magnification": values["magnification"],
        "mpp": values["mpp"].replace("p", "."),
        "variant": variant,
    }


def positive_probability(value: object) -> float:
    probabilities = np.asarray(value, dtype=float).reshape(-1)
    if probabilities.size != 2:
        raise ValueError(f"Expected two class probabilities, got {probabilities.shape}")
    return float(probabilities[1])


def fold_metrics(experiment_dir: Path, threshold: float) -> pd.DataFrame:
    rows = []
    for result_path in sorted(experiment_dir.glob("split_*_results.pkl")):
        fold = int(result_path.stem.split("_")[1])
        with result_path.open("rb") as handle:
            results = pickle.load(handle)
        labels = np.asarray([int(item["label"]) for item in results.values()])
        probabilities = np.asarray([positive_probability(item["prob"]) for item in results.values()])
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else np.nan
        specificity = tn / (tn + fp) if tn + fp else np.nan
        rows.append({
            "fold": fold,
            "n": len(labels),
            "positive_n": int(labels.sum()),
            "negative_n": int((labels == 0).sum()),
            "auroc": roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 else np.nan,
            "pr_auc": average_precision_score(labels, probabilities) if labels.sum() else np.nan,
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "threshold": threshold,
        })
    return pd.DataFrame(rows)


def collect_internal(results_root: Path, threshold: float) -> list[dict[str, object]]:
    collected = []
    for experiment_dir in sorted(results_root.iterdir()):
        summary_path = experiment_dir / "summary.csv"
        if not experiment_dir.is_dir() or not summary_path.is_file():
            continue
        info = metadata(experiment_dir.name)
        if not info["task"]:
            continue
        summary = pd.read_csv(summary_path)
        metrics = fold_metrics(experiment_dir, threshold)
        row = {
            **info,
            "evaluation_scope": "internal_cv",
            "evaluation_run": "internal_cv",
            "cohort": "internal_cv",
            "metric_unit": "slide_bag",
            "patients": "",
            "n": int(metrics["n"].sum()) if not metrics.empty else "",
            "positive_n": int(metrics["positive_n"].sum()) if not metrics.empty else "",
            "negative_n": int(metrics["negative_n"].sum()) if not metrics.empty else "",
            "accuracy": float(summary["cv_val_acc"].mean()) if "cv_val_acc" in summary else "",
            "threshold": threshold,
            "source_path": str(summary_path),
        }
        if not metrics.empty:
            for column in ("auroc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity"):
                row[column] = float(metrics[column].mean())
                row[f"{column}_std"] = float(metrics[column].std(ddof=1))
        elif "cv_val_auc" in summary:
            row["auroc"] = float(summary["cv_val_auc"].mean())
            row["auroc_std"] = float(summary["cv_val_auc"].std(ddof=1))
        patient_metrics_path = experiment_dir / "patient_level_oof" / "pooled_metrics.csv"
        if info["task"] == "future_significant_rejection" and patient_metrics_path.is_file():
            patient_metrics = pd.read_csv(patient_metrics_path).iloc[0]
            row["metric_unit"] = "patient"
            row["n"] = int(patient_metrics["patients"])
            row["positive_n"] = int(patient_metrics["positive_patients"])
            row["negative_n"] = int(patient_metrics["negative_patients"])
            for column in ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "threshold"):
                row[column] = float(patient_metrics[column])
            row["source_path"] = str(patient_metrics_path)
        collected.append(row)
    return collected


def checkpoint_experiment(summary_path: Path) -> str:
    config_path = summary_path.parent / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = config.get("checkpoint_dir")
        if checkpoint:
            return Path(checkpoint).name
    parent = summary_path.parent.name
    return parent if parent.startswith("smc_") else parent.removesuffix("mpp")


def evaluation_run(summary_path: Path, results_root: Path) -> str:
    external_root = results_root / "gse290577_external"
    try:
        relative = summary_path.relative_to(external_root)
        return relative.parts[0]
    except ValueError:
        return summary_path.parent.parent.name


def collect_external(results_root: Path) -> tuple[list[dict[str, object]], int]:
    candidates: dict[tuple[str, str, str], tuple[float, dict[str, object]]] = {}
    duplicate_count = 0
    summary_paths = set(results_root.rglob("external_summary.csv"))
    summary_paths.update(results_root.rglob("all_external_summary.csv"))
    summary_paths.update(results_root.rglob("combined_external_summary.csv"))
    for summary_path in sorted(summary_paths):
        frame = pd.read_csv(summary_path)
        for source in frame.to_dict("records"):
            source_experiment = str(source.get("experiment", ""))
            experiment = source_experiment or checkpoint_experiment(summary_path)
            info = metadata(experiment)
            task = str(source.get("task", info["task"]))
            cohort = str(source.get("cohort", ""))
            key = (experiment, task, cohort)
            row = {
                **info,
                **source,
                "experiment": experiment,
                "task": task,
                "evaluation_scope": "external_test",
                "evaluation_run": evaluation_run(summary_path, results_root),
                "source_path": str(summary_path),
            }
            modified = summary_path.stat().st_mtime
            previous = candidates.get(key)
            if previous is not None:
                duplicate_count += 1
            if previous is None or modified > previous[0]:
                candidates[key] = (modified, row)
    return [item[1] for item in candidates.values()], duplicate_count


def main() -> int:
    args = arguments()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between zero and one")
    if not args.results_root.is_dir():
        raise FileNotFoundError(args.results_root)

    internal = collect_internal(args.results_root, args.threshold)
    external, duplicates = collect_external(args.results_root)
    rows = internal + external
    if not rows:
        raise ValueError(f"No compatible result files found under {args.results_root}")

    preferred = [
        "evaluation_scope", "evaluation_run", "experiment", "task", "folds",
        "training_set", "stain_cohort", "magnification", "mpp", "cohort", "metric_unit",
        "patients", "n", "positive_n", "negative_n", "auroc", "auroc_std",
        "pr_auc", "pr_auc_std", "sensitivity", "sensitivity_std", "specificity",
        "specificity_std", "balanced_accuracy", "balanced_accuracy_std", "accuracy",
        "threshold", "variant", "source_path",
    ]
    frame = pd.DataFrame(rows)
    columns = [column for column in preferred if column in frame] + [
        column for column in frame if column not in preferred
    ]
    frame = frame[columns].sort_values(
        ["evaluation_scope", "task", "experiment", "cohort"], na_position="last"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "all_results_master.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Internal CV experiments: {len(internal)}")
    print(f"External evaluation rows: {len(external)}")
    print(f"Duplicate external rows ignored: {duplicates}")
    print(f"[OK] Master results: {output_path}")
    print("\nRows by evaluation scope/cohort:")
    print(frame.groupby(["evaluation_scope", "cohort"], dropna=False).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
