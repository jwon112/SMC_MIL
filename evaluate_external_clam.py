#!/usr/bin/env python3
"""Evaluate a CLAM checkpoint ensemble on one or more external cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score

from models.model_clam import CLAM_SB


TASK_LABELS = {
    "acr_any": "acr_any_label",
    "acr_high": "acr_high_label",
    "amr_positive": "amr_positive_label",
    "any_rejection": "any_rejection_label",
}


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else float("nan")
    specificity = float(tn / (tn + fp)) if tn + fp else float("nan")
    return {
        "n": len(labels), "positive_n": int(labels.sum()), "negative_n": int((labels == 0).sum()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float((sensitivity + specificity) / 2) if np.isfinite(sensitivity) and np.isfinite(specificity) else float("nan"),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.sum() else float("nan"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "threshold": threshold,
    }


def bootstrap_intervals(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray, threshold: float, repetitions: int, seed: int) -> dict[str, float]:
    if repetitions <= 0:
        return {}
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values: dict[str, list[float]] = {name: [] for name in ("auroc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity")}
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for _ in range(repetitions):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        result = metrics(labels[indices], probabilities[indices], threshold)
        for name in values:
            value = float(result[name])
            if np.isfinite(value):
                values[name].append(value)
    intervals = {}
    for name, samples in values.items():
        if samples:
            intervals[f"{name}_ci_low"] = float(np.quantile(samples, 0.025))
            intervals[f"{name}_ci_high"] = float(np.quantile(samples, 0.975))
    return intervals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cohort", nargs=3, action="append", metavar=("NAME", "MANIFEST", "FEAT_DIR"), required=True)
    parser.add_argument("--task", choices=sorted(TASK_LABELS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embed-dim", type=int, default=1536)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    checkpoints = sorted(args.checkpoint_dir.glob("s_*_checkpoint.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No s_*_checkpoint.pt in {args.checkpoint_dir}")
    models = []
    for checkpoint in checkpoints:
        model = CLAM_SB(size_arg=args.model_size, dropout=args.dropout, n_classes=2, embed_dim=args.embed_dim)
        state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
        # Some training runs serialize buffers owned by the instance-loss module.
        # They are not model parameters and are absent from the inference model.
        state_dict.pop("instance_loss_fn.labels", None)
        model.load_state_dict(state_dict, strict=True)
        models.append(model.eval().to(device))
    print(f"Loaded {len(models)} fold checkpoint(s)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    label_column = TASK_LABELS[args.task]
    for cohort_name, manifest_name, feat_name in args.cohort:
        manifest, feat_dir = Path(manifest_name), Path(feat_name)
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            source_rows = list(csv.DictReader(handle))
        predictions = []
        for row in source_rows:
            raw_label = row.get(label_column, "").strip()
            if raw_label == "":
                continue
            bag_path = feat_dir / "pt_files" / f"{row['slide_id']}.pt"
            if not bag_path.is_file():
                raise FileNotFoundError(f"Missing feature bag: {bag_path}")
            features = torch.load(bag_path, map_location=device, weights_only=True).float().to(device)
            fold_probabilities = []
            with torch.inference_mode():
                for model in models:
                    _, probability, _, _, _ = model(features)
                    fold_probabilities.append(float(probability[0, 1].cpu()))
            probability = float(np.mean(fold_probabilities))
            prediction = dict(row)
            prediction.update({
                "task": args.task, "label": int(float(raw_label)), "probability": probability,
                "prediction": int(probability >= args.threshold),
                **{f"probability_fold_{index}": value for index, value in enumerate(fold_probabilities)},
            })
            predictions.append(prediction)
        if not predictions:
            raise ValueError(f"No labeled rows for task {args.task} in {manifest}")
        output_csv = args.output_dir / f"{cohort_name}_predictions.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
        labels = np.asarray([row["label"] for row in predictions], dtype=int)
        probabilities = np.asarray([row["probability"] for row in predictions], dtype=float)
        groups = np.asarray([row.get("patient_id") or row.get("case_id") or row["slide_id"] for row in predictions])
        result = {"cohort": cohort_name, "task": args.task, "patients": len(np.unique(groups)), **metrics(labels, probabilities, args.threshold)}
        result.update(bootstrap_intervals(labels, probabilities, groups, args.threshold, args.bootstrap, args.seed))
        summaries.append(result)
        print(f"[OK] {cohort_name}: n={result['n']}, positive={result['positive_n']}, AUROC={result['auroc']:.3f}, PR-AUC={result['pr_auc']:.3f}")
    summary_csv = args.output_dir / "external_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(f"Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
