#!/usr/bin/env python3
"""Original-only, paired 3-fold LR diagnostic; reuse the completed 1e-4 reference.

Only initial LR changes. No augmentation generation or Stage2 training occurs.
This remains exploratory validation, not an independent test estimate.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import run_he_aug_full as workflow
from clam_view_training import training_config


REFERENCE = Path("results/he_manual_acr_aug_explore_v2_best/original_1view")
OUTPUT = Path("results/he_manual_acr_lr_check_v1")


def settings(fold: int, lr: float) -> SimpleNamespace:
    return SimpleNamespace(seed=1, fold=fold, view_seed=20260918, sampler="weighted",
        lr=lr, reg=1e-5, drop_out=.25, model_size="small", bag_weight=.7,
        no_inst_cluster=False, B=8, max_epochs=200, early_stopping=True,
        early_stop_patience=15, early_stop_min_epoch=10, early_stop_min_delta=1e-4,
        grad_clip=1., lr_factor=.5, lr_patience=5, min_lr=1e-6)


def check_report(directory: Path, fold: int, lr: float) -> dict:
    root = workflow.PROJECT / directory
    for filename in ("view_run_report.json", "best_checkpoint.pt", "val_event_predictions.csv"):
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Incomplete result; preserve/inspect before retry: {root / filename}")
    report = json.loads((root / "view_run_report.json").read_text())
    expected = {"training_protocol": "best_val_v2", "policy": "original", "total_views": 1,
        "technical_smoke": False, "fold": fold, "original_probability": 1.,
        "cohort_sha256": workflow.sha256(workflow.PROJECT / workflow.COHORT),
        "split_sha256": workflow.sha256(workflow.PROJECT / workflow.SPLIT_DIR / f"splits_{fold}.csv"),
        "training_config": training_config(settings(fold, lr))}
    different = [key for key, value in expected.items() if report.get(key) != value]
    if different:
        raise ValueError(f"Result is not a matched LR comparison ({different}): {root}")
    return report


def command(fold: int) -> list[str]:
    return [sys.executable, "-u", "train_clam_feature_views.py",
        "--cohort-csv", str(workflow.COHORT),
        "--split-csv", str(workflow.SPLIT_DIR / f"splits_{fold}.csv"),
        "--baseline-dir", str(workflow.BASELINE),
        "--policy", "original", "--total-views", "1", "--fold", str(fold),
        "--results-dir", str(OUTPUT / "lr_3e-5" / f"fold_{fold}"),
        "--training-protocol", "best_val_v2", "--lr", "3e-5",
        "--seed", "1", "--view-seed", "20260918", "--sampler", "weighted",
        "--max-epochs", "200", "--reg", "1e-5", "--drop-out", "0.25",
        "--model-size", "small", "--bag-weight", "0.7", "--B", "8",
        "--early-stop-patience", "15", "--early-stop-min-epoch", "10",
        "--early-stop-min-delta", "1e-4", "--grad-clip", "1",
        "--lr-factor", "0.5", "--lr-patience", "5", "--min-lr", "1e-6"]


def summarize() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    rows, pooled_rows = [], []
    reference_labels = {}
    def metrics(frame):
        p = frame.prob_positive.to_numpy(dtype=float)
        y = frame.label.to_numpy(dtype=int)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or set(y) != {0, 1}:
            raise ValueError("Invalid event probabilities or labels")
        return dict(event_average_precision=float(average_precision_score(y, p)),
                    event_roc_auc=float(roc_auc_score(y, p)),
                    event_brier=float(brier_score_loss(y, p)))
    for label, lr, base in (("lr_1e-4", 1e-4, REFERENCE), ("lr_3e-5", 3e-5, OUTPUT / "lr_3e-5")):
        pieces = []
        for fold in range(3):
            report = check_report(base / f"fold_{fold}", fold, lr)
            frame = pd.read_csv(workflow.PROJECT / base / f"fold_{fold}" / "val_event_predictions.csv",
                               dtype={"case_id": str, "event_id": str})
            if frame.duplicated(["case_id", "event_id"]).any():
                raise ValueError("Duplicate event within fold")
            labels = frame.set_index(["case_id", "event_id"]).label.sort_index()
            if label == "lr_1e-4":
                reference_labels[fold] = labels
            elif not labels.equals(reference_labels[fold]):
                raise ValueError(f"LR conditions have different validation events/labels: fold {fold}")
            rows.append(dict(condition=label, fold=fold, events=len(frame),
                positive_events=int(frame.label.sum()), **metrics(frame),
                best_epoch=report["best_epoch"], epochs_ran=report["epochs_ran"],
                optimizer_steps=report["optimizer_steps"]))
            pieces.append(frame)
        pooled = pd.concat(pieces, ignore_index=True)
        if pooled.duplicated(["case_id", "event_id"]).any():
            raise ValueError("An event appears in multiple folds")
        pooled_rows.append(dict(condition=label, events=len(pooled),
                               positive_events=int(pooled.label.sum()), **metrics(pooled)))
    table = pd.DataFrame(rows)
    reference_ap = table[table.condition == "lr_1e-4"].set_index("fold").event_average_precision
    table["delta_event_ap_vs_1e4"] = [row.event_average_precision - reference_ap.loc[row.fold]
                                     for row in table.itertuples()]
    destination = workflow.PROJECT / OUTPUT
    table.to_csv(destination / "lr_comparison_fold_metrics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(destination / "lr_comparison_pooled_metrics.csv", index=False)
    print("Fold event metrics (exploratory):\n" + table.to_string(index=False), flush=True)
    print("Pooled OOF metrics (not independent final performance):\n" +
          pd.DataFrame(pooled_rows).to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="Idle physical GPU index")
    parser.add_argument("--audit-only", action="store_true", help="Check reference/config/features without training")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    workflow.validate_inputs()
    for fold in range(3):
        check_report(REFERENCE / f"fold_{fold}", fold, 1e-4)
    destination = workflow.PROJECT / OUTPUT
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another LR-check runner is already active") from None
        print(f"Original-only LR check: 1e-4 reference reused; 3e-5 new training. GPU={args.gpu}", flush=True)
        for fold in range(3):
            relative = OUTPUT / "lr_3e-5" / f"fold_{fold}"
            folder = workflow.PROJECT / relative
            if (folder / "view_run_report.json").is_file():
                check_report(relative, fold, 3e-5)
                print(f"Already complete: {relative}", flush=True)
                continue
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f"Incomplete result: {folder}; preserve/inspect before retry")
            invocation = command(fold)
            workflow.run(invocation + ["--preflight-only", "--deep-feature-check"])
            if not args.audit_only:
                workflow.run(invocation)
                check_report(relative, fold, 3e-5)
        if not args.audit_only:
            summarize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
