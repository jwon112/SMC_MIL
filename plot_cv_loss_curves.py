#!/usr/bin/env python3
"""Render per-fold train/validation loss curves from SMC CV worker logs."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUN_RE = re.compile(r"^\[RUN\] GPU \d+ \| (?P<task>[^|]+) \| (?P<scale>.+)$")
FOLD_RE = re.compile(r"^Training Fold (?P<fold>\d+)!")
TRAIN_RE = re.compile(r"^Epoch: (?P<epoch>\d+), train_loss: (?P<loss>[0-9.eE+-]+)")
VAL_RE = re.compile(
    r"^Val Set, val_loss: (?P<loss>[0-9.eE+-]+), val_error: (?P<error>[0-9.eE+-]+), auc: (?P<auc>[0-9.eE+-]+)"
)


@dataclass
class FoldCurve:
    train: list[tuple[int, float]] = field(default_factory=list)
    val: list[tuple[int, float, float]] = field(default_factory=list)


@dataclass
class ExperimentCurve:
    task: str
    scale: str
    folds: dict[int, FoldCurve] = field(default_factory=lambda: defaultdict(FoldCurve))

    @property
    def key(self) -> str:
        return f"{self.task}__{self.scale}"


def parse_log(path: Path) -> list[ExperimentCurve]:
    experiments: list[ExperimentCurve] = []
    current: ExperimentCurve | None = None
    current_fold: int | None = None
    latest_epoch: int | None = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        run_match = RUN_RE.match(line)
        if run_match:
            current = ExperimentCurve(run_match["task"].strip(), run_match["scale"].strip())
            experiments.append(current)
            current_fold = None
            latest_epoch = None
            continue
        if current is None:
            continue
        fold_match = FOLD_RE.match(line)
        if fold_match:
            current_fold = int(fold_match["fold"])
            latest_epoch = None
            continue
        if current_fold is None:
            continue
        train_match = TRAIN_RE.match(line)
        if train_match:
            latest_epoch = int(train_match["epoch"])
            current.folds[current_fold].train.append((latest_epoch, float(train_match["loss"])))
            continue
        val_match = VAL_RE.match(line)
        if val_match and latest_epoch is not None:
            current.folds[current_fold].val.append(
                (latest_epoch, float(val_match["loss"]), float(val_match["auc"]))
            )
    return experiments


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_experiment(experiment: ExperimentCurve, output_dir: Path) -> dict[str, object] | None:
    valid_folds = sorted(fold for fold, curve in experiment.folds.items() if curve.train or curve.val)
    if not valid_folds:
        return None

    fig, axes = plt.subplots(1, len(valid_folds), figsize=(5.2 * len(valid_folds), 4.2), sharey=False)
    if len(valid_folds) == 1:
        axes = [axes]
    records: list[dict[str, object]] = []

    for axis, fold in zip(axes, valid_folds):
        curve = experiment.folds[fold]
        if curve.train:
            epochs, losses = zip(*curve.train)
            axis.plot(epochs, losses, color="#2563eb", linewidth=1.8, label="Train loss")
        if curve.val:
            epochs, losses, aucs = zip(*curve.val)
            axis.plot(epochs, losses, color="#dc2626", linewidth=1.8, label="Validation loss")
            best_index = min(range(len(losses)), key=losses.__getitem__)
            axis.scatter(epochs[best_index], losses[best_index], color="#dc2626", s=28, zorder=3)
            records.append({
                "task": experiment.task,
                "scale": experiment.scale,
                "fold": fold,
                "epochs_logged": len(curve.train),
                "best_val_epoch": epochs[best_index],
                "best_val_loss": losses[best_index],
                "val_auc_at_best_loss": aucs[best_index],
            })
        axis.set_title(f"Fold {fold}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

    fig.suptitle(f"{experiment.task}\n{experiment.scale}", fontsize=11, y=1.03)
    fig.tight_layout()
    output_path = output_dir / f"{safe_name(experiment.key)}_loss.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(output_path), "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, action="append", required=True,
                        help="Worker log to parse; repeat for multiple logs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves: dict[str, ExperimentCurve] = {}
    for log_path in args.log:
        if not log_path.is_file():
            raise FileNotFoundError(f"Log not found: {log_path}")
        for experiment in parse_log(log_path):
            if experiment.key in curves:
                raise ValueError(f"Duplicate experiment in logs: {experiment.key}")
            curves[experiment.key] = experiment

    summary_rows: list[dict[str, object]] = []
    created = 0
    for experiment in sorted(curves.values(), key=lambda item: item.key):
        result = plot_experiment(experiment, args.output_dir)
        if result is None:
            print(f"[SKIP] no epoch curves: {experiment.key}")
            continue
        created += 1
        summary_rows.extend(result["records"])
        print(f"[OK] {result['path']}")

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "loss_curve_summary.csv", index=False)
    print(f"Created {created} curve image(s): {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
