#!/usr/bin/env python3
"""Create patient-grouped standard cross-validation splits.

Each fold trains on all other patient folds and validates on the remaining fold.
There is deliberately no separate test split: the held-out validation metrics
from all three rotations are the reported cross-validation results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


TASKS = {
    "task_smc_acr_binary_0r_vs_1r2r3r": (
        "dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv",
        "smc_cv_acr_0r_vs_1r2r3r_standard3",
    ),
    "task_smc_acr_binary_0r1r_vs_2r3r": (
        "dataset_csv/smc_acr_binary_0r1r_vs_2r3r.csv",
        "smc_cv_acr_0r1r_vs_2r3r_standard3",
    ),
    "task_smc_amr_binary_pamr0_vs_positive": (
        "dataset_csv/smc_amr_binary_pamr0_vs_positive.csv",
        "smc_cv_amr_pamr0_vs_positive_standard3",
    ),
    "task_smc_any_rejection_binary": (
        "dataset_csv/smc_any_rejection_binary.csv",
        "smc_cv_any_rejection_standard3",
    ),
    "task_smc_significant_rejection_binary": (
        "dataset_csv/smc_significant_rejection_binary.csv",
        "smc_cv_significant_rejection_standard3",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--folds", type=int, choices=(3, 5), default=3)
    return parser.parse_args()


def patient_summary(data: pd.DataFrame) -> pd.DataFrame:
    if set(data["label"].unique()) - {0, 1}:
        raise ValueError("Standard CV currently supports binary labels only")
    patients = data.groupby("case_id", as_index=False)["label"].max()
    return patients


def describe(data: pd.DataFrame, case_ids: set[str], fold: int, split: str) -> dict[str, int | str]:
    subset = data[data["case_id"].isin(case_ids)]
    return {
        "fold": fold,
        "split": split,
        "patients": subset["case_id"].nunique(),
        "positive_patients": int(subset.groupby("case_id")["label"].max().sum()),
        "bags": len(subset),
        "positive_bags": int(subset["label"].sum()),
    }


def main() -> int:
    args = parse_args()
    default_csv, default_dir = TASKS[args.task]
    csv_path = args.csv_path or Path(default_csv)
    default_dir = default_dir.replace("standard3", f"standard{args.folds}")
    output_dir = args.output_dir or Path("splits") / default_dir
    data = pd.read_csv(csv_path)
    required = {"case_id", "slide_id", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if data["slide_id"].duplicated().any():
        raise ValueError("slide_id must be unique")

    patients = patient_summary(data)
    if patients["label"].value_counts().min() < args.folds:
        raise ValueError(f"Each class needs at least {args.folds} patients for {args.folds}-fold CV")
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, int | str]] = []
    validation_patients: set[str] = set()

    for fold, (train_idx, val_idx) in enumerate(splitter.split(patients["case_id"], patients["label"])):
        train_cases = set(patients.iloc[train_idx]["case_id"])
        val_cases = set(patients.iloc[val_idx]["case_id"])
        if train_cases & val_cases:
            raise RuntimeError("Patient leakage while creating CV splits")
        validation_patients.update(val_cases)

        split_data = {
            "train": data[data["case_id"].isin(train_cases)]["slide_id"].tolist(),
            "val": data[data["case_id"].isin(val_cases)]["slide_id"].tolist(),
            # CLAM's split reader expects this column. --cv-validation never reads it.
            "test": [],
        }
        pd.DataFrame({name: pd.Series(values, dtype="object") for name, values in split_data.items()}).to_csv(
            output_dir / f"splits_{fold}.csv", index=False
        )
        report.append(describe(data, train_cases, fold, "train"))
        report.append(describe(data, val_cases, fold, "val"))

    if validation_patients != set(patients["case_id"]):
        raise RuntimeError("Validation folds did not cover every patient exactly once")
    pd.DataFrame(report).to_csv(output_dir / "fold_summary.csv", index=False)
    print(f"[OK] standard {args.folds}-fold CV: {args.task}: patients={len(patients)}, bags={len(data)} -> {output_dir}")
    print(pd.DataFrame(report).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
