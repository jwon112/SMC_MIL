#!/usr/bin/env python3
"""Create patient-grouped outer 3-fold CV splits with an inner validation set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


TASKS = {
    "task_smc_acr_binary_0r_vs_1r2r3r": ("dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv", "smc_cv_acr_0r_vs_1r2r3r"),
    "task_smc_acr_binary_0r1r_vs_2r3r": ("dataset_csv/smc_acr_binary_0r1r_vs_2r3r.csv", "smc_cv_acr_0r1r_vs_2r3r"),
    "task_smc_amr_binary_pamr0_vs_positive": ("dataset_csv/smc_amr_binary_pamr0_vs_positive.csv", "smc_cv_amr_pamr0_vs_positive"),
    "task_smc_any_rejection_binary": ("dataset_csv/smc_any_rejection_binary.csv", "smc_cv_any_rejection"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--inner-val-frac", type=float, default=0.20)
    parser.add_argument("--no-inner-val", action="store_true",
                        help="Use all outer-training patients for training; write an empty validation column.")
    return parser.parse_args()


def patient_summary(data: pd.DataFrame) -> pd.DataFrame:
    if set(data["label"].unique()) - {0, 1}:
        raise ValueError("Nested CV supports binary labels only")
    patient_labels = data.groupby("case_id", as_index=False)["label"].max()
    if patient_labels["label"].value_counts().min() < 6:
        raise ValueError("Each class needs at least six patients for outer 3-fold plus inner validation")
    return patient_labels


def describe(data: pd.DataFrame, case_ids: set[str], fold: int, split: str) -> dict[str, int | str]:
    subset = data[data["case_id"].isin(case_ids)]
    return {
        "fold": fold,
        "split": split,
        "patients": subset["case_id"].nunique(),
        "positive_patients": subset.groupby("case_id")["label"].max().sum(),
        "bags": len(subset),
        "positive_bags": int(subset["label"].sum()),
    }


def main() -> int:
    args = parse_args()
    default_csv, default_dir = TASKS[args.task]
    csv_path = args.csv_path or Path(default_csv)
    output_dir = args.output_dir or Path("splits") / default_dir
    data = pd.read_csv(csv_path)
    required = {"case_id", "slide_id", "label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if data["slide_id"].duplicated().any():
        raise ValueError("slide_id must be unique")

    patients = patient_summary(data)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, int | str]] = []
    held_out_patients: set[str] = set()

    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer.split(patients["case_id"], patients["label"])):
        outer_train = patients.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = patients.iloc[outer_test_idx].reset_index(drop=True)
        if args.no_inner_val:
            train_cases = set(outer_train["case_id"])
            val_cases: set[str] = set()
        else:
            inner = StratifiedShuffleSplit(n_splits=1, test_size=args.inner_val_frac, random_state=args.seed + fold)
            inner_train_idx, inner_val_idx = next(inner.split(outer_train["case_id"], outer_train["label"]))
            train_cases = set(outer_train.iloc[inner_train_idx]["case_id"])
            val_cases = set(outer_train.iloc[inner_val_idx]["case_id"])
        test_cases = set(outer_test["case_id"])
        if train_cases & val_cases or train_cases & test_cases or val_cases & test_cases:
            raise RuntimeError("Patient leakage while creating splits")
        held_out_patients.update(test_cases)

        split_data = {
            "train": data[data["case_id"].isin(train_cases)]["slide_id"].tolist(),
            "val": data[data["case_id"].isin(val_cases)]["slide_id"].tolist(),
            "test": data[data["case_id"].isin(test_cases)]["slide_id"].tolist(),
        }
        pd.DataFrame({name: pd.Series(values) for name, values in split_data.items()}).to_csv(
            output_dir / f"splits_{fold}.csv", index=False
        )
        for split, case_ids in (("train", train_cases), ("val", val_cases), ("test", test_cases)):
            report.append(describe(data, case_ids, fold, split))

    if held_out_patients != set(patients["case_id"]):
        raise RuntimeError("Outer folds did not cover every patient exactly once")
    pd.DataFrame(report).to_csv(output_dir / "fold_summary.csv", index=False)
    print(f"[OK] {args.task}: patients={len(patients)}, bags={len(data)} -> {output_dir}")
    print(pd.DataFrame(report).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
