#!/usr/bin/env python3
"""Build a fixed stain-specific ACR cohort and patient-grouped CV splits.

The stain label is a separate assertion from the ACR label. This script checks
internal consistency but cannot establish whether the source ACR labels are
clinically gold-standard or whether a filename rule is pathology-verified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stain-csv", type=Path, required=True)
    parser.add_argument("--task-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path,
                        help="UNI2 scale directory containing pt_files; missing bags are excluded.")
    parser.add_argument("--stain", choices=("HE", "IHC"), default="HE")
    parser.add_argument("--label-source", choices=("manual_only", "manual_or_filename"),
                        default="manual_only")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.feature_dir is not None and not (args.feature_dir / "pt_files").is_dir():
        parser.error("--feature-dir must contain pt_files/")
    return args


def read_table(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["slide_id"].eq("").any() or frame["slide_id"].duplicated().any():
        raise ValueError(f"{path}: blank or duplicate slide_id")
    return frame


def summarize(frame: pd.DataFrame, fold: int, split: str) -> dict[str, int | str]:
    patients = frame.groupby("case_id", sort=False)["label"].max()
    events = frame.drop_duplicates(["case_id", "event_id"])
    return {
        "fold": fold,
        "split": split,
        "slides": len(frame),
        "positive_slides": int(frame["label"].sum()),
        "patients": len(patients),
        "positive_patients": int(patients.sum()),
        "events": len(events),
        "positive_events": int(events["label"].sum()),
    }


def prepare(args: argparse.Namespace) -> dict[str, object]:
    task = read_table(args.task_csv, {"slide_id", "case_id", "event_id", "label", "source_dataset"})
    stain = read_table(args.stain_csv,
                       {"slide_id", "manual_label", "final_label", "reference_source"})
    if not task["label"].isin({"0", "1"}).all():
        raise ValueError("ACR task label must contain only 0 and 1")
    for key in ("case_id", "event_id", "source_dataset"):
        if task[key].eq("").any():
            raise ValueError(f"ACR task has blank {key}; patient/event audit would be unreliable")
    task["label"] = task["label"].astype(int)

    source_ok = stain["reference_source"].eq("manual_review")
    if args.label_source == "manual_or_filename":
        source_ok |= stain["reference_source"].eq("filename_rule")
    label_ok = stain["final_label"].eq(args.stain)
    if args.label_source == "manual_only":
        label_ok &= stain["manual_label"].eq(args.stain)
    selected_stain = stain.loc[source_ok & label_ok,
                               ["slide_id", "reference_source", "manual_label"]]
    selected_stain = selected_stain.rename(columns={"reference_source": "stain_label_source"})
    if ((selected_stain["stain_label_source"] == "manual_review")
            & (selected_stain["manual_label"] != args.stain)).any():
        raise ValueError("Manual stain labels disagree with final_label")

    cohort = task.merge(selected_stain, on="slide_id", how="inner", validate="one_to_one")
    cohort["stain_group"] = args.stain
    before_feature_check = len(cohort)
    if args.feature_dir is not None:
        has_feature = cohort["slide_id"].map(
            lambda sid: (args.feature_dir / "pt_files" / f"{sid}.pt").is_file()
        )
        cohort = cohort.loc[has_feature].copy()
    if cohort.empty:
        raise ValueError("No eligible slides after stain/task/feature intersection")

    event_groups = cohort.groupby(["case_id", "event_id"], sort=False)
    if event_groups["label"].nunique().gt(1).any():
        raise ValueError("An event has conflicting ACR labels")
    if cohort.groupby("event_id")["case_id"].nunique().gt(1).any():
        raise ValueError("event_id is shared by multiple patients; verify ID mapping")

    cohort = cohort.sort_values(["case_id", "event_id", "slide_id"]).reset_index(drop=True)
    patients = cohort.groupby("case_id", as_index=False)["label"].max()
    class_counts = patients["label"].value_counts()
    if len(class_counts) != 2 or class_counts.min() < args.folds:
        raise ValueError(f"Each patient-level class needs at least {args.folds} patients; got {class_counts.to_dict()}")

    output_files = [args.output_dir / "cohort.csv", args.output_dir / "fold_summary.csv",
                    args.output_dir / "audit.json"]
    output_files += [args.output_dir / f"splits_{fold}.csv" for fold in range(args.folds)]
    output_files += [args.output_dir / f"technical_train_slides_{fold}.txt"
                     for fold in range(args.folds)]
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite prior cohort/splits: {existing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_dir / "cohort.csv", index=False)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    summaries: list[dict[str, int | str]] = []
    seen_validation: set[str] = set()
    for fold, (train_indices, val_indices) in enumerate(
            splitter.split(patients["case_id"], patients["label"])):
        train_cases = set(patients.iloc[train_indices]["case_id"])
        val_cases = set(patients.iloc[val_indices]["case_id"])
        if train_cases & val_cases or seen_validation & val_cases:
            raise RuntimeError("Patient leakage or repeated validation patient")
        seen_validation |= val_cases
        train = cohort[cohort["case_id"].isin(train_cases)]
        val = cohort[cohort["case_id"].isin(val_cases)]
        if set(train["slide_id"]) & set(val["slide_id"]):
            raise RuntimeError("Slide leakage")
        split_frame = pd.DataFrame({
            "train": pd.Series(train["slide_id"].tolist(), dtype="object"),
            "val": pd.Series(val["slide_id"].tolist(), dtype="object"),
            "test": pd.Series(dtype="object"),
        })
        split_frame.to_csv(args.output_dir / f"splits_{fold}.csv", index=False)
        technical_ids = []
        for label in (0, 1):
            selected = train.loc[train["label"] == label, "slide_id"].head(2).tolist()
            if len(selected) != 2:
                raise ValueError(f"Fold {fold} has fewer than two train slides of class {label}")
            technical_ids.extend(selected)
        (args.output_dir / f"technical_train_slides_{fold}.txt").write_text(
            "\n".join(technical_ids) + "\n", encoding="utf-8"
        )
        summaries.extend((summarize(train, fold, "train"), summarize(val, fold, "val")))
    if seen_validation != set(patients["case_id"]):
        raise RuntimeError("Validation folds do not cover every patient")
    pd.DataFrame(summaries).to_csv(args.output_dir / "fold_summary.csv", index=False)

    audit = {
        "stain": args.stain,
        "label_source": args.label_source,
        "stain_csv": str(args.stain_csv),
        "task_csv": str(args.task_csv),
        "feature_dir": str(args.feature_dir) if args.feature_dir is not None else None,
        "feature_bags_verified_present": args.feature_dir is not None,
        "task_csv_slides": len(task),
        "stain_csv_slides": len(stain),
        "selected_stain_slides_before_task_intersection": len(selected_stain),
        "task_stain_intersection_before_feature_check": before_feature_check,
        "missing_feature_slides_excluded": before_feature_check - len(cohort),
        "task_acr_gold_provenance_verified": False,
        "manual_review_authority_verified": False,
        "filename_rule_pathology_verified": False,
        "seed": args.seed,
        "folds": args.folds,
        "total": summarize(cohort, -1, "all"),
        "by_stain_label_source": cohort["stain_label_source"].value_counts().to_dict(),
        "by_source_dataset": cohort["source_dataset"].value_counts().to_dict(),
        "fold_summary": summaries,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    audit = prepare(arguments())
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
