#!/usr/bin/env python3
"""Create stain-restricted train-only weak-label cohorts for SMC CV.

The input is the weak-unique augmentation produced by
``build_smc_weak_unique_training.py``.  Each cohort filters both the training
and gold-validation bag IDs to a curated stain group, while preserving the
original patient-grouped fold assignment for every retained slide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK_SPLITS = {
    "smc_acr_binary_0r_vs_1r2r3r": "smc_cv_acr_0r_vs_1r2r3r_standard3",
    "smc_acr_binary_0r1r_vs_2r3r": "smc_cv_acr_0r1r_vs_2r3r_standard3",
    "smc_amr_binary_pamr0_vs_positive": "smc_cv_amr_pamr0_vs_positive_standard3",
    "smc_any_rejection_binary": "smc_cv_any_rejection_standard3",
}
INPUT_SUFFIX = "weak_unique_0to3"
COHORTS = {
    "mixed_known": {"HE", "IHC", "special_other"},
    "he_only": {"HE"},
    "non_he": {"IHC", "special_other"},
    "ihc_only": {"IHC"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curation-manifest", type=Path, required=True,
        help="slide_curation_manifest_curated.csv after manual stain review.",
    )
    parser.add_argument(
        "--weak-root", type=Path, required=True,
        help="Root created by build_smc_weak_unique_training.py.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--cohorts", nargs="+", choices=sorted(COHORTS),
        default=list(COHORTS), help="Cohorts to create (default: all).",
    )
    return parser.parse_args()


def read_curation(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"slide_id", "stain_group", "include_quality_usable"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Curation manifest missing columns: {sorted(missing)}")
    if frame.slide_id.duplicated().any():
        raise ValueError("Curation manifest contains duplicate slide_id values")
    frame = frame.set_index("slide_id")
    usable = frame["include_quality_usable"].astype(str).str.lower().eq("true")
    frame.loc[~usable, "stain_group"] = "exclude_quality"
    return frame[["stain_group"]]


def task_csv_name(task: str) -> str:
    return f"{task}_{INPUT_SUFFIX}.csv"


def build_cohort(task: str, cohort: str, curation: pd.DataFrame, args: argparse.Namespace) -> None:
    source_csv = args.weak_root / "dataset_csv" / task_csv_name(task)
    source_split_dir = args.weak_root / "splits" / f"{TASK_SPLITS[task]}_{INPUT_SUFFIX}"
    if not source_csv.is_file() or not source_split_dir.is_dir():
        raise FileNotFoundError(f"Missing weak-label inputs for {task}")

    data = pd.read_csv(source_csv)
    missing = set(data.slide_id).difference(curation.index)
    if missing:
        raise ValueError(f"{task}: {len(missing)} dataset slide IDs are absent from the curation manifest")
    data = data.join(curation, on="slide_id", validate="many_to_one")
    allowed = COHORTS[cohort]
    retained = data[data.stain_group.isin(allowed)].copy()
    if retained.empty:
        raise ValueError(f"{task}/{cohort}: no retained slides")
    retained_ids = set(retained.slide_id)

    output_csv_dir = args.output_root / "dataset_csv"
    output_csv_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_csv_dir / f"{task}_{INPUT_SUFFIX}_{cohort}.csv"
    retained.to_csv(output_csv, index=False)

    output_split_dir = args.output_root / "splits" / f"{TASK_SPLITS[task]}_{INPUT_SUFFIX}_{cohort}"
    output_split_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for fold in range(3):
        split = pd.read_csv(source_split_dir / f"splits_{fold}.csv")
        train = [slide_id for slide_id in split["train"].dropna().astype(str) if slide_id in retained_ids]
        val = [slide_id for slide_id in split["val"].dropna().astype(str) if slide_id in retained_ids]
        if not train or not val:
            raise ValueError(f"{task}/{cohort}/fold {fold}: empty train or validation split")
        out = pd.DataFrame({
            "train": pd.Series(train, dtype="object"),
            "val": pd.Series(val, dtype="object"),
            "test": pd.Series(dtype="object"),
        })
        out.to_csv(output_split_dir / f"splits_{fold}.csv", index=False)
        train_data = retained[retained.slide_id.isin(train)]
        val_data = retained[retained.slide_id.isin(val)]
        rows.append({
            "fold": fold,
            "train_bags": len(train),
            "train_positive_bags": int(train_data.label.sum()),
            "train_gold_bags": int(train_data.label_source.eq("gold_direct").sum()),
            "train_weak_bags": int(train_data.label_source.eq("weak_unique_date").sum()),
            "val_gold_bags": len(val),
            "val_positive_bags": int(val_data.label.sum()),
        })
    report = pd.DataFrame(rows)
    report.to_csv(output_split_dir / "fold_summary.csv", index=False)
    counts = retained.groupby(["stain_group", "label_source"], dropna=False).size()
    print(f"[OK] {task} / {cohort}: retained={len(retained)} -> {output_csv}")
    print(counts.to_string())
    print(report.to_string(index=False))


def main() -> int:
    args = parse_args()
    curation = read_curation(args.curation_manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for cohort in args.cohorts:
        for task in TASK_SPLITS:
            build_cohort(task, cohort, curation, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
