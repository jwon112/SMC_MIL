#!/usr/bin/env python3
"""Build the gold-only baseline-to-future significant rejection task."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

ACR_GRADES = ("0R", "1R", "2R", "3R")
AMR_GRADES = ("pAMR0", "pAMR1", "pAMR1(I+)", "pAMR2")


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def load_events(label_xlsx: Path, gold_csv: Path) -> pd.DataFrame:
    labels = pd.read_excel(label_xlsx, sheet_name="Sheet1").iloc[:, :5].copy()
    labels.columns = ["patient_raw", "label_biopsy_date", "acr_grade", "amr_grade", "pathology_id"]
    labels = labels.dropna(subset=["patient_raw", "pathology_id"])
    labels["event_key"] = labels.pathology_id.map(normalize)
    labels["label_biopsy_date"] = pd.to_datetime(labels.label_biopsy_date, errors="coerce").dt.normalize()
    if labels.event_key.duplicated().any():
        raise ValueError("Sheet1 pathology IDs must be unique")

    bags = pd.read_csv(gold_csv)
    required = {"case_id", "slide_id", "event_id", "source_dataset", "biopsy_date"}
    missing = required.difference(bags.columns)
    if missing:
        raise ValueError(f"Gold CSV is missing columns: {sorted(missing)}")
    bags["event_key"] = bags.event_id.map(normalize)
    event_bags = bags.groupby("event_key", as_index=False).agg(
        case_id=("case_id", "first"), event_id=("event_id", "first"),
        biopsy_date=("biopsy_date", "first"), slide_count=("slide_id", "size"),
    )
    events = event_bags.merge(
        labels[["event_key", "label_biopsy_date", "acr_grade", "amr_grade"]],
        on="event_key", how="left", validate="one_to_one",
    )
    events["biopsy_date"] = pd.to_datetime(events.biopsy_date, errors="coerce").dt.normalize()
    if events[["label_biopsy_date", "acr_grade", "amr_grade"]].isna().any().any():
        raise ValueError("At least one gold event could not be joined to Sheet1")
    if (events.biopsy_date != events.label_biopsy_date).any():
        raise ValueError("Gold CSV and Sheet1 biopsy dates disagree")
    if not events.acr_grade.isin(ACR_GRADES).all() or not events.amr_grade.isin(AMR_GRADES).all():
        raise ValueError("Unexpected rejection grade in exact-match cohort")
    if events.duplicated(["case_id", "biopsy_date"]).any():
        raise ValueError("More than one gold event occurs for a patient on the same date")
    events = events.sort_values(["case_id", "biopsy_date", "event_key"]).reset_index(drop=True)
    events["event_index"] = events.groupby("case_id").cumcount() + 1
    events["acr_high_grade"] = events.acr_grade.isin(["2R", "3R"])
    events["amr_positive"] = events.amr_grade.ne("pAMR0")
    return events.drop(columns="label_biopsy_date")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-xlsx", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def build_patient_manifest(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["significant_rejection"] = events.acr_high_grade | events.amr_positive
    repeated_ids = events.groupby("case_id").size().loc[lambda value: value >= 2].index
    repeated = events[events.case_id.isin(repeated_ids)].copy()
    baselines = repeated[repeated.event_index.eq(1) & ~repeated.significant_rejection].copy()

    rows = []
    for baseline in baselines.itertuples(index=False):
        future = repeated[
            repeated.case_id.eq(baseline.case_id)
            & repeated.biopsy_date.gt(baseline.biopsy_date)
        ]
        future_acr = bool(future.acr_high_grade.any())
        future_amr = bool(future.amr_positive.any())
        positive = future_acr or future_amr
        positive_dates = future.loc[future.significant_rejection, "biopsy_date"]
        first_positive_date = positive_dates.min() if not positive_dates.empty else pd.NaT
        rows.append({
            "case_id": baseline.case_id,
            "baseline_event_id": baseline.event_id,
            "baseline_event_key": baseline.event_key,
            "baseline_biopsy_date": baseline.biopsy_date,
            "baseline_acr_grade": baseline.acr_grade,
            "baseline_amr_grade": baseline.amr_grade,
            "future_event_count": len(future),
            "last_biopsy_date": future.biopsy_date.max(),
            "observed_followup_days": int((future.biopsy_date.max() - baseline.biopsy_date).days),
            "future_acr_high": int(future_acr),
            "future_amr_positive": int(future_amr),
            "label": int(positive),
            "label_text": "future_significant_rejection" if positive else "no_observed_significant_rejection",
            "first_positive_date": first_positive_date,
            "days_to_first_positive": (
                int((first_positive_date - baseline.biopsy_date).days) if positive else ""
            ),
        })
    return pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)


def build_slide_manifest(gold_csv: Path, patients: pd.DataFrame) -> pd.DataFrame:
    bags = pd.read_csv(gold_csv)
    bags["event_key"] = bags.event_id.map(normalize)
    patient_columns = [
        "case_id", "baseline_event_key", "baseline_biopsy_date", "future_event_count",
        "observed_followup_days", "future_acr_high", "future_amr_positive", "label",
        "label_text", "days_to_first_positive",
    ]
    manifest = bags.merge(
        patients[patient_columns],
        left_on=["case_id", "event_key"],
        right_on=["case_id", "baseline_event_key"],
        how="inner",
        validate="many_to_one",
    )
    if manifest.slide_id.duplicated().any():
        raise ValueError("Baseline slide IDs must be unique")
    observed_cases = set(manifest.case_id)
    expected_cases = set(patients.case_id)
    if observed_cases != expected_cases:
        raise ValueError(f"Missing baseline slides for {len(expected_cases - observed_cases)} patient(s)")
    first = ["case_id", "slide_id", "label", "label_text"]
    return manifest[first + [column for column in manifest if column not in first]]


def write_splits(manifest: pd.DataFrame, patients: pd.DataFrame, output: Path, folds: int, seed: int) -> pd.DataFrame:
    if patients.label.value_counts().min() < folds:
        raise ValueError(f"Each class needs at least {folds} patients")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    split_dir = output / "splits" / f"smc_future_significant_rejection_gold{folds}"
    split_dir.mkdir(parents=True, exist_ok=True)
    report = []
    held_out_cases: set[str] = set()
    for fold, (train_index, test_index) in enumerate(splitter.split(patients.case_id, patients.label)):
        train_cases = set(patients.iloc[train_index].case_id)
        test_cases = set(patients.iloc[test_index].case_id)
        held_out_cases.update(test_cases)
        train_slides = manifest[manifest.case_id.isin(train_cases)].slide_id.tolist()
        test_slides = manifest[manifest.case_id.isin(test_cases)].slide_id.tolist()
        pd.DataFrame({
            "train": pd.Series(train_slides, dtype="object"),
            "val": pd.Series(dtype="object"),
            "test": pd.Series(test_slides, dtype="object"),
        }).to_csv(split_dir / f"splits_{fold}.csv", index=False)
        for name, cases, slides in (("train", train_cases, train_slides), ("test", test_cases, test_slides)):
            subset = patients[patients.case_id.isin(cases)]
            report.append({
                "fold": fold, "split": name, "patients": len(subset),
                "positive_patients": int(subset.label.sum()), "slide_bags": len(slides),
                "positive_slide_bags": int(manifest[manifest.slide_id.isin(slides)].label.sum()),
            })
    if held_out_cases != set(patients.case_id):
        raise RuntimeError("Outer folds did not cover every eligible patient exactly once")
    report_frame = pd.DataFrame(report)
    report_frame.to_csv(split_dir / "fold_summary.csv", index=False)
    return report_frame


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.label_xlsx, args.gold_csv)
    patients = build_patient_manifest(events)
    manifest = build_slide_manifest(args.gold_csv, patients)
    dataset_dir = args.output_dir / "dataset_csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(args.output_dir / "patient_manifest.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(
        dataset_dir / "smc_future_significant_rejection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = write_splits(manifest, patients, args.output_dir, args.folds, args.seed)
    print(
        f"[OK] future significant rejection: patients={len(patients)}, "
        f"positive={int(patients.label.sum())}, baseline_slide_bags={len(manifest)}"
    )
    print(report.to_string(index=False))
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
