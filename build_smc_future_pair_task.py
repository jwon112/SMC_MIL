#!/usr/bin/env python3
"""Build patient-grouped biopsy pairs for future significant-rejection prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from build_smc_future_rejection_task import load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-xlsx", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def build_pairs(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = events.copy()
    events["significant_rejection"] = events.acr_high_grade | events.amr_positive
    rows: list[dict[str, object]] = []

    for case_id, patient in events.groupby("case_id", sort=True):
        patient = patient.sort_values(["biopsy_date", "event_key"]).reset_index(drop=True)
        if len(patient) < 2:
            continue

        # A source event is at risk only before the patient's first significant event.
        # Every later observed event is a target, so delta_days defines the horizon.
        for source_index, source in patient.iterrows():
            if bool(source.significant_rejection):
                break
            for target_index in range(source_index + 1, len(patient)):
                target = patient.iloc[target_index]
                delta_days = int((target.biopsy_date - source.biopsy_date).days)
                if delta_days <= 0:
                    raise ValueError(f"Non-positive biopsy interval for patient {case_id}")
                rows.append(
                    {
                        "pair_id": f"pair_{len(rows) + 1:06d}",
                        "case_id": case_id,
                        "source_event_id": source.event_id,
                        "source_event_key": source.event_key,
                        "source_event_index": int(source.event_index),
                        "source_biopsy_date": source.biopsy_date,
                        "source_acr_grade": source.acr_grade,
                        "source_amr_grade": source.amr_grade,
                        "source_slide_count": int(source.slide_count),
                        "target_event_id": target.event_id,
                        "target_event_key": target.event_key,
                        "target_event_index": int(target.event_index),
                        "target_biopsy_date": target.biopsy_date,
                        "target_acr_grade": target.acr_grade,
                        "target_amr_grade": target.amr_grade,
                        "delta_days": delta_days,
                        "label": int(target.significant_rejection),
                        "label_text": (
                            "future_significant_rejection"
                            if bool(target.significant_rejection)
                            else "future_no_significant_rejection"
                        ),
                    }
                )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No eligible future biopsy pairs were generated")
    if pairs.pair_id.duplicated().any():
        raise RuntimeError("Pair IDs are not unique")

    patients = (
        pairs.groupby("case_id", as_index=False)
        .agg(
            pairs=("pair_id", "size"),
            source_events=("source_event_key", "nunique"),
            target_events=("target_event_key", "nunique"),
            positive_pairs=("label", "sum"),
            first_source_date=("source_biopsy_date", "min"),
            last_target_date=("target_biopsy_date", "max"),
        )
    )
    patients["label"] = patients.positive_pairs.gt(0).astype(int)
    patients["observed_days"] = (
        patients.last_target_date - patients.first_source_date
    ).dt.days.astype(int)
    return pairs, patients


def write_splits(
    pairs: pd.DataFrame,
    patients: pd.DataFrame,
    output_dir: Path,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    counts = patients.label.value_counts()
    if len(counts) != 2 or counts.min() < folds:
        raise ValueError(f"Each patient class needs at least {folds} members: {counts.to_dict()}")

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = patients[["case_id", "label"]].copy()
    assignments["held_out_fold"] = -1
    for fold, (_, test_index) in enumerate(splitter.split(patients.case_id, patients.label)):
        assignments.loc[test_index, "held_out_fold"] = fold
    if assignments.held_out_fold.lt(0).any():
        raise RuntimeError("At least one patient was not assigned to a held-out fold")

    rows: list[dict[str, int | str]] = []
    for fold in range(folds):
        for split, selected in (
            ("train", assignments.held_out_fold.ne(fold)),
            ("test", assignments.held_out_fold.eq(fold)),
        ):
            case_ids = set(assignments.loc[selected, "case_id"])
            patient_part = patients[patients.case_id.isin(case_ids)]
            pair_part = pairs[pairs.case_id.isin(case_ids)]
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "patients": len(patient_part),
                    "positive_patients": int(patient_part.label.sum()),
                    "pairs": len(pair_part),
                    "positive_pairs": int(pair_part.label.sum()),
                    "source_events": int(pair_part.source_event_key.nunique()),
                    "target_events": int(pair_part.target_event_key.nunique()),
                    "positive_target_events": int(
                        pair_part.loc[pair_part.label.eq(1), "target_event_key"].nunique()
                    ),
                }
            )

    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        split_dir / f"future_pair_patient_folds_{folds}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = pd.DataFrame(rows)
    report.to_csv(
        split_dir / f"future_pair_fold_summary_{folds}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return report


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.label_xlsx, args.gold_csv)
    pairs, patients = build_pairs(events)
    report = write_splits(pairs, patients, args.output_dir, args.folds, args.seed)

    events.to_csv(args.output_dir / "event_manifest.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(args.output_dir / "future_pair_manifest.csv", index=False, encoding="utf-8-sig")
    patients.to_csv(args.output_dir / "patient_manifest.csv", index=False, encoding="utf-8-sig")

    print(
        "[OK] future biopsy pairs: "
        f"patients={len(patients)}, positive_patients={int(patients.label.sum())}, "
        f"pairs={len(pairs)}, positive_pairs={int(pairs.label.sum())}, "
        f"target_events={pairs.target_event_key.nunique()}, "
        f"positive_target_events={pairs.loc[pairs.label.eq(1), 'target_event_key'].nunique()}"
    )
    print(report.to_string(index=False))
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
