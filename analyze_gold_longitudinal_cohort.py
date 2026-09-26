#!/usr/bin/env python3
"""Describe the longitudinal structure of the exact pathology-ID gold cohort."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ACR_GRADES = ("0R", "1R", "2R", "3R")
AMR_GRADES = ("pAMR0", "pAMR1", "pAMR1(I+)", "pAMR2")
HORIZONS = (30, 90, 180, 365)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-xlsx", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def load_events(label_xlsx: Path, gold_csv: Path) -> pd.DataFrame:
    labels = pd.read_excel(label_xlsx, sheet_name="Sheet1")
    if labels.shape[1] < 5:
        raise ValueError("Sheet1 must contain patient, date, ACR, AMR, and pathology ID columns")
    labels = labels.iloc[:, :5].copy()
    labels.columns = ["patient_raw", "label_biopsy_date", "acr_grade", "amr_grade", "pathology_id"]
    labels = labels.dropna(subset=["patient_raw", "pathology_id"])
    labels["event_key"] = labels["pathology_id"].map(normalize)
    labels["label_biopsy_date"] = pd.to_datetime(labels["label_biopsy_date"], errors="coerce").dt.normalize()
    if labels.event_key.duplicated().any():
        raise ValueError("Sheet1 pathology IDs must be unique")

    bags = pd.read_csv(gold_csv)
    required = {"case_id", "slide_id", "event_id", "source_dataset", "biopsy_date"}
    missing = required.difference(bags.columns)
    if missing:
        raise ValueError(f"Gold CSV is missing columns: {sorted(missing)}")
    bags["event_key"] = bags.event_id.map(normalize)
    event_bags = bags.groupby("event_key", as_index=False).agg(
        case_id=("case_id", "first"),
        event_id=("event_id", "first"),
        biopsy_date=("biopsy_date", "first"),
        slide_count=("slide_id", "size"),
        source_dataset=("source_dataset", lambda values: "+".join(sorted(set(values)))),
    )
    events = event_bags.merge(
        labels[["event_key", "label_biopsy_date", "acr_grade", "amr_grade"]],
        on="event_key",
        how="left",
        validate="one_to_one",
    )
    events["biopsy_date"] = pd.to_datetime(events.biopsy_date, errors="coerce").dt.normalize()
    if events[["label_biopsy_date", "acr_grade", "amr_grade"]].isna().any().any():
        raise ValueError("At least one gold event could not be joined to Sheet1")
    if (events.biopsy_date != events.label_biopsy_date).any():
        raise ValueError("Gold CSV and Sheet1 biopsy dates disagree")
    if not events.acr_grade.isin(ACR_GRADES).all():
        raise ValueError("Unexpected ACR grade in exact-match cohort")
    if not events.amr_grade.isin(AMR_GRADES).all():
        raise ValueError("Unexpected AMR grade in exact-match cohort")
    if events.duplicated(["case_id", "biopsy_date"]).any():
        raise ValueError("More than one exact-match pathology event occurs for a patient on the same date")

    events = events.sort_values(["case_id", "biopsy_date", "event_key"]).reset_index(drop=True)
    events["event_index"] = events.groupby("case_id").cumcount() + 1
    events["patient_event_count"] = events.groupby("case_id").event_key.transform("size")
    events["acr_positive"] = events.acr_grade.ne("0R")
    events["acr_high_grade"] = events.acr_grade.isin(["2R", "3R"])
    events["amr_positive"] = events.amr_grade.ne("pAMR0")
    events["any_rejection"] = events.acr_positive | events.amr_positive
    events["previous_biopsy_date"] = events.groupby("case_id").biopsy_date.shift()
    events["next_biopsy_date"] = events.groupby("case_id").biopsy_date.shift(-1)
    events["days_since_previous"] = (events.biopsy_date - events.previous_biopsy_date).dt.days
    events["days_to_next"] = (events.next_biopsy_date - events.biopsy_date).dt.days
    for column in ("acr_grade", "amr_grade", "acr_positive", "acr_high_grade", "amr_positive", "any_rejection"):
        events[f"next_{column}"] = events.groupby("case_id")[column].shift(-1)
    return events.drop(columns=["label_biopsy_date"])


def patient_summary(events: pd.DataFrame) -> pd.DataFrame:
    summary = events.groupby("case_id", as_index=False).agg(
        event_count=("event_key", "size"),
        slide_count=("slide_count", "sum"),
        first_biopsy_date=("biopsy_date", "min"),
        last_biopsy_date=("biopsy_date", "max"),
        ever_acr_positive=("acr_positive", "max"),
        ever_acr_high_grade=("acr_high_grade", "max"),
        ever_amr_positive=("amr_positive", "max"),
        ever_any_rejection=("any_rejection", "max"),
    )
    summary["observed_span_days"] = (summary.last_biopsy_date - summary.first_biopsy_date).dt.days
    intervals = events.groupby("case_id").days_since_previous.agg(["mean", "median", "min", "max"])
    intervals.columns = ["mean_interval_days", "median_interval_days", "min_interval_days", "max_interval_days"]
    grade_counts = []
    for case_id, group in events.groupby("case_id"):
        row: dict[str, object] = {"case_id": case_id}
        for grade in ACR_GRADES:
            row[f"acr_{grade}_events"] = int(group.acr_grade.eq(grade).sum())
        for grade in AMR_GRADES:
            safe_grade = grade.replace("(", "_").replace(")", "").replace("+", "plus")
            row[f"amr_{safe_grade}_events"] = int(group.amr_grade.eq(grade).sum())
        grade_counts.append(row)
    return summary.merge(intervals, left_on="case_id", right_index=True, how="left").merge(
        pd.DataFrame(grade_counts), on="case_id", how="left"
    )


def patient_composite_summary(events: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    repeated = patients[patients.event_count >= 2].copy()
    selected = events[events.case_id.isin(repeated.case_id)].copy()
    selected["high_grade_or_amr"] = selected.acr_high_grade | selected.amr_positive
    first = selected[selected.event_index.eq(1)].set_index("case_id")
    first_positive_date = (
        selected[selected.high_grade_or_amr]
        .groupby("case_id")
        .biopsy_date.min()
        .rename("first_composite_positive_date")
    )
    result = repeated.set_index("case_id").join(first_positive_date)
    result["ever_high_grade_or_amr"] = result.ever_acr_high_grade | result.ever_amr_positive
    result["baseline_high_grade_or_amr"] = first.high_grade_or_amr
    result["incident_after_negative_baseline"] = (
        result.ever_high_grade_or_amr & ~result.baseline_high_grade_or_amr
    )
    result["days_to_first_composite_positive"] = (
        result.first_composite_positive_date - result.first_biopsy_date
    ).dt.days
    result["composite_phenotype"] = "negative"
    result.loc[result.ever_acr_high_grade & ~result.ever_amr_positive, "composite_phenotype"] = "ACR>=2R only"
    result.loc[~result.ever_acr_high_grade & result.ever_amr_positive, "composite_phenotype"] = "AMR>=pAMR1 only"
    result.loc[result.ever_acr_high_grade & result.ever_amr_positive, "composite_phenotype"] = "both"
    return result.reset_index()


def cohort_summary(events: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    repeated = patients[patients.event_count >= 2]
    intervals = events.days_since_previous.dropna()
    rows = [
        ("Patients", len(patients), len(patients), "Unique pseudonymous patients"),
        ("Biopsy events", len(events), len(events), "One exact pathology-ID event per row"),
        ("Slide bags", int(events.slide_count.sum()), int(events.slide_count.sum()), "Multiple stains/slides may belong to one event"),
        ("Patients with >=2 events", len(repeated), len(patients), "Patients contributing at least one longitudinal pair"),
        ("Patients with 1 event", int(patients.event_count.eq(1).sum()), len(patients), "Not usable for within-patient future targets"),
        ("Mean events per patient", patients.event_count.mean(), len(patients), "Event-level, not slide-level"),
        ("Median events per patient", patients.event_count.median(), len(patients), "Event-level, not slide-level"),
        ("Mean inter-biopsy interval (days)", intervals.mean(), len(intervals), "All consecutive event pairs"),
        ("Median inter-biopsy interval (days)", intervals.median(), len(intervals), "All consecutive event pairs"),
        ("Mean observed span among repeated patients (days)", repeated.observed_span_days.mean(), len(repeated), "First to last matched biopsy"),
        ("Median observed span among repeated patients (days)", repeated.observed_span_days.median(), len(repeated), "First to last matched biopsy"),
        ("Consecutive event pairs", int(events.next_biopsy_date.notna().sum()), int(events.next_biopsy_date.notna().sum()), "Correlated within 110 patients"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "denominator", "definition"])


def distribution_summary(events: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for count, number in patients.event_count.value_counts().sort_index().items():
        rows.append({"section": "Events per patient", "category": str(count), "count": int(number), "denominator": len(patients)})
    interval_bins = pd.cut(
        events.days_since_previous.dropna(),
        bins=[0, 30, 90, 180, 365, float("inf")],
        labels=["1-30 days", "31-90 days", "91-180 days", "181-365 days", ">365 days"],
        include_lowest=True,
    )
    for category, number in interval_bins.value_counts(sort=False).items():
        rows.append({"section": "Inter-biopsy interval", "category": str(category), "count": int(number), "denominator": len(interval_bins)})
    for grade in ACR_GRADES:
        mask = events.acr_grade.eq(grade)
        rows.append({"section": "ACR event grade", "category": grade, "count": int(mask.sum()), "denominator": len(events)})
        rows.append({"section": "Patients ever ACR grade", "category": grade, "count": events.loc[mask, "case_id"].nunique(), "denominator": len(patients)})
    for grade in AMR_GRADES:
        mask = events.amr_grade.eq(grade)
        rows.append({"section": "AMR event grade", "category": grade, "count": int(mask.sum()), "denominator": len(events)})
        rows.append({"section": "Patients ever AMR grade", "category": grade, "count": events.loc[mask, "case_id"].nunique(), "denominator": len(patients)})
    result = pd.DataFrame(rows)
    result["fraction"] = result["count"] / result["denominator"]
    return result


def transition_counts(events: pd.DataFrame) -> pd.DataFrame:
    pairs = events[events.next_biopsy_date.notna()].copy()
    rows: list[dict[str, object]] = []
    for variable, next_variable, analysis in (
        ("acr_grade", "next_acr_grade", "ACR grade"),
        ("amr_grade", "next_amr_grade", "AMR grade"),
        ("acr_positive", "next_acr_positive", "ACR >=1R"),
        ("acr_high_grade", "next_acr_high_grade", "ACR >=2R"),
        ("amr_positive", "next_amr_positive", "AMR positive"),
        ("any_rejection", "next_any_rejection", "Any rejection"),
    ):
        table = pairs.groupby([variable, next_variable], dropna=False).size().rename("count").reset_index()
        table.columns = ["from_state", "to_state", "count"]
        table["analysis"] = analysis
        table["from_total"] = table.groupby("from_state")["count"].transform("sum")
        table["transition_fraction"] = table["count"] / table["from_total"]
        rows.extend(table[["analysis", "from_state", "to_state", "count", "from_total", "transition_fraction"]].to_dict("records"))
    return pd.DataFrame(rows)


def horizon_feasibility(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = ("acr_positive", "acr_high_grade", "amr_positive", "any_rejection")
    for horizon in HORIZONS:
        eligible: list[dict[str, object]] = []
        for row in events.itertuples(index=False):
            future = events[
                events.case_id.eq(row.case_id)
                & events.biopsy_date.gt(row.biopsy_date)
                & events.biopsy_date.le(row.biopsy_date + pd.Timedelta(days=horizon))
            ]
            if future.empty:
                continue
            record: dict[str, object] = {"case_id": row.case_id}
            for target in targets:
                future_positive = bool(future[target].any())
                record[f"future_{target}"] = future_positive
                record[f"baseline_negative_{target}"] = not bool(getattr(row, target))
                record[f"incident_{target}"] = record[f"baseline_negative_{target}"] and future_positive
            eligible.append(record)
        frame = pd.DataFrame(eligible)
        for target in targets:
            baseline_negative = int(frame[f"baseline_negative_{target}"].sum()) if not frame.empty else 0
            future_positive = int(frame[f"future_{target}"].sum()) if not frame.empty else 0
            incident_positive = int(frame[f"incident_{target}"].sum()) if not frame.empty else 0
            rows.append(
                {
                    "horizon_days": horizon,
                    "target": target,
                    "index_events_with_later_biopsy": len(frame),
                    "patients_with_later_biopsy": frame.case_id.nunique() if not frame.empty else 0,
                    "future_positive_index_events": future_positive,
                    "future_positive_fraction": future_positive / len(frame) if len(frame) else float("nan"),
                    "baseline_negative_index_events": baseline_negative,
                    "incident_positive_index_events": incident_positive,
                    "incident_positive_fraction": incident_positive / baseline_negative if baseline_negative else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def first_index_feasibility(events: pd.DataFrame) -> pd.DataFrame:
    first = events[events.event_index.eq(1) & events.next_biopsy_date.notna()].copy()
    targets = ("acr_positive", "acr_high_grade", "amr_positive", "any_rejection")
    rows = []
    for target in targets:
        baseline_negative = first[~first[target]]
        rows.append(
            {
                "target": target,
                "patients_with_next_biopsy": len(first),
                "next_event_positive": int(first[f"next_{target}"].sum()),
                "next_event_positive_fraction": first[f"next_{target}"].mean(),
                "baseline_negative_patients": len(baseline_negative),
                "incident_next_event_positive": int(baseline_negative[f"next_{target}"].sum()),
                "incident_next_event_fraction": baseline_negative[f"next_{target}"].mean(),
                "median_days_to_next": first.days_to_next.median(),
                "min_days_to_next": first.days_to_next.min(),
                "max_days_to_next": first.days_to_next.max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.label_xlsx, args.gold_csv)
    patients = patient_summary(events)
    composite_patients = patient_composite_summary(events, patients)
    outputs = {
        "cohort_summary.csv": cohort_summary(events, patients),
        "distribution_summary.csv": distribution_summary(events, patients),
        "patient_summary.csv": patients,
        "patient_composite_summary.csv": composite_patients,
        "event_sequence.csv": events,
        "transition_counts.csv": transition_counts(events),
        "horizon_feasibility.csv": horizon_feasibility(events),
        "first_index_feasibility.csv": first_index_feasibility(events),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
    print(
        f"[OK] patients={len(patients)}, events={len(events)}, slides={int(events.slide_count.sum())}, "
        f"repeated_patients={int(patients.event_count.ge(2).sum())} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
