#!/usr/bin/env python3
"""Create date-based EHR biopsy candidates for pathology-ID-unmatched DICOM WSIs.

This tool never assigns a training label.  It calibrates the observed difference
between WSI scan dates and gold-label biopsy dates, then creates a review queue
of possible EHR biopsies for each unmatched pathology event.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--label-xlsx", type=Path, required=True,
                        help="Gold pathology-ID label workbook; Sheet1 is used.")
    parser.add_argument("--ehr-xlsx", type=Path, required=True,
                        help="Biopsy-level EHR workbook; biopsy_features sheet is used.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lower-quantile", type=float, default=0.01,
                        help="Lower gold-anchor scan-minus-biopsy date-offset quantile.")
    parser.add_argument("--upper-quantile", type=float, default=0.99,
                        help="Upper gold-anchor scan-minus-biopsy date-offset quantile.")
    parser.add_argument("--extra-days", type=int, default=0,
                        help="Optional symmetric widening after empirical calibration.")
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    parsed = pd.to_datetime(value, errors="coerce")
    return pd.NaT if pd.isna(parsed) else parsed.normalize()


def read_gold_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    if frame.shape[1] < 5:
        raise ValueError("Gold label Sheet1 must have at least five columns")
    result = pd.DataFrame({
        "gold_smc_id": pd.to_numeric(frame.iloc[:, 0], errors="coerce").astype("Int64"),
        "gold_biopsy_date": frame.iloc[:, 1].map(parse_date),
        "gold_pathology_id": frame.iloc[:, 4].astype(str).str.strip(),
    })
    result["event_key"] = result["gold_pathology_id"].map(normalize)
    result = result.dropna(subset=["gold_smc_id", "gold_biopsy_date"])
    result = result[result["event_key"].ne("")].copy()
    if result["event_key"].duplicated().any():
        duplicate = result.loc[result["event_key"].duplicated(keep=False), "gold_pathology_id"].head(10).tolist()
        raise ValueError(f"Gold pathology IDs must be unique after normalization: {duplicate}")
    return result.set_index("event_key", verify_integrity=True)


def read_ehr(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="biopsy_features")
    if frame.shape[1] < 8 or str(frame.columns[0]) != "biopsy_id":
        raise ValueError("Unexpected biopsy_features layout; expected biopsy_id as the first column")

    # The workbook's Korean headers vary by export encoding, so use its documented
    # stable column positions and validate the non-localized first header above.
    result = pd.DataFrame({
        "biopsy_id": frame.iloc[:, 0].astype(str).str.strip(),
        "tpl_no": frame.iloc[:, 1].astype(str).str.strip(),
        "patient_hash": frame.iloc[:, 3].astype(str).str.strip(),
        "smc_id": pd.to_numeric(frame.iloc[:, 4], errors="coerce").astype("Int64"),
        "biopsy_date": frame.iloc[:, 5].map(parse_date),
    })
    result = result.dropna(subset=["smc_id", "biopsy_date"])
    result = result[result["biopsy_id"].ne("")].copy()
    if result["biopsy_id"].duplicated().any():
        raise ValueError("biopsy_id must be unique in the EHR workbook")
    return result.sort_values(["biopsy_date", "smc_id", "biopsy_id"]).reset_index(drop=True)


def read_slide_paths(dataset_root: Path) -> list[str]:
    qc_path = dataset_root / "_qc" / "mask_qc_labels.csv"
    if not qc_path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {qc_path}")
    with qc_path.open(newline="", encoding="utf-8-sig") as handle:
        paths = [row.get("slide_rel_path", "").strip().replace("\\", "/") for row in csv.DictReader(handle)]
    return sorted(path for path in paths if path)


def first_dicom(slide_dir: Path) -> Path | None:
    direct = sorted(slide_dir.glob("*.dcm"))
    if direct:
        return direct[0]
    nested = sorted(slide_dir.rglob("*.dcm"))
    return nested[0] if nested else None


def read_scan_date(path: Path) -> pd.Timestamp | pd.NaT:
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=["StudyDate", "SeriesDate", "ContentDate"],
        )
    except Exception:
        return pd.NaT
    for field in ("StudyDate", "SeriesDate", "ContentDate"):
        value = parse_date(getattr(dataset, field, None))
        if not pd.isna(value):
            return value
    return pd.NaT


def build_events(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    slides: list[dict[str, object]] = []
    for rel_path in read_slide_paths(dataset_root):
        slide_dir = dataset_root / rel_path
        dcm = first_dicom(slide_dir)
        event_raw = rel_path.split("/")[0]
        slides.append({
            "event_key": normalize(event_raw),
            "event_id": event_raw,
            "slide_rel_path": rel_path,
            "scan_date": read_scan_date(dcm) if dcm else pd.NaT,
            "metadata_state": "ok" if dcm else "missing_dicom",
        })

    slide_frame = pd.DataFrame(slides)
    if slide_frame.empty:
        raise ValueError("No WSI rows found in the DICOM QC manifest")

    events: list[dict[str, object]] = []
    for event_key, group in slide_frame.groupby("event_key", sort=True):
        dates = sorted({date for date in group["scan_date"] if not pd.isna(date)})
        events.append({
            "event_key": event_key,
            "event_id": group["event_id"].iloc[0],
            "slide_count": len(group),
            "scan_date_min": dates[0] if dates else pd.NaT,
            "scan_date_max": dates[-1] if dates else pd.NaT,
            "scan_dates": ";".join(date.date().isoformat() for date in dates),
            "metadata_missing_slides": int(group["metadata_state"].ne("ok").sum()),
        })
    return pd.DataFrame(events), slide_frame


def closest_offset(scan_dates: str, biopsy_date: pd.Timestamp) -> tuple[int, str]:
    dates = [parse_date(value) for value in str(scan_dates).split(";") if value]
    offsets = [(int((date - biopsy_date).days), date.date().isoformat()) for date in dates if not pd.isna(date)]
    if not offsets:
        raise ValueError("No valid WSI scan dates")
    return min(offsets, key=lambda item: (abs(item[0]), item[0]))


def print_counts(events: pd.DataFrame, candidates: pd.DataFrame) -> None:
    status_counts = events["linkage_status"].value_counts().sort_index()
    print("\nEvent linkage status")
    for status, count in status_counts.items():
        print(f"{status:28s} {count:5d}")
    print(f"Candidate rows: {len(candidates)}")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.lower_quantile <= args.upper_quantile <= 1.0:
        raise ValueError("Quantiles must satisfy 0 <= lower <= upper <= 1")
    if args.extra_days < 0:
        raise ValueError("--extra-days must be non-negative")

    gold = read_gold_labels(args.label_xlsx)
    ehr = read_ehr(args.ehr_xlsx)
    events, slides = build_events(args.dicom_root)
    events = events.join(gold, on="event_key")
    events["gold_pathology_id_match"] = events["gold_pathology_id"].notna()

    anchors = events[events["gold_pathology_id_match"] & events["scan_dates"].ne("")].copy()
    anchor_rows: list[dict[str, object]] = []
    for row in anchors.itertuples(index=False):
        offset_days, closest_scan_date = closest_offset(row.scan_dates, row.gold_biopsy_date)
        ehr_rows = ehr[(ehr.smc_id == row.gold_smc_id) & (ehr.biopsy_date == row.gold_biopsy_date)]
        anchor_rows.append({
            "event_key": row.event_key,
            "event_id": row.event_id,
            "gold_smc_id": row.gold_smc_id,
            "gold_biopsy_date": row.gold_biopsy_date,
            "closest_scan_date": closest_scan_date,
            "scan_minus_biopsy_days": offset_days,
            "ehr_same_patient_same_date_count": len(ehr_rows),
        })
    calibration = pd.DataFrame(anchor_rows)
    if calibration.empty:
        raise ValueError("No gold-matched WSI events with readable scan dates; cannot calibrate date linkage")

    lower = int(np.floor(calibration["scan_minus_biopsy_days"].quantile(args.lower_quantile))) - args.extra_days
    upper = int(np.ceil(calibration["scan_minus_biopsy_days"].quantile(args.upper_quantile))) + args.extra_days
    if lower > upper:
        raise AssertionError("Invalid calibrated date-offset range")

    candidates: list[dict[str, object]] = []
    unmatched = events[~events["gold_pathology_id_match"]].copy()
    for row in unmatched.itertuples(index=False):
        if not row.scan_dates:
            continue
        for ehr_row in ehr.itertuples(index=False):
            offset_days, closest_scan_date = closest_offset(row.scan_dates, ehr_row.biopsy_date)
            if lower <= offset_days <= upper:
                candidates.append({
                    "event_key": row.event_key,
                    "event_id": row.event_id,
                    "slide_count": row.slide_count,
                    "wsi_scan_dates": row.scan_dates,
                    "candidate_biopsy_id": ehr_row.biopsy_id,
                    "candidate_smc_id": ehr_row.smc_id,
                    "candidate_patient_hash": ehr_row.patient_hash,
                    "candidate_biopsy_date": ehr_row.biopsy_date,
                    "closest_wsi_scan_date": closest_scan_date,
                    "scan_minus_biopsy_days": offset_days,
                    "absolute_date_difference_days": abs(offset_days),
                })
    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.empty:
        candidate_frame = pd.DataFrame(columns=[
            "event_key", "event_id", "slide_count", "wsi_scan_dates", "candidate_biopsy_id",
            "candidate_smc_id", "candidate_patient_hash", "candidate_biopsy_date",
            "closest_wsi_scan_date", "scan_minus_biopsy_days", "absolute_date_difference_days",
        ])
    else:
        candidate_frame = candidate_frame.sort_values(
            ["event_key", "absolute_date_difference_days", "candidate_biopsy_date", "candidate_biopsy_id"]
        ).reset_index(drop=True)

    candidate_counts = candidate_frame.groupby("event_key").agg(
        candidate_biopsies=("candidate_biopsy_id", "nunique"),
        candidate_patients=("candidate_smc_id", "nunique"),
    )
    events = events.join(candidate_counts, on="event_key")
    events[["candidate_biopsies", "candidate_patients"]] = events[["candidate_biopsies", "candidate_patients"]].fillna(0).astype(int)
    events["linkage_status"] = np.select(
        [
            events["gold_pathology_id_match"],
            events["scan_dates"].eq(""),
            events["candidate_biopsies"].eq(0),
            events["candidate_biopsies"].eq(1),
        ],
        ["gold_pathology_match", "missing_wsi_date", "no_date_candidate", "date_unique_biopsy"],
        default="date_ambiguous",
    )

    review = events[events["linkage_status"].isin(["date_unique_biopsy", "date_ambiguous"])].copy()
    review = review[[
        "event_key", "event_id", "slide_count", "scan_dates", "scan_date_min", "scan_date_max",
        "candidate_biopsies", "candidate_patients", "linkage_status",
    ]]
    review["reviewer_selected_biopsy_id"] = ""
    review["review_status"] = "pending"
    review["reviewer"] = ""
    review["review_note"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output_dir / "wsi_event_linkage_inventory.csv", index=False)
    slides.to_csv(args.output_dir / "wsi_slide_metadata_dates.csv", index=False)
    calibration.to_csv(args.output_dir / "gold_wsi_date_offset_calibration.csv", index=False)
    candidate_frame.to_csv(args.output_dir / "weak_wsi_ehr_candidates_blinded.csv", index=False)
    review.to_csv(args.output_dir / "weak_wsi_linkage_review_queue.csv", index=False)
    pd.DataFrame([{
        "gold_anchor_events": len(calibration),
        "calibration_lower_quantile": args.lower_quantile,
        "calibration_upper_quantile": args.upper_quantile,
        "scan_minus_biopsy_days_lower": lower,
        "scan_minus_biopsy_days_upper": upper,
        "extra_days": args.extra_days,
        "unmatched_events": int((~events.gold_pathology_id_match).sum()),
        "reviewable_unmatched_events": len(review),
    }]).to_csv(args.output_dir / "weak_wsi_linkage_calibration_summary.csv", index=False)

    print(f"WSI events: {len(events)}; slides: {len(slides)}")
    print(f"Gold date anchors: {len(calibration)}")
    print(f"Empirical scan-minus-biopsy window: [{lower}, {upper}] days")
    print_counts(events, candidate_frame)
    print(f"Review queue: {args.output_dir / 'weak_wsi_linkage_review_queue.csv'}")
    print(f"Candidates: {args.output_dir / 'weak_wsi_ehr_candidates_blinded.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
