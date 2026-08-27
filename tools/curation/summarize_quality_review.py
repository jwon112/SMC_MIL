#!/usr/bin/env python3
"""Summarize manual WSI quality review to calibrate the screening queue."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


QUALITY_STATUSES = {"usable", "usable_low_quality", "exclude"}
METRICS = [
    "tissue_ratio",
    "sharpness_score",
    "tissue_luminance_mean",
    "tissue_luminance_std",
    "tissue_saturation_mean",
    "grid_periodicity_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quality-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def main() -> int:
    args = parse_args()
    manifest = read_csv(args.manifest)
    decisions = read_csv(args.quality_decisions)
    required = {"slide_id", "quality_manual_status"}
    for name, frame in [("manifest", manifest), ("quality decisions", decisions)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        if frame.slide_id.duplicated().any():
            raise ValueError(f"{name} contains duplicate slide_id values")

    decisions = decisions[decisions.quality_manual_status.ne("")].copy()
    invalid = set(decisions.quality_manual_status).difference(QUALITY_STATUSES)
    if invalid:
        raise ValueError(f"Unsupported quality_manual_status values: {sorted(invalid)}")
    reviewed = manifest.drop(columns=["quality_manual_status"], errors="ignore").merge(
        decisions[["slide_id", "quality_manual_status"]], on="slide_id", how="inner", validate="one_to_one"
    )
    if reviewed.empty:
        raise ValueError("No completed quality decisions found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(args.output_dir / "quality_reviewed_slides.csv", index=False)

    summary = (
        reviewed.groupby("quality_manual_status", dropna=False)
        .agg(
            slides=("slide_id", "count"),
            flagged=("quality_auto_priority", lambda values: int((values != "none").sum())),
            audit_only=("quality_audit_sample", lambda values: int(((values == "True") | (values == "true")).sum())),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "quality_review_summary.csv", index=False)

    flags = reviewed[["slide_id", "quality_manual_status", "quality_auto_flags", "quality_audit_sample"]].copy()
    flags["review_trigger"] = flags["quality_auto_flags"]
    audit_only = flags.review_trigger.eq("") & flags.quality_audit_sample.astype(str).str.lower().eq("true")
    flags.loc[audit_only, "review_trigger"] = "random_audit"
    flags = flags.assign(review_trigger=flags.review_trigger.str.split(";"))
    flags = flags.explode("review_trigger")
    flags = flags[flags.review_trigger.ne("")]
    flag_outcomes = pd.crosstab(flags.review_trigger, flags.quality_manual_status)
    for status in sorted(QUALITY_STATUSES):
        if status not in flag_outcomes:
            flag_outcomes[status] = 0
    flag_outcomes["reviewed"] = flag_outcomes.sum(axis=1)
    flag_outcomes["exclude_rate"] = flag_outcomes["exclude"] / flag_outcomes["reviewed"]
    flag_outcomes.reset_index().to_csv(args.output_dir / "quality_flag_outcomes.csv", index=False)

    metric_rows = []
    for status, group in reviewed.groupby("quality_manual_status", sort=True):
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            metric_rows.append({
                "quality_manual_status": status,
                "metric": metric,
                "slides": len(values),
                "median": values.median() if not values.empty else None,
                "q25": values.quantile(0.25) if not values.empty else None,
                "q75": values.quantile(0.75) if not values.empty else None,
            })
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "quality_metric_by_decision.csv", index=False)

    excluded = reviewed[reviewed.quality_manual_status.eq("exclude")]
    audit_excluded = excluded[
        excluded.quality_auto_flags.eq("")
        & excluded.quality_audit_sample.astype(str).str.lower().eq("true")
    ]
    print(f"Reviewed slides: {len(reviewed)}")
    print(f"Excluded slides: {len(excluded)}")
    print(f"Excluded random-audit slides: {len(audit_excluded)}")
    print(f"Report directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
