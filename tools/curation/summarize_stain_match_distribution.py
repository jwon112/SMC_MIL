#!/usr/bin/env python3
"""Summarize curated stain groups by exact pathology-ID match status."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STAIN_GROUPS = ["HE", "IHC", "special_other", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Curated slide_curation_manifest_curated.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSV summaries.",
    )
    parser.add_argument(
        "--include-quality-unusable",
        action="store_true",
        help="Include slides outside the quality-usable cohort.",
    )
    return parser.parse_args()


def add_percentages(counts: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    result = counts.copy()
    totals = result.groupby(group_columns, dropna=False)["slides"].transform("sum")
    result["percent_within_group"] = (100.0 * result["slides"] / totals).round(2)
    return result


def print_table(frame: pd.DataFrame, index: list[str], title: str) -> None:
    counts = pd.crosstab(
        [frame[column] for column in index],
        frame["stain_group"],
        dropna=False,
    ).reindex(columns=STAIN_GROUPS, fill_value=0)
    counts["TOTAL"] = counts.sum(axis=1)
    print(f"\n{title}")
    print(counts.to_string())


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    required = {
        "source_dataset",
        "gold_pathology_id_match",
        "stain_group",
        "include_quality_usable",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if not args.include_quality_unusable:
        frame = frame[frame["include_quality_usable"].astype(bool)].copy()

    frame["match_status"] = frame["gold_pathology_id_match"].map(
        {True: "exact_pathology_match", False: "unmatched"}
    ).fillna("unmatched")
    frame["stain_group"] = frame["stain_group"].fillna("unknown")
    unexpected = set(frame["stain_group"]).difference(STAIN_GROUPS)
    if unexpected:
        raise ValueError(f"Unexpected stain groups: {sorted(unexpected)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_match = (
        frame.groupby(["match_status", "stain_group"], dropna=False)
        .size()
        .rename("slides")
        .reset_index()
    )
    by_match = add_percentages(by_match, ["match_status"])
    by_match.to_csv(args.output_dir / "stain_distribution_by_match_status.csv", index=False)

    by_source_match = (
        frame.groupby(["source_dataset", "match_status", "stain_group"], dropna=False)
        .size()
        .rename("slides")
        .reset_index()
    )
    by_source_match = add_percentages(by_source_match, ["source_dataset", "match_status"])
    by_source_match.to_csv(
        args.output_dir / "stain_distribution_by_source_and_match_status.csv",
        index=False,
    )

    print(f"Slides included: {len(frame)}")
    print_table(frame, ["match_status"], "Stain distribution by exact pathology-ID match")
    print_table(frame, ["source_dataset", "match_status"], "Stain distribution by source and match")
    print(f"\nWrote: {args.output_dir / 'stain_distribution_by_match_status.csv'}")
    print(f"Wrote: {args.output_dir / 'stain_distribution_by_source_and_match_status.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
