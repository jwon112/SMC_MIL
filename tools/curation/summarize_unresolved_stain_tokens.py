#!/usr/bin/env python3
"""Find repeated filename tokens among slides unresolved by stain filename rules."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


STAIN_GROUPS = ["HE", "IHC", "special_other", "unknown"]
IGNORED_TOKENS = {
    "BLOCK", "CASE", "LEVEL", "SCAN", "SERIES", "SLIDE", "SPECIMEN", "TISSUE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument(
        "--include-nonusable", action="store_true",
        help="Include slides outside the quality-usable cohort.",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def candidate_tokens(signature: str) -> list[str]:
    tokens = re.findall(r"[A-Z][A-Z0-9+]*", signature.upper())
    return sorted(
        {
            token for token in tokens
            if len(token) >= 2
            and token not in IGNORED_TOKENS
            and not re.fullmatch(r"S\d+", token)
        }
    )


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.manifest, dtype=str).fillna("")
    required = {
        "slide_id", "stain_signature", "stain_group", "stain_source",
        "stain_group_auto",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    frame = frame[frame.stain_group_auto.str.strip().isin(["", "unknown"])].copy()
    if not args.include_nonusable:
        if "include_quality_usable" not in frame:
            raise ValueError("Manifest missing include_quality_usable")
        frame = frame[as_bool(frame.include_quality_usable)].copy()

    frame["token"] = frame.stain_signature.map(candidate_tokens)
    exploded = frame.explode("token")
    exploded = exploded[exploded.token.notna() & exploded.token.ne("")].copy()
    if exploded.empty:
        raise ValueError("No candidate tokens were found")

    rows = []
    for token, group in exploded.groupby("token", sort=False):
        counts = group.stain_group.value_counts()
        known_counts = counts.reindex(STAIN_GROUPS[:-1], fill_value=0)
        known_total = int(known_counts.sum())
        dominant_group = known_counts.idxmax() if known_total else "unknown"
        dominant_count = int(known_counts.max()) if known_total else 0
        source_counts = group.stain_source.value_counts()
        examples = group.stain_signature.drop_duplicates().head(5).tolist()
        rows.append(
            {
                "token": token,
                "slides": int(group.slide_id.nunique()),
                "signatures": int(group.stain_signature.nunique()),
                "HE": int(counts.get("HE", 0)),
                "IHC": int(counts.get("IHC", 0)),
                "special_other": int(counts.get("special_other", 0)),
                "unknown": int(counts.get("unknown", 0)),
                "dominant_group": dominant_group,
                "dominant_purity": dominant_count / known_total if known_total else 0.0,
                "slide_review": int(source_counts.get("slide_review", 0)),
                "signature_map": int(source_counts.get("signature_map", 0)),
                "color_cluster_map": int(source_counts.get("color_cluster_map", 0)),
                "examples": " | ".join(examples),
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary[summary.slides.ge(args.min_count)].copy()
    summary.sort_values(
        ["slides", "dominant_purity", "token"],
        ascending=[False, False, True], inplace=True,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    display_columns = [
        "token", "slides", "HE", "IHC", "special_other", "unknown",
        "dominant_group", "dominant_purity", "slide_review",
    ]
    display = summary[display_columns].head(args.top).copy()
    display["dominant_purity"] = display.dominant_purity.map(lambda value: f"{value:.1%}")
    print(f"Filename-unresolved quality-usable slides: {len(frame)}")
    print(f"Repeated candidate tokens (n >= {args.min_count}): {len(summary)}")
    print(display.to_string(index=False))
    print(f"\nWrote: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
