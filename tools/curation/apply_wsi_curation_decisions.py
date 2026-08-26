#!/usr/bin/env python3
"""Merge reviewed quality and stain decisions into a validated curation manifest."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from PIL import Image


STAIN_GROUPS = {"HE", "IHC", "special_other", "unknown"}
QUALITY_STATUSES = {"usable", "usable_low_quality", "exclude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="slide_curation_manifest.csv from build_wsi_curation_manifest.py")
    parser.add_argument("--quality-decisions", type=Path, default=None,
                        help="Edited quality_review_queue.csv with quality_manual_status/reason filled in.")
    parser.add_argument("--stain-map", type=Path, default=None,
                        help="Edited stain_signature_review.csv with stain_group filled in.")
    parser.add_argument("--stain-cluster-map", type=Path, default=None,
                        help="Edited stain_color_cluster_review.csv with pure-cluster stain_group values filled in.")
    parser.add_argument("--stain-decisions", type=Path, default=None,
                        help="Edited stain_manual_review_queue.csv with final per-slide stain_group values.")
    parser.add_argument("--require-quality-review-complete", action="store_true",
                        help="Fail if any auto-flagged or audit slide has no manual quality decision.")
    parser.add_argument("--export-unresolved-stain-previews", action="store_true",
                        help="Export compact thumbnails for unknown stain slides after applying supplied maps.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def apply_quality(frame: pd.DataFrame, decisions_path: Path | None) -> pd.DataFrame:
    frame = frame.copy()
    if decisions_path is not None:
        decisions = read_csv(decisions_path)
        required = {"slide_id", "quality_manual_status"}
        missing = required.difference(decisions.columns)
        if missing:
            raise ValueError(f"Quality decisions missing columns: {sorted(missing)}")
        if decisions.slide_id.duplicated().any():
            raise ValueError("Quality decisions contain duplicate slide_id values")
        decisions = decisions[decisions.quality_manual_status.ne("")].copy()
        invalid = set(decisions.quality_manual_status).difference(QUALITY_STATUSES)
        if invalid:
            raise ValueError(f"Unsupported quality_manual_status values: {sorted(invalid)}")
        unknown = set(decisions.slide_id).difference(frame.slide_id)
        if unknown:
            raise ValueError(f"Quality decisions reference unknown slide_id values: {sorted(unknown)[:10]}")
        decisions = decisions.set_index("slide_id")
        for column in ["quality_manual_status", "quality_manual_reason", "quality_reviewer", "quality_reviewed_at"]:
            if column in decisions.columns:
                frame[column] = frame["slide_id"].map(decisions[column]).fillna(frame[column])

    frame["quality_status_final"] = "usable_auto"
    frame.loc[frame["known_quality_exclusion"].astype(str).str.lower().eq("true"), "quality_status_final"] = "exclude"
    reviewed = frame["quality_manual_status"].isin(QUALITY_STATUSES)
    frame.loc[reviewed, "quality_status_final"] = frame.loc[reviewed, "quality_manual_status"]
    frame["include_quality_clean"] = ~frame["quality_status_final"].eq("exclude")
    return frame


def apply_stain(
    frame: pd.DataFrame,
    stain_map_path: Path | None,
    cluster_map_path: Path | None,
    stain_decisions_path: Path | None,
) -> pd.DataFrame:
    frame = frame.copy()
    if stain_map_path is not None:
        stain_map = read_csv(stain_map_path)
        required = {"stain_signature", "stain_group"}
        missing = required.difference(stain_map.columns)
        if missing:
            raise ValueError(f"Stain map missing columns: {sorted(missing)}")
        if stain_map.stain_signature.duplicated().any():
            raise ValueError("Stain map contains duplicate stain_signature values")
        stain_map = stain_map[stain_map.stain_group.ne("")].copy()
        invalid = set(stain_map.stain_group).difference(STAIN_GROUPS)
        if invalid:
            raise ValueError(f"Unsupported stain_group values: {sorted(invalid)}")
        unknown = set(stain_map.stain_signature).difference(frame.stain_signature)
        if unknown:
            raise ValueError(f"Stain map references unknown signatures: {sorted(unknown)[:10]}")
        stain_map = stain_map.set_index("stain_signature")
        for column in ["stain_group", "stain_raw", "stain_confidence", "stain_note"]:
            if column in stain_map.columns:
                frame[column] = frame["stain_signature"].map(stain_map[column]).fillna(frame[column])
        mapped = frame["stain_signature"].isin(stain_map.index)
        frame.loc[mapped, "stain_source"] = "signature_map"
        frame.loc[mapped, "needs_stain_signature_review"] = False
    if cluster_map_path is not None:
        cluster_map = read_csv(cluster_map_path)
        required = {"stain_color_cluster", "stain_group"}
        missing = required.difference(cluster_map.columns)
        if missing:
            raise ValueError(f"Stain cluster map missing columns: {sorted(missing)}")
        cluster_map = cluster_map[cluster_map.stain_group.ne("")].copy()
        invalid = set(cluster_map.stain_group).difference(STAIN_GROUPS)
        if invalid:
            raise ValueError(f"Unsupported cluster stain_group values: {sorted(invalid)}")
        cluster_map["cluster_key"] = pd.to_numeric(cluster_map.stain_color_cluster, errors="raise").astype(int).astype(str)
        if cluster_map.cluster_key.duplicated().any():
            raise ValueError("Stain cluster map contains duplicate cluster identifiers")
        if "stain_color_cluster" not in frame.columns:
            raise ValueError("Manifest has no stain_color_cluster column")
        frame_cluster_key = pd.to_numeric(frame.stain_color_cluster, errors="coerce").astype("Int64").astype(str)
        cluster_map = cluster_map.set_index("cluster_key")
        mapped_group = frame_cluster_key.map(cluster_map.stain_group)
        eligible = frame.stain_group.eq("unknown") & mapped_group.notna()
        frame.loc[eligible, "stain_group"] = mapped_group[eligible]
        frame.loc[eligible, "stain_source"] = "color_cluster_map"
        if "stain_confidence" in cluster_map.columns:
            mapped_confidence = frame_cluster_key.map(cluster_map.stain_confidence)
            frame.loc[eligible & mapped_confidence.notna(), "stain_confidence"] = mapped_confidence[eligible & mapped_confidence.notna()]
        if "stain_note" in cluster_map.columns:
            mapped_note = frame_cluster_key.map(cluster_map.stain_note)
            frame.loc[eligible & mapped_note.notna(), "stain_note"] = mapped_note[eligible & mapped_note.notna()]
    if stain_decisions_path is not None:
        decisions = read_csv(stain_decisions_path)
        required = {"slide_id", "stain_group"}
        missing = required.difference(decisions.columns)
        if missing:
            raise ValueError(f"Stain decisions missing columns: {sorted(missing)}")
        if decisions.slide_id.duplicated().any():
            raise ValueError("Stain decisions contain duplicate slide_id values")
        decisions = decisions[decisions.stain_group.ne("")].copy()
        invalid = set(decisions.stain_group).difference(STAIN_GROUPS)
        if invalid:
            raise ValueError(f"Unsupported stain_group values: {sorted(invalid)}")
        unknown = set(decisions.slide_id).difference(frame.slide_id)
        if unknown:
            raise ValueError(f"Stain decisions reference unknown slide_id values: {sorted(unknown)[:10]}")
        decisions = decisions.set_index("slide_id")
        for column in ["stain_group", "stain_raw", "stain_confidence", "stain_note"]:
            if column in decisions.columns:
                frame[column] = frame["slide_id"].map(decisions[column]).fillna(frame[column])
        mapped = frame["slide_id"].isin(decisions.index)
        frame.loc[mapped, "stain_source"] = "slide_review"
    frame["include_he_only"] = frame["stain_group"].eq("HE") & frame["include_quality_clean"]
    frame["include_non_he_only"] = frame["stain_group"].isin(["IHC", "special_other"]) & frame["include_quality_clean"]
    return frame


def quality_review_required(frame: pd.DataFrame) -> pd.Series:
    return (
        ~frame["known_quality_exclusion"].astype(str).str.lower().eq("true")
        & ((frame["quality_auto_priority"] != "none") | frame["quality_audit_sample"].astype(str).str.lower().eq("true"))
    )


def export_thumbnail_previews(queue: pd.DataFrame, output_dir: Path) -> None:
    image_dir = output_dir / "unresolved_stain_review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for rank, row in enumerate(queue.itertuples(index=False), start=1):
        source = Path(row.thumbnail_path)
        if not source.is_file():
            continue
        filename = f"{rank:04d}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', row.slide_id)}.jpg"
        try:
            with Image.open(source) as image:
                preview = image.convert("RGB")
                preview.thumbnail((1000, 1000))
                preview.save(image_dir / filename, quality=90)
            index_rows.append({"rank": rank, "slide_id": row.slide_id, "preview_file": filename})
        except Exception:
            continue
    pd.DataFrame(index_rows).to_csv(output_dir / "unresolved_stain_preview_index.csv", index=False)


def main() -> int:
    args = parse_args()
    frame = read_csv(args.manifest)
    required = {"slide_id", "slide_rel_path", "source_dataset", "stain_signature", "stain_group", "known_quality_exclusion"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.slide_id.duplicated().any():
        raise ValueError("Manifest contains duplicate slide_id values")
    frame = apply_quality(frame, args.quality_decisions)
    if args.require_quality_review_complete:
        required_review = quality_review_required(frame)
        pending = frame[required_review & ~frame["quality_manual_status"].isin(QUALITY_STATUSES)]
        if not pending.empty:
            raise ValueError(f"Quality review is incomplete: {len(pending)} queued slides have no manual decision")
    frame = apply_stain(frame, args.stain_map, args.stain_cluster_map, args.stain_decisions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "slide_curation_manifest_curated.csv", index=False)

    exclusions = frame[~frame.include_quality_clean].copy()
    exclusions["reason"] = exclusions["quality_manual_reason"].where(
        exclusions["quality_manual_reason"].ne(""), "WSI quality curation exclusion"
    )
    exclusions["source"] = "wsi_curation"
    exclusions["exclude_from"] = "all_training_tasks"
    exclusions[["source_dataset", "slide_rel_path", "reason", "source", "exclude_from"]].to_csv(
        args.output_dir / "quality_exclusions_curated.csv", index=False
    )

    summary = (
        frame.groupby(["source_dataset", "quality_status_final", "stain_group"], dropna=False)
        .size().rename("slides").reset_index()
    )
    summary.to_csv(args.output_dir / "curation_summary.csv", index=False)

    unresolved_stain = frame[frame.include_quality_clean & frame.stain_group.eq("unknown")].copy()
    unresolved_stain.to_csv(args.output_dir / "stain_manual_review_queue.csv", index=False)
    if args.export_unresolved_stain_previews:
        export_thumbnail_previews(unresolved_stain, args.output_dir)

    queued_quality_pending = quality_review_required(frame) & ~frame["quality_manual_status"].isin(QUALITY_STATUSES)
    print(f"Slides: {len(frame)}")
    print(f"Quality-clean slides: {int(frame.include_quality_clean.sum())}")
    print(f"H&E-only slides: {int(frame.include_he_only.sum())}")
    print(f"Non-H&E-only slides: {int(frame.include_non_he_only.sum())}")
    print(f"Unknown stain group: {int(frame.stain_group.eq('unknown').sum())}")
    print(f"Queued quality slides without manual decision: {int(queued_quality_pending.sum())}")
    print(f"Unresolved stain slides for per-slide review: {len(unresolved_stain)}")
    print(f"Curated manifest: {args.output_dir / 'slide_curation_manifest_curated.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
