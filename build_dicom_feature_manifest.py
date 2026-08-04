#!/usr/bin/env python3
"""Build an encoding manifest from AtlasPatch QC outputs.

The manifest selects manual coordinates when they exist, otherwise the original
AtlasPatch coordinates. Postprocessed coordinates are deliberately ignored.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_QC_MANIFEST = "_qc/mask_qc_labels.csv"
DEFAULT_OUTPUT = "_clam/dicom_feature_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--qc-manifest", type=Path, default=Path(DEFAULT_QC_MANIFEST))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--output-dir-name", default="atlaspatch")
    parser.add_argument("--manual-coords-filename", default="patch_coords_manual.h5")
    parser.add_argument("--original-coords-filename", default="patch_coords.h5")
    parser.add_argument(
        "--coords-filename",
        default=None,
        help="Use one level-specific coordinate file for every slide instead of L0 manual/original selection.",
    )
    return parser.parse_args()


def resolve_path(path: Path, dataset_root: Path) -> Path:
    return path if path.is_absolute() else dataset_root / path


def slide_id_from_rel_path(slide_rel_path: str) -> str:
    parts = [part for part in slide_rel_path.replace("\\", "/").split("/") if part]
    raw = "__".join(parts)
    # Keep underscores intact: folders such as "slide" and "slide_" are
    # distinct valid dataset entries and must produce distinct feature files.
    slide_id = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(".")
    if not slide_id:
        raise ValueError(f"Could not create a slide_id from: {slide_rel_path!r}")
    return slide_id


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    qc_manifest = resolve_path(args.qc_manifest, dataset_root)
    output_path = resolve_path(args.output, dataset_root)
    if not qc_manifest.is_file():
        raise FileNotFoundError(f"QC manifest not found: {qc_manifest}")

    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    with qc_manifest.open(newline="", encoding="utf-8-sig") as handle:
        for source_row in csv.DictReader(handle):
            slide_rel_path = source_row.get("slide_rel_path", "").strip().replace("\\", "/")
            if not slide_rel_path:
                warnings.append("blank slide_rel_path")
                continue
            if slide_rel_path in seen_paths:
                warnings.append(f"duplicate slide_rel_path: {slide_rel_path}")
                continue
            seen_paths.add(slide_rel_path)

            slide_dir = dataset_root / Path(slide_rel_path)
            atlaspatch_dir = slide_dir / args.output_dir_name
            manual_coords = atlaspatch_dir / args.manual_coords_filename
            original_coords = atlaspatch_dir / args.original_coords_filename
            manual_mask = atlaspatch_dir / "tissue_mask_manual.png"

            if args.coords_filename:
                coords_path = atlaspatch_dir / args.coords_filename
                coords_source = args.coords_filename
                if not coords_path.is_file():
                    warnings.append(f"missing level-specific coordinates: {slide_rel_path}")
                    continue
            else:
                if manual_mask.is_file() and not manual_coords.is_file():
                    warnings.append(f"manual mask has no manual coordinates: {slide_rel_path}")
                    continue
                if manual_coords.is_file():
                    coords_path = manual_coords
                    coords_source = "manual"
                elif original_coords.is_file():
                    coords_path = original_coords
                    coords_source = "original"
                else:
                    warnings.append(f"missing manual/original coordinates: {slide_rel_path}")
                    continue

            slide_id = slide_id_from_rel_path(slide_rel_path)
            if slide_id in seen_ids:
                raise ValueError(f"slide_id collision: {slide_id} ({slide_rel_path})")
            seen_ids.add(slide_id)
            parts = slide_rel_path.split("/")
            rows.append(
                {
                    "case_id": parts[0],
                    "slide_id": slide_id,
                    "slide_rel_path": slide_rel_path,
                    "qc_status": source_row.get("qc_status", "").strip(),
                    "coords_source": coords_source,
                    "coords_rel_path": coords_path.relative_to(dataset_root).as_posix(),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "slide_id",
        "slide_rel_path",
        "qc_status",
        "coords_source",
        "coords_rel_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    manual_count = sum(row["coords_source"] == "manual" for row in rows)
    print(f"[OK] manifest -> {output_path}")
    print(f"Slides: {len(rows)} (manual: {manual_count}, original: {len(rows) - manual_count})")
    if warnings:
        warning_path = output_path.with_name(f"{output_path.stem}_warnings.csv")
        with warning_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["warning"])
            writer.writerows([[warning] for warning in warnings])
        print(f"Warnings: {len(warnings)} -> {warning_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
