#!/usr/bin/env python3
"""Export a portable thumbnail dataset for stain-family classification."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image

from build_wsi_curation_manifest import automatic_stain_group


AUTO_GROUP_TO_LABEL = {
    "HE": "HE",
    "IHC": "IHC",
    "special_other": "other",
}
REVIEW_GROUP_TO_LABEL = {**AUTO_GROUP_TO_LABEL, "unknown": "unknown"}
LABEL_IDS = {"HE": 0, "IHC": 1, "other": 2}
TRUE_VALUES = {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="Curated slide manifest containing thumbnail and automatic stain columns.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thumbnail-max-px", type=int, default=1024)
    parser.add_argument("--image-format", choices=["png", "jpeg"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--include-all-quality", action="store_true",
        help="Do not restrict export to include_quality_usable rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"slide_id", "thumbnail_path", "stain_signature"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame.slide_id.duplicated().any():
        duplicates = frame.loc[frame.slide_id.duplicated(), "slide_id"].tolist()[:5]
        raise ValueError(f"Manifest contains duplicate slide_id values: {duplicates}")
    return frame


def usable_quality_mask(frame: pd.DataFrame) -> pd.Series:
    if "include_quality_usable" not in frame:
        return pd.Series(True, index=frame.index)
    return frame.include_quality_usable.str.strip().str.lower().isin(TRUE_VALUES)


def resolve_thumbnail(raw_path: str, manifest_path: Path) -> Path:
    source = Path(raw_path)
    candidates = [source]
    if not source.is_absolute():
        candidates.extend([manifest_path.parent / source, Path.cwd() / source])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(raw_path)


def safe_image_name(slide_id: str, image_format: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", slide_id).strip("._") or "slide"
    digest = hashlib.sha1(slide_id.encode("utf-8")).hexdigest()[:10]
    extension = "png" if image_format == "png" else "jpg"
    return f"{safe_id[:120]}__{digest}.{extension}"


def write_thumbnail(
    source: Path, destination: Path, max_px: int, image_format: str, quality: int,
) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        if image_format == "png":
            image.save(destination, format="PNG", optimize=True)
        else:
            image.save(destination, format="JPEG", quality=quality, optimize=True)


def prepare_output(path: Path, overwrite: bool) -> Path:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    image_dir = path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir


def main() -> int:
    args = parse_args()
    if args.thumbnail_max_px <= 0:
        raise ValueError("--thumbnail-max-px must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    frame = read_manifest(args.manifest)
    if not args.include_all_quality:
        frame = frame[usable_quality_mask(frame)].copy()
    if frame.empty:
        raise ValueError("No slides remain after quality filtering")

    image_dir = prepare_output(args.output_dir, args.overwrite)
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    passthrough = [
        "source_dataset", "slide_rel_path", "event_key", "patient_id",
        "gold_pathology_id_match", "stain_signature", "stain_group",
        "stain_raw", "stain_source", "stain_confidence", "stain_note",
        "stain_color_cluster",
        "include_quality_usable", "include_quality_clean",
    ]

    for row in frame.to_dict("records"):
        slide_id = str(row["slide_id"])
        image_name = safe_image_name(slide_id, args.image_format)
        try:
            source = resolve_thumbnail(str(row["thumbnail_path"]), args.manifest)
            write_thumbnail(
                source, image_dir / image_name,
                max_px=args.thumbnail_max_px, image_format=args.image_format,
                quality=args.jpeg_quality,
            )
        except Exception as exc:
            failures.append({
                "slide_id": slide_id,
                "error": type(exc).__name__,
                "thumbnail_path": str(row["thumbnail_path"]),
            })
            continue

        signature = str(row["stain_signature"]).strip()
        auto_group, auto_raw, auto_source, auto_confidence = automatic_stain_group(signature)
        filename_label = (
            AUTO_GROUP_TO_LABEL.get(auto_group, "") if auto_source == "filename_rule" else ""
        )
        manual_group = str(row.get("stain_group", "")).strip()
        manual_source = str(row.get("stain_source", "")).strip()
        manual_label = (
            REVIEW_GROUP_TO_LABEL.get(manual_group, "")
            if manual_source == "slide_review" else ""
        )
        if filename_label and manual_label:
            label_relation = "agreement" if filename_label == manual_label else "conflict"
        elif filename_label:
            label_relation = "filename_only"
        elif manual_label:
            label_relation = "manual_only"
        else:
            label_relation = "unlabeled"
        record: dict[str, object] = {
            "slide_id": slide_id,
            "image_path": f"images/{image_name}",
            "filename_label": filename_label,
            "filename_label_id": LABEL_IDS.get(filename_label, ""),
            "manual_label": manual_label,
            "manual_label_id": LABEL_IDS.get(manual_label, ""),
            "label_relation": label_relation,
            "stain_detail_auto": auto_raw,
            "stain_group_auto": auto_group,
            "stain_source_auto": auto_source,
            "stain_confidence_auto": auto_confidence,
            "manifest_stain_group_auto": str(row.get("stain_group_auto", "")),
            "manifest_stain_detail_auto": str(row.get("stain_raw_auto", "")),
        }
        record.update({column: str(row.get(column, "")) for column in passthrough})
        records.append(record)

    if not records:
        raise RuntimeError("No readable thumbnails were exported")

    dataset = pd.DataFrame(records).sort_values(
        ["label_relation", "filename_label", "manual_label", "slide_id"]
    )
    dataset.to_csv(args.output_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig")

    filename_labels = dataset[dataset.filename_label.ne("")].copy()
    filename_labels[
        ["slide_id", "image_path", "filename_label", "filename_label_id",
         "stain_detail_auto", "stain_signature", "source_dataset", "slide_rel_path"]
    ].rename(
        columns={
            "filename_label": "label",
            "filename_label_id": "label_id",
            "stain_detail_auto": "stain_detail",
        }
    ).to_csv(
        args.output_dir / "filename_rule_labels.csv", index=False, encoding="utf-8-sig"
    )

    manual_labels = dataset[dataset.manual_label.ne("")].copy()
    manual_columns = [
        "slide_id", "image_path", "manual_label", "manual_label_id", "stain_raw",
        "stain_confidence", "stain_note", "stain_signature", "stain_color_cluster",
        "source_dataset", "slide_rel_path",
    ]
    manual_columns = [column for column in manual_columns if column in manual_labels]
    manual_labels[manual_columns].rename(
        columns={
            "manual_label": "label",
            "manual_label_id": "label_id",
            "stain_raw": "stain_detail",
        }
    ).to_csv(
        args.output_dir / "manual_review_labels.csv", index=False, encoding="utf-8-sig"
    )

    summary_parts = []
    for label_set, column in [
        ("filename_rule", "filename_label"),
        ("manual_review", "manual_label"),
    ]:
        part = (
            dataset.loc[dataset[column].ne(""), column]
            .value_counts()
            .rename_axis("label")
            .rename("slides")
            .reset_index()
        )
        part.insert(0, "label_set", label_set)
        summary_parts.append(part)
    summary = pd.concat(summary_parts, ignore_index=True)
    summary.to_csv(args.output_dir / "label_summary.csv", index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(
            args.output_dir / "thumbnail_failures.csv", index=False, encoding="utf-8-sig"
        )

    print(f"[OK] stain-classifier dataset: {args.output_dir}")
    print(
        f"Slides exported: {len(dataset)} (filename labels={len(filename_labels)}, "
        f"manual labels={len(manual_labels)}, format={args.image_format})"
    )
    print(summary.to_string(index=False))
    if failures:
        print(f"[WARN] Thumbnail failures: {len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
