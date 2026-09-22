#!/usr/bin/env python3
"""Create review-only high-resolution stain thumbnails from the original WSI.

This deliberately writes outside each slide's ``atlaspatch`` directory, so
existing tissue masks, patch coordinates, and model inputs remain untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dicom_pyramid import discover_pyramid_levels  # noqa: E402
from extract_features_dicom import DicomPyramidReader  # noqa: E402
from mrxs_pyramid import choose_mrxs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--slide-ids-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-role", action="append", default=[],
        help="Exclude rows in --slide-ids-csv whose role equals this value; repeatable.",
    )
    parser.add_argument("--max-px", type=int, default=2560)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def choose_dicom_level(slide_dir: Path, max_px: int) -> int:
    levels = discover_pyramid_levels(slide_dir)
    if not levels:
        raise ValueError(f"No tiled DICOM WSI pyramid: {slide_dir}")
    return next(
        (level.index for level in levels if max(level.total_width, level.total_height) <= max_px),
        levels[-1].index,
    )


def resize_if_needed(image: Image.Image, max_px: int) -> Image.Image:
    if max(image.size) <= max_px:
        return image
    scale = max_px / max(image.size)
    return image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )


def read_dicom_thumbnail(slide_dir: Path, max_px: int) -> Image.Image:
    level = choose_dicom_level(slide_dir, max_px)
    reader = DicomPyramidReader(slide_dir, level, tile_cache_size=64)
    image = reader.read_patch(0, 0, reader.total_w, reader.total_h)
    return resize_if_needed(image, max_px)


def read_mrxs_thumbnail(slide_dir: Path, max_px: int) -> Image.Image:
    import openslide

    with openslide.OpenSlide(str(choose_mrxs(slide_dir))) as slide:
        level = next(
            (index for index, size in enumerate(slide.level_dimensions) if max(size) <= max_px),
            slide.level_count - 1,
        )
        rgba = slide.read_region((0, 0), level, slide.level_dimensions[level])
    canvas = Image.new("RGBA", rgba.size, "white")
    canvas.alpha_composite(rgba)
    return resize_if_needed(canvas.convert("RGB"), max_px)


def read_thumbnail(row: dict[str, str], max_px: int) -> tuple[Image.Image, str]:
    source = Path(row["thumbnail_path"])
    slide_dir = source.parent.parent
    dataset = row.get("source_dataset", "").strip().lower()
    if dataset == "mrxs13" or any(slide_dir.glob("*.mrxs")):
        return read_mrxs_thumbnail(slide_dir, max_px), "mrxs"
    return read_dicom_thumbnail(slide_dir, max_px), "dicom"


def main() -> int:
    args = parse_args()
    if args.max_px < 512:
        raise ValueError("--max-px must be at least 512")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")

    manifest = read_csv(args.manifest)
    queue = read_csv(args.slide_ids_csv)
    if "slide_id" not in manifest or "thumbnail_path" not in manifest:
        raise ValueError("--manifest requires slide_id and thumbnail_path columns")
    if "slide_id" not in queue:
        raise ValueError("--slide-ids-csv requires a slide_id column")
    if manifest.slide_id.duplicated().any():
        raise ValueError("Duplicate slide_id in manifest")

    excluded = {value.strip() for value in args.exclude_role if value.strip()}
    if excluded:
        if "role" not in queue:
            raise ValueError("--exclude-role requires a role column in --slide-ids-csv")
        queue = queue[~queue.role.str.strip().isin(excluded)].copy()
    ids = queue.slide_id.str.strip()
    ids = ids[ids.ne("")].drop_duplicates()
    selected = manifest[manifest.slide_id.str.strip().isin(set(ids))].copy()
    selected = selected.set_index("slide_id").reindex(ids).reset_index()
    missing = sorted(set(ids).difference(set(manifest.slide_id)))
    if missing:
        raise ValueError(f"{len(missing)} requested slide IDs missing from manifest; first: {missing[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    records: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(selected.to_dict("records"), start=1):
        slide_id = str(row["slide_id"])
        destination = image_dir / f"{index:04d}__{slide_id}.jpg"
        try:
            if destination.is_file() and not args.overwrite:
                with Image.open(destination) as image:
                    width, height = image.size
                source_type = "existing"
            else:
                image, source_type = read_thumbnail(row, args.max_px)
                image.save(destination, "JPEG", quality=args.jpeg_quality, optimize=True)
                width, height = image.size
            row.update({
                "thumbnail_path": str(destination.resolve()),
                "review_thumbnail_source": source_type,
                "review_thumbnail_width": str(width),
                "review_thumbnail_height": str(height),
            })
            records.append(row)
            print(f"[OK {index}/{len(selected)}] {slide_id}: {width}x{height} ({source_type})")
        except Exception as exc:  # noqa: BLE001
            failures.append({"slide_id": slide_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[FAIL {index}/{len(selected)}] {slide_id}: {type(exc).__name__}: {exc}")

    output_manifest = args.output_dir / "highres_thumbnail_manifest.csv"
    pd.DataFrame(records).to_csv(output_manifest, index=False, encoding="utf-8-sig")
    if failures:
        pd.DataFrame(failures).to_csv(
            args.output_dir / "thumbnail_failures.csv", index=False, encoding="utf-8-sig"
        )
    print(f"[DONE] thumbnails={len(records)}, failures={len(failures)} -> {output_manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
