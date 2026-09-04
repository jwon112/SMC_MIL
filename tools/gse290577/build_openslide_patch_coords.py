#!/usr/bin/env python3
"""Create physical-resolution patch coordinates from AtlasPatch masks."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openslide_mpp import get_slide_geometry  # noqa: E402


def number(row: dict[str, str], name: str) -> int:
    return int(float(row.get(name, "0") or 0))


def mpp_tag(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def keep_patch(mask: np.ndarray, x: int, y: int, width: int, height: int, span_x: int, span_y: int, mode: str, threshold: float) -> bool:
    mask_h, mask_w = mask.shape
    if mode == "center":
        mx = min(mask_w - 1, max(0, round((x + span_x / 2) * mask_w / width)))
        my = min(mask_h - 1, max(0, round((y + span_y / 2) * mask_h / height)))
        return bool(mask[my, mx])
    x0 = min(mask_w, max(0, math.floor(x * mask_w / width)))
    x1 = min(mask_w, max(x0 + 1, math.ceil((x + span_x) * mask_w / width)))
    y0 = min(mask_h, max(0, math.floor(y * mask_h / height)))
    y1 = min(mask_h, max(y0 + 1, math.ceil((y + span_y) * mask_h / height)))
    return float(mask[y0:y1, x0:x1].mean()) >= threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--processing-root", type=Path, required=True)
    parser.add_argument("--target-mpp", type=float, required=True)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256, help="Stride in target-MPP pixels.")
    parser.add_argument("--coord-mode", choices=["center", "ratio"], default="center")
    parser.add_argument("--tissue-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--fallback-mpp", type=float)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.target_mpp <= 0 or args.patch_size <= 0 or args.stride <= 0:
        raise ValueError("MPP, patch size, and stride must be positive")

    dataset_root, processing_root = args.dataset_root.resolve(), args.processing_root.resolve()
    with args.inventory.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    output_rows: list[dict[str, str]] = []
    tag = mpp_tag(args.target_mpp)
    for index, row in enumerate(rows, 1):
        source = dataset_root / row["source_rel_path"]
        atlaspatch = processing_root / row["processing_rel_path"] / "atlaspatch"
        manual_mask = atlaspatch / "tissue_mask_manual.png"
        mask_path = manual_mask if manual_mask.is_file() else atlaspatch / "tissue_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing tissue mask: {mask_path}")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        with openslide.OpenSlide(str(source)) as slide:
            geometry = get_slide_geometry(slide, args.fallback_mpp)
        roi_x, roi_y = number(row, "roi_x"), number(row, "roi_y")
        roi_w = number(row, "roi_width") or geometry.width
        roi_h = number(row, "roi_height") or geometry.height
        span_x = max(1, round(args.patch_size * args.target_mpp / geometry.mpp_x))
        span_y = max(1, round(args.patch_size * args.target_mpp / geometry.mpp_y))
        step_x = max(1, round(args.stride * args.target_mpp / geometry.mpp_x))
        step_y = max(1, round(args.stride * args.target_mpp / geometry.mpp_y))
        coords = []
        for local_y in range(0, max(roi_h - span_y + 1, 1), step_y):
            for local_x in range(0, max(roi_w - span_x + 1, 1), step_x):
                if keep_patch(mask, local_x, local_y, roi_w, roi_h, span_x, span_y, args.coord_mode, args.tissue_ratio_threshold):
                    coords.append((roi_x + local_x, roi_y + local_y))
        values = np.asarray(coords, dtype=np.int32).reshape((-1, 2))
        coords_path = atlaspatch / f"patch_coords_{tag}mpp.h5"
        with h5py.File(coords_path, "w") as h5:
            h5.create_dataset("coords", data=values, compression="gzip")
            h5.create_dataset("coords_level0", data=values, compression="gzip")
            h5.attrs["num_patches"] = len(values)
            h5.attrs["patch_size"] = args.patch_size
            h5.attrs["patch_size_level"] = args.patch_size
            h5.attrs["pyramid_level"] = 0
            h5.attrs["target_mpp"] = args.target_mpp
            h5.attrs["source_mpp_x"] = geometry.mpp_x
            h5.attrs["source_mpp_y"] = geometry.mpp_y
            h5.attrs["read_span_level0_x"] = span_x
            h5.attrs["read_span_level0_y"] = span_y
            h5.attrs["roi_level0"] = np.asarray([roi_x, roi_y, roi_w, roi_h], dtype=np.int64)
            h5.attrs["mask_path"] = str(mask_path)
        manifest_row = dict(row)
        manifest_row.update({
            "slide_rel_path": row["processing_rel_path"],
            "coords_source": "manual" if mask_path == manual_mask else "atlaspatch",
            "coords_rel_path": coords_path.relative_to(processing_root).as_posix(),
            "target_mpp": str(args.target_mpp),
        })
        output_rows.append(manifest_row)
        print(f"[OK] [{index}/{len(rows)}] {row['slide_id']}: {len(values)} patches at {args.target_mpp:g} um/px")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    print(f"Manifest: {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
