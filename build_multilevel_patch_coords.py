#!/usr/bin/env python3
"""Generate independent patch grids for a tiled DICOM WSI pyramid level.

The AtlasPatch thumbnail mask is reused without modification. Manual masks take
precedence over automatic masks, matching the L0 feature pipeline. Coordinates
are generated in the selected pyramid level's pixel space and include level-0
equivalents for later visualization.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from dicom_pyramid import PyramidLevel, discover_pyramid_levels


DEFAULT_QC_MANIFEST = "_qc/mask_qc_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--qc-manifest", type=Path, default=Path(DEFAULT_QC_MANIFEST))
    parser.add_argument("--output-dir-name", default="atlaspatch")
    parser.add_argument("--pyramid-level", type=int, required=True)
    parser.add_argument("--coords-filename", default=None)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--coord-mode", choices=["center", "ratio"], default="center")
    parser.add_argument("--tissue-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--slide-rel-path",
        action="append",
        default=[],
        help="Restrict generation to one or more exact dataset-relative slide paths.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--failure-log", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path, dataset_root: Path) -> Path:
    return path if path.is_absolute() else dataset_root / path


def decode_attr(value: object, default: str) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value) if value is not None else default


def select_mask(
    atlaspatch_dir: Path,
) -> tuple[Path, str, str, float]:
    manual_mask = atlaspatch_dir / "tissue_mask_manual.png"
    auto_mask = atlaspatch_dir / "tissue_mask.png"
    manual_coords = atlaspatch_dir / "patch_coords_manual.h5"
    original_coords = atlaspatch_dir / "patch_coords.h5"

    if manual_mask.is_file():
        if not manual_coords.is_file():
            raise ValueError(f"Manual mask has no manual coordinates: {atlaspatch_dir}")
        mask_path = manual_mask
        base_coords = manual_coords
        mask_source = "manual"
    else:
        if not auto_mask.is_file():
            raise FileNotFoundError(f"Missing automatic tissue mask: {auto_mask}")
        if not original_coords.is_file():
            raise FileNotFoundError(f"Missing original coordinates: {original_coords}")
        mask_path = auto_mask
        base_coords = original_coords
        mask_source = "original"

    with h5py.File(base_coords, "r") as handle:
        coord_mode = decode_attr(handle.attrs.get("coord_mode"), "center")
        tissue_ratio_threshold = float(handle.attrs.get("tissue_ratio_threshold", 0.10))
    if coord_mode not in {"center", "ratio"}:
        coord_mode = "center"
    return mask_path, mask_source, coord_mode, tissue_ratio_threshold


def build_coords(
    mask: np.ndarray,
    level: PyramidLevel,
    patch_size: int,
    step_size: int,
    coord_mode: str,
    tissue_ratio_threshold: float,
) -> np.ndarray:
    mask_h, mask_w = mask.shape
    coords: list[tuple[int, int, int, int, int]] = []
    half = patch_size // 2

    for y in range(0, max(level.total_height - patch_size + 1, 1), step_size):
        for x in range(0, max(level.total_width - patch_size + 1, 1), step_size):
            if coord_mode == "center":
                my = min(mask_h - 1, max(0, round((y + half) * mask_h / level.total_height)))
                mx = min(mask_w - 1, max(0, round((x + half) * mask_w / level.total_width)))
                keep = bool(mask[my, mx])
            else:
                my0 = min(mask_h, max(0, math.floor(y * mask_h / level.total_height)))
                my1 = min(
                    mask_h,
                    max(my0 + 1, math.ceil((y + patch_size) * mask_h / level.total_height)),
                )
                mx0 = min(mask_w, max(0, math.floor(x * mask_w / level.total_width)))
                mx1 = min(
                    mask_w,
                    max(mx0 + 1, math.ceil((x + patch_size) * mask_w / level.total_width)),
                )
                keep = float(mask[my0:my1, mx0:mx1].mean()) >= tissue_ratio_threshold
            if keep:
                coords.append((x, y, patch_size, patch_size, level.index))

    level_coords = np.asarray(coords, dtype=np.int32).reshape((-1, 5))
    return level_coords


def level0_coordinates(level_coords: np.ndarray, level: PyramidLevel, level0: PyramidLevel) -> np.ndarray:
    if not len(level_coords):
        return np.empty((0, 2), dtype=np.int32)
    x_scale = level0.total_width / level.total_width
    y_scale = level0.total_height / level.total_height
    return np.column_stack(
        (
            np.rint(level_coords[:, 0] * x_scale),
            np.rint(level_coords[:, 1] * y_scale),
        )
    ).astype(np.int32, copy=False)


def write_coordinate_h5(
    out_path: Path,
    level_coords: np.ndarray,
    coords_level0: np.ndarray,
    level: PyramidLevel,
    level0: PyramidLevel,
    mask_path: Path,
    mask_source: str,
    coord_mode: str,
    tissue_ratio_threshold: float,
    patch_size: int,
    step_size: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("coords", data=level_coords, compression="gzip")
        handle.create_dataset("coords_level0", data=coords_level0, compression="gzip")
        handle.attrs["num_patches"] = int(len(level_coords))
        handle.attrs["pyramid_level"] = int(level.index)
        handle.attrs["coordinate_space"] = "pyramid_level"
        handle.attrs["patch_size"] = int(patch_size)
        handle.attrs["patch_size_level"] = int(patch_size)
        handle.attrs["step_size_level"] = int(step_size)
        handle.attrs["patch_size_level0_x"] = int(
            round(patch_size * level0.total_width / level.total_width)
        )
        handle.attrs["patch_size_level0_y"] = int(
            round(patch_size * level0.total_height / level.total_height)
        )
        handle.attrs["mask_path"] = str(mask_path)
        handle.attrs["mask_source"] = mask_source
        handle.attrs["coordinate_source"] = f"{mask_source} AtlasPatch thumbnail mask"
        handle.attrs["coord_mode"] = coord_mode
        handle.attrs["tissue_ratio_threshold"] = float(tissue_ratio_threshold)
        handle.attrs["mpp_x_um"] = np.nan if level.mpp_x_um is None else level.mpp_x_um
        handle.attrs["mpp_y_um"] = np.nan if level.mpp_y_um is None else level.mpp_y_um
        handle.attrs["level_total_pixel_matrix"] = np.asarray(
            [level.total_width, level.total_height], dtype=np.int64
        )
        handle.attrs["level0_total_pixel_matrix"] = np.asarray(
            [level0.total_width, level0.total_height], dtype=np.int64
        )
        handle.attrs["source_dicom_paths"] = "|".join(str(path) for path in level.paths)


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_rel_path", "error_type", "error_message"])
        writer.writeheader()
        writer.writerows(failures)


def main() -> int:
    args = parse_args()
    if args.pyramid_level < 0:
        raise ValueError("--pyramid-level must be non-negative")
    if args.patch_size < 1 or args.step_size < 1:
        raise ValueError("--patch-size and --step-size must be positive")

    dataset_root = args.dataset_root.expanduser().resolve()
    qc_manifest = resolve_path(args.qc_manifest, dataset_root)
    if not qc_manifest.is_file():
        raise FileNotFoundError(f"QC manifest not found: {qc_manifest}")
    coords_filename = args.coords_filename or f"patch_coords_l{args.pyramid_level}.h5"
    failure_log = args.failure_log or Path(
        f"_clam/multilevel_coords_l{args.pyramid_level}_failures.csv"
    )
    failure_log = resolve_path(failure_log, dataset_root)
    requested_paths = {value.strip().replace("\\", "/") for value in args.slide_rel_path if value.strip()}

    rows: list[dict[str, str]] = []
    with qc_manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            slide_rel_path = row.get("slide_rel_path", "").strip().replace("\\", "/")
            if slide_rel_path and (not requested_paths or slide_rel_path in requested_paths):
                rows.append({"slide_rel_path": slide_rel_path})
    if requested_paths:
        found_paths = {row["slide_rel_path"] for row in rows}
        missing_paths = sorted(requested_paths - found_paths)
        if missing_paths:
            raise ValueError(f"Requested slide paths missing from QC manifest: {missing_paths}")
    if args.limit is not None:
        rows = rows[: args.limit]

    print(
        f"Generating L{args.pyramid_level} coordinates for {len(rows)} slide(s): "
        f"{coords_filename}"
    )
    completed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        slide_rel_path = row["slide_rel_path"]
        try:
            slide_dir = dataset_root / Path(slide_rel_path)
            atlaspatch_dir = slide_dir / args.output_dir_name
            out_path = atlaspatch_dir / coords_filename
            if args.skip_existing and out_path.is_file():
                skipped += 1
                print(f"[SKIP] [{index}/{len(rows)}] {slide_rel_path}")
                continue

            mask_path, mask_source, coord_mode, tissue_ratio_threshold = select_mask(atlaspatch_dir)
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            levels = discover_pyramid_levels(slide_dir)
            if args.pyramid_level >= len(levels):
                raise ValueError(f"Only {len(levels)} pyramid level(s) available")
            level = levels[args.pyramid_level]
            level0 = levels[0]
            level_coords = build_coords(
                mask,
                level,
                args.patch_size,
                args.step_size,
                coord_mode,
                tissue_ratio_threshold,
            )
            if not len(level_coords):
                raise ValueError("No tissue patches selected")
            coords_level0 = level0_coordinates(level_coords, level, level0)
            write_coordinate_h5(
                out_path,
                level_coords,
                coords_level0,
                level,
                level0,
                mask_path,
                mask_source,
                coord_mode,
                tissue_ratio_threshold,
                args.patch_size,
                args.step_size,
            )
            completed += 1
            mpp = "unknown" if level.mpp_x_um is None else f"{level.mpp_x_um:.3f} um/px"
            print(f"[OK] [{index}/{len(rows)}] {slide_rel_path}: {len(level_coords)} patches ({mpp})")
        except Exception as exc:
            failures.append(
                {
                    "slide_rel_path": slide_rel_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print(f"[FAIL] [{index}/{len(rows)}] {slide_rel_path}: {type(exc).__name__}: {exc}")

    write_failures(failure_log, failures)
    print(f"Completed: {completed}; skipped: {skipped}; failures: {len(failures)}")
    print(f"Failure log: {failure_log}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
