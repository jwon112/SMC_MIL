#!/usr/bin/env python3
"""Generate level-specific patch grids for MRXS WSI, using AtlasPatch masks.

Manual masks take precedence over automatic AtlasPatch masks. Each coordinate
H5 records both selected-level and level-0 coordinates so feature bags remain
comparable to the DICOM multi-resolution pipeline.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from mrxs_pyramid import MrxsPyramidLevel, choose_mrxs, discover_mrxs_pyramid_levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--qc-manifest", type=Path, default=Path("_qc/mask_qc_labels.csv"))
    parser.add_argument("--output-dir-name", default="atlaspatch")
    parser.add_argument("--pyramid-level", type=int, required=True)
    parser.add_argument("--coords-filename", default=None)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--coord-mode", choices=["center", "ratio"], default="center")
    parser.add_argument("--tissue-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--failure-log", type=Path)
    return parser.parse_args()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def select_mask(atlaspatch_dir: Path) -> tuple[Path, str, str, float]:
    manual_mask = atlaspatch_dir / "tissue_mask_manual.png"
    auto_mask = atlaspatch_dir / "tissue_mask.png"
    manual_coords = atlaspatch_dir / "patch_coords_manual.h5"
    base_coords = manual_coords if manual_mask.is_file() else atlaspatch_dir / "patch_coords.h5"
    mask_path = manual_mask if manual_mask.is_file() else auto_mask
    source = "manual" if manual_mask.is_file() else "original"
    if not mask_path.is_file() or not base_coords.is_file():
        raise FileNotFoundError(f"Missing {source} AtlasPatch mask/coordinates: {atlaspatch_dir}")
    with h5py.File(base_coords, "r") as handle:
        mode = handle.attrs.get("coord_mode", "center")
        if isinstance(mode, bytes):
            mode = mode.decode("utf-8")
        threshold = float(handle.attrs.get("tissue_ratio_threshold", 0.10))
    return mask_path, source, mode if mode in {"center", "ratio"} else "center", threshold


def build_coords(mask: np.ndarray, level: MrxsPyramidLevel, patch_size: int, step_size: int, mode: str, threshold: float) -> np.ndarray:
    mask_h, mask_w = mask.shape
    values: list[tuple[int, int, int, int, int]] = []
    half = patch_size // 2
    for y in range(0, max(level.height - patch_size + 1, 1), step_size):
        for x in range(0, max(level.width - patch_size + 1, 1), step_size):
            if mode == "center":
                mx = min(mask_w - 1, max(0, round((x + half) * mask_w / level.width)))
                my = min(mask_h - 1, max(0, round((y + half) * mask_h / level.height)))
                keep = bool(mask[my, mx])
            else:
                mx0 = min(mask_w, max(0, math.floor(x * mask_w / level.width)))
                mx1 = min(mask_w, max(mx0 + 1, math.ceil((x + patch_size) * mask_w / level.width)))
                my0 = min(mask_h, max(0, math.floor(y * mask_h / level.height)))
                my1 = min(mask_h, max(my0 + 1, math.ceil((y + patch_size) * mask_h / level.height)))
                keep = float(mask[my0:my1, mx0:mx1].mean()) >= threshold
            if keep:
                values.append((x, y, patch_size, patch_size, level.index))
    return np.asarray(values, dtype=np.int32).reshape((-1, 5))


def write_coords(path: Path, values: np.ndarray, level: MrxsPyramidLevel, level0: MrxsPyramidLevel, mrxs_path: Path, mask_path: Path, source: str, mode: str, threshold: float, patch_size: int, step_size: int) -> None:
    coords_level0 = np.column_stack((
        np.rint(values[:, 0] * level0.width / level.width),
        np.rint(values[:, 1] * level0.height / level.height),
    )).astype(np.int32, copy=False) if len(values) else np.empty((0, 2), dtype=np.int32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coords", data=values, compression="gzip")
        handle.create_dataset("coords_level0", data=coords_level0, compression="gzip")
        handle.attrs["num_patches"] = len(values)
        handle.attrs["pyramid_level"] = level.index
        handle.attrs["coordinate_space"] = "pyramid_level"
        handle.attrs["patch_size"] = patch_size
        handle.attrs["patch_size_level"] = patch_size
        handle.attrs["step_size_level"] = step_size
        handle.attrs["patch_size_level0_x"] = round(patch_size * level0.width / level.width)
        handle.attrs["patch_size_level0_y"] = round(patch_size * level0.height / level.height)
        handle.attrs["mask_path"] = str(mask_path)
        handle.attrs["mask_source"] = source
        handle.attrs["coordinate_source"] = f"{source} AtlasPatch thumbnail mask"
        handle.attrs["coord_mode"] = mode
        handle.attrs["tissue_ratio_threshold"] = threshold
        handle.attrs["mpp_x_um"] = np.nan if level.mpp_x_um is None else level.mpp_x_um
        handle.attrs["mpp_y_um"] = np.nan if level.mpp_y_um is None else level.mpp_y_um
        handle.attrs["level_total_pixel_matrix"] = np.asarray([level.width, level.height], dtype=np.int64)
        handle.attrs["level0_total_pixel_matrix"] = np.asarray([level0.width, level0.height], dtype=np.int64)
        handle.attrs["source_mrxs_path"] = str(mrxs_path)


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    manifest = resolve_path(args.qc_manifest, root)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8-sig")))
    paths = [row.get("slide_rel_path", "").strip().replace("\\", "/") for row in rows]
    paths = [path for path in paths if path]
    if args.limit is not None:
        paths = paths[:args.limit]
    filename = args.coords_filename or f"patch_coords_l{args.pyramid_level}.h5"
    failures: list[dict[str, str]] = []
    completed = skipped = 0
    for index, rel_path in enumerate(paths, start=1):
        try:
            slide_dir = root / rel_path
            atlaspatch_dir = slide_dir / args.output_dir_name
            output = atlaspatch_dir / filename
            if args.skip_existing and output.is_file():
                skipped += 1
                continue
            mrxs_path = choose_mrxs(slide_dir)
            levels = discover_mrxs_pyramid_levels(mrxs_path)
            if args.pyramid_level >= len(levels):
                raise ValueError(f"Requested L{args.pyramid_level}, but slide has {len(levels)} levels")
            mask_path, source, mode, threshold = select_mask(atlaspatch_dir)
            values = build_coords(np.asarray(Image.open(mask_path).convert("L")) > 0, levels[args.pyramid_level], args.patch_size, args.step_size, mode, threshold)
            if not len(values):
                raise ValueError("No tissue patches selected")
            write_coords(output, values, levels[args.pyramid_level], levels[0], mrxs_path, mask_path, source, mode, threshold, args.patch_size, args.step_size)
            completed += 1
            print(f"[OK] [{index}/{len(paths)}] {rel_path}: {len(values)} patches ({levels[args.pyramid_level].mpp_x_um:.3f} um/px)")
        except Exception as exc:  # noqa: BLE001
            failures.append({"slide_rel_path": rel_path, "error_type": type(exc).__name__, "error_message": str(exc)})
            print(f"[FAIL] [{index}/{len(paths)}] {rel_path}: {exc}")
    failure_log = resolve_path(args.failure_log or Path(f"_clam/mrxs_multilevel_coords_l{args.pyramid_level}_failures.csv"), root)
    failure_log.parent.mkdir(parents=True, exist_ok=True)
    with failure_log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_rel_path", "error_type", "error_message"])
        writer.writeheader(); writer.writerows(failures)
    print(f"Completed: {completed}; skipped: {skipped}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
