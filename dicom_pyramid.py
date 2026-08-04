"""Metadata helpers for tiled DICOM WSI pyramid levels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom


@dataclass(frozen=True)
class PyramidLevel:
    """One resolution level, potentially composed of multiple DICOM instances."""

    index: int
    paths: tuple[Path, ...]
    total_width: int
    total_height: int
    tile_width: int
    tile_height: int
    mpp_x_um: float | None
    mpp_y_um: float | None


def is_tiled_volume_wsi(dataset: pydicom.Dataset) -> bool:
    image_type = [str(value).upper() for value in getattr(dataset, "ImageType", [])]
    return (
        "VOLUME" in image_type
        and int(getattr(dataset, "TotalPixelMatrixColumns", 0) or 0) > 0
        and int(getattr(dataset, "TotalPixelMatrixRows", 0) or 0) > 0
        and int(getattr(dataset, "Columns", 0) or 0) > 0
        and int(getattr(dataset, "Rows", 0) or 0) > 0
    )


def pixel_spacing_um(dataset: pydicom.Dataset) -> tuple[float | None, float | None]:
    """Return x/y microns-per-pixel when the DICOM metadata provides it."""

    try:
        values = dataset.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        # DICOM PixelSpacing is row (y), then column (x), in millimetres.
        return float(values[1]) * 1000.0, float(values[0]) * 1000.0
    except Exception:
        pass

    try:
        width = int(dataset.TotalPixelMatrixColumns)
        height = int(dataset.TotalPixelMatrixRows)
        return (
            float(dataset.ImagedVolumeWidth) * 1000.0 / width,
            float(dataset.ImagedVolumeHeight) * 1000.0 / height,
        )
    except Exception:
        return None, None


def discover_pyramid_levels(slide_dir: Path) -> tuple[PyramidLevel, ...]:
    """Discover tissue VOLUME levels ordered from highest to lowest resolution."""

    grouped: dict[tuple[int, int, int, int], list[Path]] = {}
    metadata_by_path: dict[Path, pydicom.Dataset] = {}
    for path in sorted(slide_dir.glob("*.dcm")):
        try:
            dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not is_tiled_volume_wsi(dataset):
            continue
        key = (
            int(dataset.TotalPixelMatrixColumns),
            int(dataset.TotalPixelMatrixRows),
            int(dataset.Columns),
            int(dataset.Rows),
        )
        grouped.setdefault(key, []).append(path)
        metadata_by_path[path] = dataset

    levels: list[PyramidLevel] = []
    for index, (key, paths) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0][0] * item[0][1], reverse=True)
    ):
        total_width, total_height, tile_width, tile_height = key
        first = metadata_by_path[paths[0]]
        mpp_x_um, mpp_y_um = pixel_spacing_um(first)
        levels.append(
            PyramidLevel(
                index=index,
                paths=tuple(paths),
                total_width=total_width,
                total_height=total_height,
                tile_width=tile_width,
                tile_height=tile_height,
                mpp_x_um=mpp_x_um,
                mpp_y_um=mpp_y_um,
            )
        )
    return tuple(levels)


def get_pyramid_level(slide_dir: Path, level_index: int) -> PyramidLevel:
    levels = discover_pyramid_levels(slide_dir)
    if not levels:
        raise ValueError(f"No tiled VOLUME DICOM found: {slide_dir}")
    if level_index < 0 or level_index >= len(levels):
        raise ValueError(
            f"Requested pyramid level {level_index}, but {slide_dir} has {len(levels)} level(s)"
        )
    return levels[level_index]
