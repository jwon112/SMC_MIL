"""OpenSlide metadata helpers for 3DHISTECH MRXS WSI pyramids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openslide


@dataclass(frozen=True)
class MrxsPyramidLevel:
    index: int
    width: int
    height: int
    downsample: float
    mpp_x_um: float | None
    mpp_y_um: float | None


def choose_mrxs(slide_dir: Path) -> Path:
    paths = sorted(path for path in slide_dir.glob("*.mrxs") if path.is_file())
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one .mrxs file in {slide_dir}, found {len(paths)}")
    return paths[0]


def discover_mrxs_pyramid_levels(mrxs_path: Path) -> tuple[MrxsPyramidLevel, ...]:
    with openslide.OpenSlide(str(mrxs_path)) as slide:
        try:
            base_mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            base_mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        except (KeyError, TypeError, ValueError):
            base_mpp_x = base_mpp_y = None
        return tuple(
            MrxsPyramidLevel(
                index=index,
                width=int(dimensions[0]),
                height=int(dimensions[1]),
                downsample=float(slide.level_downsamples[index]),
                mpp_x_um=None if base_mpp_x is None else base_mpp_x * slide.level_downsamples[index],
                mpp_y_um=None if base_mpp_y is None else base_mpp_y * slide.level_downsamples[index],
            )
            for index, dimensions in enumerate(slide.level_dimensions)
        )
