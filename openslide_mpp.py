"""OpenSlide helpers for reading WSI regions at a requested physical resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import openslide
from PIL import Image


@dataclass(frozen=True)
class SlideGeometry:
    width: int
    height: int
    mpp_x: float
    mpp_y: float


def get_slide_geometry(slide: openslide.OpenSlide, fallback_mpp: float | None = None) -> SlideGeometry:
    def read_mpp(name: str) -> float:
        try:
            value = float(slide.properties[name])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0:
            if fallback_mpp is None:
                raise ValueError(f"Missing usable {name} metadata")
            value = fallback_mpp
        return value

    return SlideGeometry(
        width=int(slide.dimensions[0]),
        height=int(slide.dimensions[1]),
        mpp_x=read_mpp(openslide.PROPERTY_NAME_MPP_X),
        mpp_y=read_mpp(openslide.PROPERTY_NAME_MPP_Y),
    )


def composite_white(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    return background.convert("RGB")


def best_level_for_mpp(slide: openslide.OpenSlide, base_mpp: float, target_mpp: float) -> int:
    desired_downsample = max(1.0, target_mpp / base_mpp)
    valid = [
        (index, float(downsample))
        for index, downsample in enumerate(slide.level_downsamples)
        if float(downsample) <= desired_downsample * 1.001
    ]
    return max(valid, key=lambda item: item[1])[0] if valid else 0


def read_region_at_mpp(
    slide: openslide.OpenSlide,
    x_level0: int,
    y_level0: int,
    output_size: int,
    target_mpp: float,
    geometry: SlideGeometry,
) -> Image.Image:
    """Read a square patch whose field of view is output_size * target_mpp."""
    level = best_level_for_mpp(slide, min(geometry.mpp_x, geometry.mpp_y), target_mpp)
    downsample = float(slide.level_downsamples[level])
    span_x = max(1, round(output_size * target_mpp / geometry.mpp_x))
    span_y = max(1, round(output_size * target_mpp / geometry.mpp_y))
    read_w = max(1, math.ceil(span_x / downsample))
    read_h = max(1, math.ceil(span_y / downsample))
    image = composite_white(slide.read_region((int(x_level0), int(y_level0)), level, (read_w, read_h)))
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return image


def read_roi_thumbnail(
    source_path: Path,
    roi: tuple[int, int, int, int] | None,
    max_size: int,
    fallback_mpp: float | None = None,
) -> tuple[Image.Image, SlideGeometry, tuple[int, int, int, int]]:
    with openslide.OpenSlide(str(source_path)) as slide:
        geometry = get_slide_geometry(slide, fallback_mpp)
        x, y, width, height = roi or (0, 0, geometry.width, geometry.height)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid ROI: {(x, y, width, height)}")
        scale = min(1.0, max_size / max(width, height))
        target_w, target_h = max(1, round(width * scale)), max(1, round(height * scale))
        desired_downsample = 1.0 / scale
        level = min(
            range(slide.level_count),
            key=lambda i: abs(math.log(max(float(slide.level_downsamples[i]), 1e-9) / desired_downsample)),
        )
        downsample = float(slide.level_downsamples[level])
        read_size = (max(1, math.ceil(width / downsample)), max(1, math.ceil(height / downsample)))
        image = composite_white(slide.read_region((x, y), level, read_size))
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return image, geometry, (x, y, width, height)
