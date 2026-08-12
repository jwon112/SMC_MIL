#!/usr/bin/env python3
"""Extract CLAM-compatible feature bags from MRXS WSI pyramid levels.

The manifest format is shared with the DICOM pipeline. Coordinate H5 files
determine the MRXS pyramid level; selected-level coordinates are converted to
level-0 coordinates in the output H5 for consistent downstream visualization.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch
from PIL import Image

from extract_features_dicom import (
    SlideSpec,
    append_batch,
    output_paths,
    read_coordinates,
    read_manifest,
    sample_coordinate_set,
    write_failures,
)
from models import get_encoder
from mrxs_pyramid import choose_mrxs


class MrxsPyramidReader:
    """Random-access OpenSlide reader for a single MRXS pyramid level."""

    def __init__(self, mrxs_path: Path, pyramid_level: int):
        self.path = mrxs_path
        self.slide = openslide.OpenSlide(str(mrxs_path))
        if pyramid_level < 0 or pyramid_level >= self.slide.level_count:
            self.slide.close()
            raise ValueError(
                f"Requested pyramid level {pyramid_level}, but {mrxs_path} has {self.slide.level_count} levels"
            )
        self.level = pyramid_level
        self.total_w, self.total_h = (int(value) for value in self.slide.level_dimensions[pyramid_level])
        self.level0_w, self.level0_h = (int(value) for value in self.slide.dimensions)
        self.downsample = float(self.slide.level_downsamples[pyramid_level])
        try:
            base_mpp_x = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X])
            base_mpp_y = float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        except (KeyError, TypeError, ValueError):
            base_mpp_x = base_mpp_y = float("nan")
        self.mpp_x_um = base_mpp_x * self.downsample
        self.mpp_y_um = base_mpp_y * self.downsample

    def close(self) -> None:
        self.slide.close()

    def read_patch(self, x: int, y: int, size: int) -> Image.Image:
        level0_x = int(round(x * self.downsample))
        level0_y = int(round(y * self.downsample))
        rgba = self.slide.read_region((level0_x, level0_y), self.level, (size, size))
        # OpenSlide uses transparent black outside slide bounds; preserve a white background.
        image = Image.new("RGBA", rgba.size, "white")
        image.alpha_composite(rgba)
        return image.convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feat-dir", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="uni_v2",
        choices=["resnet50_trunc", "uni_v1", "uni_v2", "uni_v2_l", "conch_v1", "conch_v1_5"],
    )
    parser.add_argument("--target-patch-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def encode_slide(spec: SlideSpec, root: Path, h5_path: Path, pt_path: Path, model: torch.nn.Module, transform: object, device: torch.device, batch_size: int, amp: bool, max_patches: int | None, sample_seed: int) -> int:
    coordinate_set, source_patch_count = sample_coordinate_set(read_coordinates(spec.coords_path), max_patches, sample_seed)
    if not len(coordinate_set.coords_level):
        raise ValueError(f"No selected patches: {spec.coords_path}")
    slide_dir = root / Path(spec.slide_rel_path)
    reader = MrxsPyramidReader(choose_mrxs(slide_dir), coordinate_set.pyramid_level)
    coords_level0 = coordinate_set.coords_level0
    if coords_level0 is None:
        coords_level0 = np.column_stack((
            np.rint(coordinate_set.coords_level[:, 0] * reader.level0_w / reader.total_w),
            np.rint(coordinate_set.coords_level[:, 1] * reader.level0_h / reader.total_h),
        )).astype(np.int32, copy=False)
    patch_size = coordinate_set.patch_size
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    h5_tmp, pt_tmp = h5_path.with_suffix(".h5.tmp"), pt_path.with_suffix(".pt.tmp")
    for path in (h5_tmp, pt_tmp):
        if path.exists():
            path.unlink()
    try:
        with h5py.File(h5_tmp, "w") as output:
            output.attrs["slide_id"] = spec.slide_id
            output.attrs["case_id"] = spec.case_id
            output.attrs["slide_rel_path"] = spec.slide_rel_path
            output.attrs["coords_source"] = spec.coords_source
            output.attrs["coords_path"] = str(spec.coords_path)
            output.attrs["patch_level"] = coordinate_set.pyramid_level
            output.attrs["patch_size_level"] = patch_size
            output.attrs["patch_size_level0_x"] = round(patch_size * reader.level0_w / reader.total_w)
            output.attrs["patch_size_level0_y"] = round(patch_size * reader.level0_h / reader.total_h)
            output.attrs["coordinate_storage_space"] = "level0"
            output.attrs["source_patch_count"] = source_patch_count
            output.attrs["encoded_patch_count"] = len(coordinate_set.coords_level)
            output.attrs["patch_sampling"] = "full" if len(coordinate_set.coords_level) == source_patch_count else f"random_without_replacement(seed={sample_seed})"
            output.attrs["source_mrxs_path"] = str(reader.path)
            output.attrs["total_pixel_matrix"] = np.asarray([reader.total_w, reader.total_h])
            output.attrs["level0_total_pixel_matrix"] = np.asarray([reader.level0_w, reader.level0_h])
            output.attrs["mpp_x_um"] = reader.mpp_x_um
            output.attrs["mpp_y_um"] = reader.mpp_y_um
            for start in range(0, len(coordinate_set.coords_level), batch_size):
                batch = coordinate_set.coords_level[start : start + batch_size]
                images = [transform(reader.read_patch(int(x), int(y), patch_size)) for x, y in batch]
                inputs = torch.stack(images).to(device, non_blocking=device.type == "cuda")
                with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                    features = model(inputs)
                if not isinstance(features, torch.Tensor):
                    raise TypeError(f"Encoder returned {type(features).__name__}, expected Tensor")
                append_batch(output, features.detach().cpu().float().numpy().astype(np.float32, copy=False), coords_level0[start : start + len(batch)], batch if coordinate_set.pyramid_level else None)
        with h5py.File(h5_tmp, "r") as output:
            torch.save(torch.from_numpy(output["features"][:]), pt_tmp)
        os.replace(h5_tmp, h5_path)
        os.replace(pt_tmp, pt_path)
    except Exception:
        for path in (h5_tmp, pt_tmp):
            if path.exists():
                path.unlink()
        raise
    finally:
        reader.close()
    return len(coordinate_set.coords_level)


def main() -> int:
    args = parse_args()
    root, feat_dir = args.dataset_root.resolve(), args.feat_dir.resolve()
    if args.batch_size < 1 or args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid batch/shard arguments")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    specs = read_manifest(resolve_path(args.manifest, root), root)
    if args.limit is not None:
        specs = specs[:args.limit]
    specs = specs[args.shard_index::args.num_shards]
    print(f"Encoding MRXS shard {args.shard_index + 1}/{args.num_shards}: {len(specs)} slide(s)")
    if args.dry_run:
        for spec in specs:
            coordinates, total = sample_coordinate_set(read_coordinates(spec.coords_path), args.max_patches_per_slide, args.sample_seed)
            print(f"[DRY RUN] {spec.slide_id}: {len(coordinates.coords_level)}/{total} patches, level={coordinates.pyramid_level}, size={coordinates.patch_size}, {spec.coords_source}")
        return 0
    model, transform = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model = model.eval().to(device)
    failures: list[dict[str, str]] = []
    completed = skipped = 0
    for index, spec in enumerate(specs, start=1):
        h5_path, pt_path = output_paths(feat_dir, spec.slide_id)
        if not args.overwrite and h5_path.is_file() and pt_path.is_file():
            skipped += 1; continue
        try:
            count = encode_slide(spec, root, h5_path, pt_path, model, transform, device, args.batch_size, args.amp, args.max_patches_per_slide, args.sample_seed)
            completed += 1
            print(f"[OK] [{index}/{len(specs)}] {spec.slide_id}: {count} patches")
        except Exception as exc:  # noqa: BLE001
            failures.append({"slide_id": spec.slide_id, "slide_rel_path": spec.slide_rel_path, "error_type": type(exc).__name__, "error_message": str(exc)})
            print(f"[FAIL] [{index}/{len(specs)}] {spec.slide_id}: {exc}")
    failure_path = feat_dir / "logs" / f"mrxs_feature_failures_shard_{args.shard_index + 1}_of_{args.num_shards}.csv"
    write_failures(failure_path, failures)
    print(f"Completed: {completed}; skipped: {skipped}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
