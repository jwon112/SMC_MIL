#!/usr/bin/env python3
"""Extract CLAM-compatible feature bags from OpenSlide WSI at a target MPP."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import openslide
import torch

from extract_features_dicom import append_batch, output_paths, sample_coordinate_set, read_coordinates
from models import get_encoder
from openslide_mpp import get_slide_geometry, read_region_at_mpp


@dataclass(frozen=True)
class OpenSlideSpec:
    row: dict[str, str]
    source_path: Path
    coords_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--processing-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feat-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="uni_v2", choices=["resnet50_trunc", "uni_v1", "uni_v2", "uni_v2_l", "conch_v1", "conch_v1_5"])
    parser.add_argument("--target-patch-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--fallback-mpp", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_specs(manifest: Path, dataset_root: Path, processing_root: Path) -> list[OpenSlideSpec]:
    required = {"slide_id", "source_rel_path", "slide_rel_path", "coords_rel_path", "coords_source", "target_mpp"}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    seen: set[str] = set()
    specs = []
    for row in rows:
        slide_id = row["slide_id"].strip()
        if slide_id in seen:
            raise ValueError(f"Duplicate slide_id: {slide_id}")
        seen.add(slide_id)
        source = Path(row["source_rel_path"])
        coords = Path(row["coords_rel_path"])
        specs.append(OpenSlideSpec(
            row=row,
            source_path=source if source.is_absolute() else dataset_root / source,
            coords_path=coords if coords.is_absolute() else processing_root / coords,
        ))
    return specs


def encode(spec: OpenSlideSpec, h5_path: Path, pt_path: Path, model, transform, device: torch.device, args: argparse.Namespace) -> int:
    coordinate_set, source_count = sample_coordinate_set(
        read_coordinates(spec.coords_path), args.max_patches_per_slide, args.sample_seed
    )
    if not len(coordinate_set.coords_level):
        raise ValueError(f"No selected patches: {spec.coords_path}")
    target_mpp = float(spec.row["target_mpp"])
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    h5_tmp, pt_tmp = h5_path.with_suffix(".h5.tmp"), pt_path.with_suffix(".pt.tmp")
    for path in (h5_tmp, pt_tmp):
        if path.exists():
            path.unlink()
    with openslide.OpenSlide(str(spec.source_path)) as slide:
        geometry = get_slide_geometry(slide, args.fallback_mpp)
        try:
            with h5py.File(h5_tmp, "w") as output:
                for name in ("slide_id", "case_id", "patient_id", "biopsy_id", "source_type", "stain", "stain_group"):
                    if spec.row.get(name, "") != "":
                        output.attrs[name] = spec.row[name]
                output.attrs["slide_rel_path"] = spec.row["slide_rel_path"]
                output.attrs["source_rel_path"] = spec.row["source_rel_path"]
                output.attrs["coords_source"] = spec.row["coords_source"]
                output.attrs["coords_path"] = str(spec.coords_path)
                output.attrs["patch_level"] = 0
                output.attrs["patch_size_level"] = coordinate_set.patch_size
                output.attrs["target_mpp"] = target_mpp
                output.attrs["source_mpp_x"] = geometry.mpp_x
                output.attrs["source_mpp_y"] = geometry.mpp_y
                output.attrs["source_patch_count"] = source_count
                output.attrs["encoded_patch_count"] = len(coordinate_set.coords_level)
                coords = coordinate_set.coords_level
                for start in range(0, len(coords), args.batch_size):
                    batch = coords[start : start + args.batch_size]
                    images = [
                        transform(read_region_at_mpp(slide, int(x), int(y), coordinate_set.patch_size, target_mpp, geometry))
                        for x, y in batch
                    ]
                    inputs = torch.stack(images).to(device, non_blocking=device.type == "cuda")
                    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                        features = model(inputs)
                    if not isinstance(features, torch.Tensor):
                        raise TypeError(f"Encoder returned {type(features).__name__}, expected Tensor")
                    append_batch(output, features.detach().cpu().float().numpy().astype(np.float32, copy=False), batch)
            with h5py.File(h5_tmp, "r") as output:
                torch.save(torch.from_numpy(output["features"][:]), pt_tmp)
            os.replace(h5_tmp, h5_path)
            os.replace(pt_tmp, pt_path)
        except Exception:
            for path in (h5_tmp, pt_tmp):
                if path.exists():
                    path.unlink()
            raise
    return len(coordinate_set.coords_level)


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard arguments")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    specs = load_specs(args.manifest, args.dataset_root.resolve(), args.processing_root.resolve())
    if args.limit is not None:
        specs = specs[: args.limit]
    specs = specs[args.shard_index::args.num_shards]
    print(f"Encoding OpenSlide shard {args.shard_index + 1}/{args.num_shards}: {len(specs)} slide(s)")
    if args.dry_run:
        for spec in specs:
            coords, total = sample_coordinate_set(read_coordinates(spec.coords_path), args.max_patches_per_slide, args.sample_seed)
            print(f"[DRY RUN] {spec.row['slide_id']}: {len(coords.coords_level)}/{total} patches at {spec.row['target_mpp']} um/px")
        return 0
    model, transform = get_encoder(args.model_name, target_img_size=args.target_patch_size)
    model = model.eval().to(device)
    completed = skipped = 0
    failures: list[dict[str, str]] = []
    for index, spec in enumerate(specs, 1):
        h5_path, pt_path = output_paths(args.feat_dir.resolve(), spec.row["slide_id"])
        if not args.overwrite and h5_path.is_file() and pt_path.is_file():
            skipped += 1
            continue
        try:
            count = encode(spec, h5_path, pt_path, model, transform, device, args)
            completed += 1
            print(f"[OK] [{index}/{len(specs)}] {spec.row['slide_id']}: {count} patches")
        except Exception as exc:  # noqa: BLE001
            failures.append({"slide_id": spec.row["slide_id"], "source_rel_path": spec.row["source_rel_path"], "error_type": type(exc).__name__, "error_message": str(exc)})
            print(f"[FAIL] [{index}/{len(specs)}] {spec.row['slide_id']}: {exc}")
    log = args.feat_dir.resolve() / "logs" / f"openslide_failures_shard_{args.shard_index + 1}_of_{args.num_shards}.csv"
    if failures:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["slide_id", "source_rel_path", "error_type", "error_message"])
            writer.writeheader(); writer.writerows(failures)
    print(f"Completed: {completed}; skipped: {skipped}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
