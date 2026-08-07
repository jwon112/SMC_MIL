#!/usr/bin/env python3
"""Generate AtlasPatch tissue masks and level-0 patch coordinates from MRXS WSI."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np
import openslide
from PIL import Image


SCRIPT_VERSION = "2026-08-07-v1"


def build_predictor(checkpoint: Path, config: Path, device: str):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model_cfg = OmegaConf.load(str(config)).get("model")
    model = instantiate(model_cfg)
    predictor = SAM2ImagePredictor(model, mask_threshold=0.0)
    state = torch.load(checkpoint, map_location=torch_device)
    predictor.model.load_state_dict(state["model"], strict=True)
    predictor.model.to(torch_device).eval()
    return predictor


def segment_thumbnail(thumbnail: Image.Image, predictor) -> np.ndarray:
    import torch

    original_size = thumbnail.size
    sam_image = thumbnail.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    with torch.inference_mode():
        predictor.set_image(np.asarray(sam_image, dtype=np.uint8))
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([0, 0, 1024, 1024], dtype=np.float32),
            multimask_output=False,
            return_logits=False,
        )
    mask_1024 = Image.fromarray(masks[0].astype(np.uint8) * 255)
    return np.asarray(mask_1024.resize(original_size, Image.Resampling.NEAREST)) > 0


def save_mask_outputs(mask: np.ndarray, thumbnail: Image.Image, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "tissue_mask.png")
    red = Image.new("RGBA", thumbnail.size, (255, 0, 0, 0))
    red.putalpha(Image.fromarray(mask.astype(np.uint8) * 110))
    Image.alpha_composite(thumbnail.convert("RGBA"), red).save(output_dir / "tissue_overlay.png")


def write_patch_coords(
    mrxs_path: Path,
    mask: np.ndarray,
    output_path: Path,
    patch_size: int,
    step_size: int,
    coord_mode: str,
    tissue_ratio_threshold: float,
) -> None:
    with openslide.OpenSlide(str(mrxs_path)) as slide:
        width, height = slide.dimensions
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, "nan"))
        mpp_y = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, "nan"))
    if not math.isfinite(mpp_x) or not math.isfinite(mpp_y):
        raise ValueError(f"No usable MPP metadata: {mrxs_path}")

    mask_h, mask_w = mask.shape
    half = patch_size // 2
    coords: list[tuple[int, int, int, int, int]] = []
    for y in range(0, max(height - patch_size + 1, 1), step_size):
        for x in range(0, max(width - patch_size + 1, 1), step_size):
            if coord_mode == "center":
                mx = min(mask_w - 1, max(0, round((x + half) * mask_w / width)))
                my = min(mask_h - 1, max(0, round((y + half) * mask_h / height)))
                keep = bool(mask[my, mx])
            else:
                mx0 = min(mask_w, max(0, math.floor(x * mask_w / width)))
                mx1 = min(mask_w, max(mx0 + 1, math.ceil((x + patch_size) * mask_w / width)))
                my0 = min(mask_h, max(0, math.floor(y * mask_h / height)))
                my1 = min(mask_h, max(my0 + 1, math.ceil((y + patch_size) * mask_h / height)))
                keep = float(mask[my0:my1, mx0:mx1].mean()) >= tissue_ratio_threshold
            if keep:
                coords.append((x, y, patch_size, patch_size, 0))

    values = np.asarray(coords, dtype=np.int32).reshape((-1, 5))
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("coords", data=values, compression="gzip")
        h5.attrs["num_patches"] = int(values.shape[0])
        h5.attrs["patch_size"] = patch_size
        h5.attrs["patch_size_level0"] = patch_size
        h5.attrs["target_magnification"] = 20
        h5.attrs["source_wsi"] = str(mrxs_path)
        h5.attrs["coordinate_source"] = "AtlasPatch SAM2 tissue segmentation on MRXS thumbnail"
        h5.attrs["coord_mode"] = coord_mode
        h5.attrs["tissue_ratio_threshold"] = tissue_ratio_threshold
        h5.attrs["mpp_x_um"] = mpp_x
        h5.attrs["mpp_y_um"] = mpp_y
    print(f"[OK] coords -> {output_path} (patches={values.shape[0]})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir-name", default="atlaspatch")
    parser.add_argument("--max-size", type=int, default=3072)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--coord-mode", choices=["center", "ratio"], default="center")
    parser.add_argument("--tissue-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--failure-log", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset_root.resolve()
    slides = sorted(root.rglob("*.mrxs"))
    if args.limit is not None:
        slides = slides[: args.limit]
    to_process = [path for path in slides if not (args.skip_existing and (path.parent / args.output_dir_name / "patch_coords.h5").is_file())]
    print(f"MRXS AtlasPatch {SCRIPT_VERSION}: discovered={len(slides)}, to_process={len(to_process)}")
    if args.dry_run:
        for path in to_process:
            print(f"[DRY RUN] {path} -> {path.parent / args.output_dir_name}")
        return 0

    predictor = build_predictor(args.checkpoint, args.config, args.device)
    failures: list[dict[str, str]] = []
    try:
        for index, path in enumerate(to_process, start=1):
            output = path.parent / args.output_dir_name
            try:
                print(f"[{index}/{len(to_process)}] {path}")
                with openslide.OpenSlide(str(path)) as slide:
                    thumbnail = slide.get_thumbnail((args.max_size, args.max_size)).convert("RGB")
                output.mkdir(parents=True, exist_ok=True)
                thumbnail.save(output / "thumbnail.png")
                mask = segment_thumbnail(thumbnail, predictor)
                save_mask_outputs(mask, thumbnail, output)
                write_patch_coords(path, mask, output / "patch_coords.h5", args.patch_size, args.step_size, args.coord_mode, args.tissue_ratio_threshold)
            except Exception as exc:  # noqa: BLE001
                failures.append({"mrxs_path": str(path), "error_type": type(exc).__name__, "error_message": str(exc)})
                print(f"[FAIL] {path}: {exc}")
    finally:
        predictor.model.cpu()
    if args.failure_log is not None:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        with args.failure_log.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["mrxs_path", "error_type", "error_message"])
            writer.writeheader()
            writer.writerows(failures)
    print(f"Completed: {len(to_process) - len(failures)}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
