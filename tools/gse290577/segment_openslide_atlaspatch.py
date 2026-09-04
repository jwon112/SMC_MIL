#!/usr/bin/env python3
"""Run AtlasPatch tissue segmentation for OpenSlide WSI or manifest-defined ROIs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openslide_mpp import read_roi_thumbnail  # noqa: E402


def build_predictor(checkpoint: Path, config: Path, device: str):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model = instantiate(OmegaConf.load(str(config)).get("model"))
    predictor = SAM2ImagePredictor(model, mask_threshold=0.0)
    state = torch.load(checkpoint, map_location=torch_device)
    predictor.model.load_state_dict(state["model"], strict=True)
    predictor.model.to(torch_device).eval()
    return predictor


def segment_thumbnail(thumbnail: Image.Image, predictor) -> np.ndarray:
    import torch

    sam_image = thumbnail.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
    pixels = np.array(sam_image, dtype=np.uint8, copy=True)
    with torch.inference_mode():
        predictor.set_image(pixels)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([0, 0, 1024, 1024], dtype=np.float32),
            multimask_output=False,
            return_logits=False,
        )
    mask = Image.fromarray(masks[0].astype(np.uint8) * 255)
    return np.asarray(mask.resize(thumbnail.size, Image.Resampling.NEAREST)) > 0


def save_outputs(thumbnail: Image.Image, mask: np.ndarray, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    thumbnail.save(output / "thumbnail.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "tissue_mask.png")
    overlay = Image.new("RGBA", thumbnail.size, (255, 0, 0, 0))
    overlay.putalpha(Image.fromarray(mask.astype(np.uint8) * 110))
    Image.alpha_composite(thumbnail.convert("RGBA"), overlay).save(output / "tissue_overlay.png")


def int_or_zero(value: str | None) -> int:
    return int(float(value)) if value not in (None, "") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--processing-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-size", type=int, default=3072)
    parser.add_argument("--fallback-mpp", type=float)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    processing_root = args.processing_root.resolve()
    with args.inventory.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]
    queue = []
    for row in rows:
        output = processing_root / row["processing_rel_path"] / "atlaspatch"
        if not args.skip_existing or not (output / "tissue_mask.png").is_file():
            queue.append((row, output))
    print(f"AtlasPatch OpenSlide: discovered={len(rows)}, to_process={len(queue)}")
    if args.dry_run:
        for row, output in queue:
            print(f"[DRY RUN] {row['source_rel_path']} -> {output}")
        return 0

    predictor = build_predictor(args.checkpoint, args.config, args.device)
    failures: list[dict[str, str]] = []
    for index, (row, output) in enumerate(queue, 1):
        source = dataset_root / row["source_rel_path"]
        try:
            roi = (
                int_or_zero(row.get("roi_x")), int_or_zero(row.get("roi_y")),
                int_or_zero(row.get("roi_width")), int_or_zero(row.get("roi_height")),
            )
            thumbnail, _, _ = read_roi_thumbnail(source, roi, args.max_size, args.fallback_mpp)
            save_outputs(thumbnail, segment_thumbnail(thumbnail, predictor), output)
            print(f"[OK] [{index}/{len(queue)}] {row['slide_id']} -> {output}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"slide_id": row["slide_id"], "source_rel_path": row["source_rel_path"], "error": str(exc)})
            print(f"[FAIL] [{index}/{len(queue)}] {row['slide_id']}: {exc}")
    log = processing_root / "logs" / "atlaspatch_failures.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "source_rel_path", "error"])
        writer.writeheader(); writer.writerows(failures)
    print(f"Completed: {len(queue) - len(failures)}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
