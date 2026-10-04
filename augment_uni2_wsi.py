#!/usr/bin/env python3
"""Build augmented UNI2-h feature views from *existing* WSI patch coordinates.

Copy this file to the SMC_MIL repository root on the server. It imports that
repository's DICOM/MRXS readers and encoder. Original feature bags are read
only for alignment checks; they are never modified. Train with the separate
train_clam_feature_views.py runner after verifying a complete feature bank.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TRANSFORMS = (
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("dicom", "mrxs"), required=True)
    parser.add_argument("--source-dataset", choices=("exp3", "mrxs13"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True,
                        help="Existing UNI2-h scale directory with pt_files/ and, ideally, h5_files/.")
    parser.add_argument("--stain-csv", type=Path, required=True,
                        help="Local stain_aug_labels.csv transferred to the server.")
    parser.add_argument("--task-csv", type=Path, required=True,
                        help="Existing ACR CSV with slide_id, label and source_dataset.")
    parser.add_argument("--split-csv", type=Path, required=True,
                        help="One fold's splits_N.csv; only the train column is augmented.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stain", choices=("HE", "IHC"), default="HE")
    parser.add_argument("--label-source", choices=("manual_only", "manual_or_filename"),
                        default="manual_only")
    parser.add_argument("--policy", choices=("geometry", "geometry_he_scale_light"),
                        default="geometry")
    parser.add_argument("--num-views", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--bank-seed", type=int, default=20260917)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--smoke-patches", type=int, default=None,
                        help="First N coordinates only; writes under output-root/smoke.")
    parser.add_argument("--preview-count", type=int, default=0,
                        help="Save this many original/augmented patch pairs per view and slide.")
    parser.add_argument("--check-original-patches", type=int, default=0,
                        help="Re-encode N spread-out unmodified patches and compare with baseline PT features.")
    parser.add_argument("--allow-mrxs-baseline-h5-coord-mismatch", action="store_true",
                        help="Only for MRXS: permit stale baseline H5 coords after per-slide original PT re-encoding passes.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only for --dry-run or --smoke-patches.")
    parser.add_argument("--include-slide-ids", type=Path,
                        help="Optional newline-separated train slide IDs for complete-bag technical checks.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source == "dicom" and args.source_dataset != "exp3":
        parser.error("DICOM source requires --source-dataset exp3")
    if args.source == "mrxs" and args.source_dataset != "mrxs13":
        parser.error("MRXS source requires --source-dataset mrxs13")
    if args.policy == "geometry_he_scale_light" and args.stain != "HE":
        parser.error("H/E scaling is restricted to confirmed HE slides")
    if args.batch_size < 1 or (args.smoke_patches is not None and args.smoke_patches < 1):
        parser.error("Batch and smoke patch counts must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid --num-shards or --shard-index")
    if not 0 <= args.preview_count <= 32:
        parser.error("--preview-count must be between 0 and 32")
    if not 0 <= args.check_original_patches <= 32:
        parser.error("--check-original-patches must be between 0 and 32")
    if args.allow_mrxs_baseline_h5_coord_mismatch:
        if args.source != "mrxs" or args.check_original_patches < 8:
            parser.error("MRXS H5 coordinate mismatch requires --source mrxs and at least 8 original checks")
    if args.limit is not None and (args.limit < 1 or not (args.dry_run or args.smoke_patches)):
        parser.error("--limit requires --dry-run or --smoke-patches")
    if args.baseline_dir.resolve() == args.output_root.resolve():
        parser.error("Output root must differ from the existing baseline feature directory")
    return args


def rows_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "slide_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing slide_id column: {path}")
        result: dict[str, dict[str, str]] = {}
        for row in reader:
            slide_id = row["slide_id"].strip()
            if not slide_id or slide_id in result:
                raise ValueError(f"Blank or duplicate slide_id in {path}: {slide_id!r}")
            result[slide_id] = row
    return result


def split_ids(path: Path) -> tuple[set[str], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"train", "val"}.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV requires train and val columns")
        train = [row["train"].strip() for row in reader if row["train"].strip()]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        val = [row["val"].strip() for row in csv.DictReader(handle) if row["val"].strip()]
    if len(train) != len(set(train)) or len(val) != len(set(val)):
        raise ValueError("Duplicate slide ID within a split column")
    if set(train) & set(val):
        raise ValueError("Train and validation slide IDs overlap")
    return set(train), set(val)


def select_specs(specs: list[Any], stain_rows: dict[str, dict[str, str]],
                 task_rows: dict[str, dict[str, str]], train: set[str], val: set[str],
                 source_dataset: str, stain: str, label_source: str) -> list[Any]:
    selected = []
    for spec in specs:
        sid = spec.slide_id
        if sid not in train:
            continue
        if sid in val:
            raise ValueError(f"Validation leakage: {sid}")
        task = task_rows.get(sid)
        if task is None or task.get("source_dataset", "").strip() != source_dataset:
            continue
        label = stain_rows.get(sid)
        if label is None:
            continue
        if label_source == "manual_only":
            eligible = (label.get("manual_label", "").strip() == stain
                        and label.get("reference_source", "").strip() == "manual_review")
        else:
            eligible = (label.get("final_label", "").strip() == stain
                        and label.get("reference_source", "").strip()
                        in {"manual_review", "filename_rule"})
        if eligible:
            selected.append(spec)
    return selected


def stable_seed(bank_seed: int, slide_id: str, patch_index: int, view: int,
                kind: str) -> int:
    payload = f"{bank_seed}|{slide_id}|{patch_index}|{view}|{kind}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def geometry_code(bank_seed: int, slide_id: str, patch_index: int, view: int) -> int:
    if not 1 <= view <= len(TRANSFORMS):
        raise ValueError(f"Augmented view must be 1..{len(TRANSFORMS)}")
    # One deterministic permutation per patch guarantees distinct non-identity
    # D4 transforms across view_1..view_4 while matching between policies.
    rng = np.random.default_rng(stable_seed(bank_seed, slide_id, patch_index,
                                             0, "geometry_permutation"))
    return int(rng.permutation(len(TRANSFORMS))[view - 1])


def he_scale_rgb(image: Image.Image, h_scale: float, e_scale: float) -> Image.Image:
    """Apply an H/E perturbation delta to RGB, preserving scale=1 exactly.

    The third *mathematical HED component* is held fixed. On H&E this does not
    imply that a real IHC DAB signal was present or biologically preserved.
    """
    try:
        from skimage.color import hed2rgb, rgb2hed
    except ImportError as exc:
        raise RuntimeError("geometry_he_scale_light requires scikit-image") from exc
    original_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hed = rgb2hed(original_rgb)
    varied = hed.copy()
    varied[..., 0] *= h_scale
    varied[..., 1] *= e_scale
    # rgb2hed may clip negative stain concentrations: raw HED round-tripping
    # can shift color even with both scales at 1. Use only the RGB delta.
    delta = hed2rgb(varied) - hed2rgb(hed)
    rgb = np.clip(original_rgb + delta, 0.0, 1.0)
    if not np.isfinite(rgb).all():
        raise ValueError("H/E color reconstruction produced NaN or Inf")
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB")


def augment_patch(image: Image.Image, *, slide_id: str, patch_index: int,
                  view: int, bank_seed: int, policy: str,
                  he_scales: tuple[float, float] | None = None) -> Image.Image:
    # Geometry is deliberately identical for geometry and H/E-scale policies.
    result = image.convert("RGB").transpose(
        TRANSFORMS[geometry_code(bank_seed, slide_id, patch_index, view)]
    )
    if policy == "geometry_he_scale_light":
        rng = np.random.default_rng(stable_seed(bank_seed, slide_id, patch_index,
                                                 view, "hed"))
        # One pair of independent Uniform(0.95, 1.05) factors per patch/view.
        # This is an exploratory setting, not the published HED-light recipe.
        h_scale, e_scale = he_scales or (rng.uniform(0.95, 1.05),
                                        rng.uniform(0.95, 1.05))
        result = he_scale_rgb(result, h_scale, e_scale)
    elif policy != "geometry":
        raise ValueError(f"Unsupported augmentation policy: {policy}")
    return result


def load_baseline_shape(path: Path) -> tuple[int, int]:
    import torch

    try:
        features = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older trusted server PyTorch checkpoint format.
        features = torch.load(path, map_location="cpu")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"Baseline feature file is not a [patch,dim] tensor: {path}")
    if not torch.isfinite(features).all():
        raise ValueError(f"Feature bag contains NaN or Inf: {path}")
    return tuple(int(value) for value in features.shape)


def verify_baseline(spec: Any, coordinates: np.ndarray, baseline_dir: Path,
                    *, allow_mrxs_h5_mismatch: bool = False) -> tuple[str, bool]:
    import h5py

    pt_path = baseline_dir / "pt_files" / f"{spec.slide_id}.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"Baseline feature missing: {pt_path}")
    shape = load_baseline_shape(pt_path)
    if shape != (len(coordinates), 1536):
        raise ValueError(f"Baseline {spec.slide_id} shape {shape} != {(len(coordinates), 1536)}")
    h5_path = baseline_dir / "h5_files" / f"{spec.slide_id}.h5"
    h5_mismatch = False
    if h5_path.is_file():
        with h5py.File(h5_path, "r") as handle:
            if "coords" not in handle:
                raise ValueError(f"Baseline H5 has no coords: {h5_path}")
            h5_mismatch = not np.array_equal(np.asarray(handle["coords"]), coordinates)
            if h5_mismatch and not allow_mrxs_h5_mismatch:
                raise ValueError(f"Baseline coordinate order differs: {spec.slide_id}")
    return hashlib.sha256(coordinates.tobytes()).hexdigest(), h5_mismatch


def output_paths(root: Path, policy: str, view: int,
                 slide_id: str) -> tuple[Path, Path]:
    base = root / policy / f"view_{view}"
    return (base / "pt_files" / f"{slide_id}.pt",
            base / "metadata" / f"{slide_id}.json")


def encode_one(spec: Any, args: argparse.Namespace, model: Any, transform: Any,
               device: Any, root: Path) -> str:
    import torch
    from extract_features_dicom import DicomPyramidReader, read_coordinates

    coordinate_set = read_coordinates(spec.coords_path)
    if coordinate_set.pyramid_level != 0:
        raise ValueError("Initial augmentation script supports only L0/40x coordinates")
    full_coordinates = coordinate_set.coords_level
    if coordinate_set.coords_level0 is not None and not np.array_equal(
        full_coordinates, coordinate_set.coords_level0
    ):
        raise ValueError(f"L0 and stored level-0 coordinates differ: {spec.slide_id}")
    if len(full_coordinates) == 0:
        raise ValueError(f"No coordinates: {spec.slide_id}")
    coordinate_hash, h5_mismatch = verify_baseline(
        spec, full_coordinates, args.baseline_dir,
        allow_mrxs_h5_mismatch=(args.source == "mrxs"
                                and args.allow_mrxs_baseline_h5_coord_mismatch),
    )
    coordinates = full_coordinates
    if args.smoke_patches is not None:
        coordinates = coordinates[:args.smoke_patches]
    views = list(range(1, args.num_views + 1))
    pending: list[tuple[int, Path, Path]] = []
    for view in views:
        pt_path, meta_path = output_paths(root, args.policy, view, spec.slide_id)
        if pt_path.exists() or meta_path.exists():
            if not args.overwrite:
                if not (pt_path.is_file() and meta_path.is_file()):
                    raise RuntimeError(f"Incomplete output pair for {spec.slide_id}, view {view}")
                saved = json.loads(meta_path.read_text(encoding="utf-8"))
                if (saved.get("coordinate_sha256") != coordinate_hash
                        or saved.get("bank_seed") != args.bank_seed
                        or saved.get("patch_count") != len(coordinates)
                        or saved.get("policy") != args.policy
                        or saved.get("source_dataset") != args.source_dataset
                        or saved.get("stain") != args.stain
                        or saved.get("view") != view
                        or bool(saved.get("baseline_h5_coords_mismatch", False)) != h5_mismatch
                        or saved.get("smoke_only") != (args.smoke_patches is not None)):
                    raise RuntimeError(f"Existing output metadata differs: {meta_path}")
                required_samples = min(args.check_original_patches, len(full_coordinates))
                if required_samples and not (
                    saved.get("original_reencoding_samples", 0) >= required_samples
                    and saved.get("original_reencoding_min_cosine", -1) >= 0.999
                    and saved.get("original_reencoding_mean_absolute_error", 1) <= 0.01
                ):
                    raise RuntimeError(f"Existing view lacks required original re-encoding QA: {meta_path}")
                if load_baseline_shape(pt_path) != (len(coordinates), 1536):
                    raise RuntimeError(f"Existing augmented feature shape differs: {pt_path}")
                continue
        pending.append((view, pt_path, meta_path))
    if not pending:
        return "skipped"

    reader: Any
    if args.source == "dicom":
        reader = DicomPyramidReader(args.dataset_root / spec.slide_rel_path, 0)
        read_patch = lambda x, y: reader.read_patch(x, y,
                                                    coordinate_set.patch_size,
                                                    coordinate_set.patch_size)
    else:
        from extract_features_mrxs import MrxsPyramidReader
        from mrxs_pyramid import choose_mrxs

        reader = MrxsPyramidReader(
            choose_mrxs(args.dataset_root / spec.slide_rel_path), 0
        )
        read_patch = lambda x, y: reader.read_patch(x, y, coordinate_set.patch_size)

    original_check: dict[str, Any] | None = None
    try:
        if args.check_original_patches:
            check_indices = np.unique(np.linspace(
                0, len(full_coordinates) - 1,
                min(args.check_original_patches, len(full_coordinates)), dtype=int,
            ))
            reencoded_batches = []
            for start in range(0, len(check_indices), args.batch_size):
                batch_indices = check_indices[start:start + args.batch_size]
                images = [read_patch(int(full_coordinates[index, 0]),
                                     int(full_coordinates[index, 1]))
                          for index in batch_indices]
                inputs = torch.stack([transform(image.convert("RGB")) for image in images]).to(
                    device, non_blocking=device.type == "cuda")
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
                        reencoded_batches.append(model(inputs).detach().cpu().float())
            reencoded = torch.cat(reencoded_batches)
            baseline_path = args.baseline_dir / "pt_files" / f"{spec.slide_id}.pt"
            try:
                baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
            except TypeError:
                baseline = torch.load(baseline_path, map_location="cpu")
            reference = baseline[torch.from_numpy(check_indices)].float()
            if reencoded.shape != reference.shape or not torch.isfinite(reencoded).all():
                raise ValueError(f"Original re-encoding failed for {spec.slide_id}")
            difference = (reencoded - reference).abs()
            cosine = torch.nn.functional.cosine_similarity(reencoded, reference, dim=1)
            original_check = {
                "slide_id": spec.slide_id,
                "patches": len(check_indices),
                "indices": check_indices.tolist(),
                "mean_absolute_error": float(difference.mean()),
                "max_absolute_error": float(difference.max()),
                "min_cosine_similarity": float(cosine.min()),
                "mean_cosine_similarity": float(cosine.mean()),
                "baseline_h5_coords_mismatch": h5_mismatch,
                "note": "Min cosine >= 0.999 and mean absolute error <= 0.01 are required QA gates when original checks are requested.",
            }
            quality_path = root / "quality" / "original_reencoding" / f"{spec.slide_id}.json"
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            quality_path.write_text(json.dumps(original_check, indent=2) + "\n", encoding="utf-8")
            print("Original re-encoding:", json.dumps(original_check), flush=True)
            if (float(cosine.min()) < 0.999 or float(difference.mean()) > 0.01):
                raise ValueError(f"Current coordinates do not reproduce baseline PT: {spec.slide_id}")
        elif h5_mismatch:
            raise ValueError(f"MRXS H5 mismatch requires original re-encoding: {spec.slide_id}")
    except Exception:
        if args.source == "mrxs":
            reader.close()
        raise

    buffers: dict[int, np.memmap] = {}
    temporary: list[Path] = []
    try:
        for view, pt_path, _ in pending:
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            temp_npy = pt_path.with_suffix(".npy.tmp")
            if temp_npy.exists():
                temp_npy.unlink()
            buffers[view] = np.lib.format.open_memmap(
                temp_npy, mode="w+", dtype=np.float32, shape=(len(coordinates), 1536)
            )
            temporary.append(temp_npy)
        for start in range(0, len(coordinates), args.batch_size):
            stop = min(start + args.batch_size, len(coordinates))
            images = [read_patch(int(x), int(y)) for x, y in coordinates[start:stop]]
            for view, _, _ in pending:
                augmented = [
                    augment_patch(image, slide_id=spec.slide_id,
                                  patch_index=index, view=view,
                                  bank_seed=args.bank_seed, policy=args.policy)
                    for index, image in enumerate(images, start=start)
                ]
                if start < args.preview_count:
                    preview_dir = root / "previews" / args.stain / args.policy / f"view_{view}"
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    for local_index in range(min(len(images), args.preview_count - start)):
                        original = images[local_index].convert("RGB")
                        changed = augmented[local_index]
                        comparison = Image.new(
                            "RGB", (original.width + changed.width,
                                    max(original.height, changed.height)), "white"
                        )
                        comparison.paste(original, (0, 0))
                        comparison.paste(changed, (original.width, 0))
                        comparison.save(preview_dir / f"{spec.slide_id}__patch_{start + local_index:05d}.png")
                inputs = torch.stack([transform(image) for image in augmented]).to(
                    device, non_blocking=device.type == "cuda"
                )
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", enabled=args.amp
                                        and device.type == "cuda"):
                        encoded = model(inputs)
                if not isinstance(encoded, torch.Tensor) or encoded.shape != (
                    len(images), 1536
                ):
                    raise ValueError(f"Unexpected UNI2-h feature shape: {getattr(encoded, 'shape', None)}")
                if not torch.isfinite(encoded).all():
                    raise ValueError(f"UNI2-h produced NaN or Inf for {spec.slide_id}, view {view}")
                buffers[view][start:stop] = encoded.detach().cpu().float().numpy()
        for view, pt_path, meta_path in pending:
            pt_tmp = pt_path.with_suffix(".pt.tmp")
            meta_tmp = meta_path.with_suffix(".json.tmp")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.extend((pt_tmp, meta_tmp))
            buffers[view].flush()
            torch.save(torch.from_numpy(buffers[view]), pt_tmp)
            metadata = {
                "slide_id": spec.slide_id,
                "stain": args.stain,
                "label_source": args.label_source,
                "source_dataset": args.source_dataset,
                "source_wsi": str(args.dataset_root / spec.slide_rel_path),
                "coordinate_h5": str(spec.coords_path),
                "coordinate_sha256": coordinate_hash,
                "baseline_h5_coords_mismatch": h5_mismatch,
                "original_reencoding_samples": original_check["patches"] if original_check else 0,
                "original_reencoding_min_cosine": (
                    original_check["min_cosine_similarity"] if original_check else None
                ),
                "original_reencoding_mean_absolute_error": (
                    original_check["mean_absolute_error"] if original_check else None
                ),
                "patch_count": len(coordinates),
                "patch_size": coordinate_set.patch_size,
                "pyramid_level": 0,
                "feature_dim": 1536,
                "encoder": "uni_v2",
                "torch_version": torch.__version__,
                "policy": args.policy,
                "view": view,
                "bank_seed": args.bank_seed,
                "he_scale_range": [0.95, 1.05] if args.policy == "geometry_he_scale_light" else None,
                "he_scale_sampling": "independent_uniform_per_patch_view" if args.policy == "geometry_he_scale_light" else None,
                "he_implementation": (
                    "scikit-image rgb2hed/hed2rgb; apply varied-minus-unvaried RGB delta"
                    if args.policy == "geometry_he_scale_light" else None
                ),
                "skimage_version": (
                    __import__("skimage").__version__
                    if args.policy == "geometry_he_scale_light" else None
                ),
                "smoke_only": args.smoke_patches is not None,
            }
            meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(pt_tmp, pt_path)
            os.replace(meta_tmp, meta_path)
    finally:
        for buffer in buffers.values():
            del buffer
        for path in temporary:
            if path.exists():
                path.unlink()
        if args.source == "mrxs":
            reader.close()
    return "completed"


def main() -> int:
    args = arguments()
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu for a tiny smoke test")
    from extract_features_dicom import read_manifest, read_coordinates

    specs = read_manifest(args.feature_manifest, args.dataset_root)
    stain_rows = rows_by_id(args.stain_csv)
    task_rows = rows_by_id(args.task_csv)
    train, val = split_ids(args.split_csv)
    if not stain_rows or not task_rows:
        raise ValueError("Stain and task CSVs must be nonempty")
    if (train | val) - task_rows.keys():
        raise ValueError("Split CSV contains slide IDs absent from task CSV")
    if not {"manual_label", "final_label", "reference_source"}.issubset(
        next(iter(stain_rows.values())).keys()
    ):
        raise ValueError("--stain-csv must include manual_label, final_label and reference_source")
    if not {"label", "source_dataset"}.issubset(next(iter(task_rows.values())).keys()):
        raise ValueError("Task CSV requires label and source_dataset columns")
    if "case_id" in next(iter(task_rows.values())):
        train_patients = {task_rows[sid]["case_id"] for sid in train}
        val_patients = {task_rows[sid]["case_id"] for sid in val}
        if train_patients & val_patients:
            raise ValueError("Train and validation patients overlap")
    selected = select_specs(specs, stain_rows, task_rows, train, val,
                            args.source_dataset, args.stain, args.label_source)
    if args.include_slide_ids is not None:
        include_ids = {sid.strip() for sid in args.include_slide_ids.read_text(
            encoding="utf-8").splitlines() if sid.strip()}
        if not include_ids or not include_ids <= train:
            raise ValueError("--include-slide-ids must contain only IDs from this fold's train split")
        selected = [spec for spec in selected if spec.slide_id in include_ids]
    selected = selected[args.shard_index::args.num_shards]
    if args.limit is not None:
        selected = selected[:args.limit]
    print(f"Selected {len(selected)} {args.stain} train slides from {args.source_dataset}; "
          f"policy={args.policy}, views={args.num_views}, fold={args.split_csv.name}, "
          f"shard={args.shard_index + 1}/{args.num_shards}")
    if not selected:
        raise ValueError("No eligible slides; inspect label source, source dataset and split")
    print("First slide IDs:", [spec.slide_id for spec in selected[:5]])
    if args.dry_run:
        for spec in selected[:10]:
            coord = read_coordinates(spec.coords_path)
            baseline = args.baseline_dir / "pt_files" / f"{spec.slide_id}.pt"
            print(spec.slide_id, "patches=", len(coord.coords_level),
                  "level=", coord.pyramid_level, "baseline=", baseline.is_file())
        return 0

    import torch
    from models import get_encoder

    root = args.output_root / "smoke" if args.smoke_patches is not None else args.output_root
    device = torch.device(args.device)
    model, transform = get_encoder("uni_v2", target_img_size=224)
    model = model.eval().to(device)
    completed = skipped = 0
    for index, spec in enumerate(selected, 1):
        started = time.perf_counter()
        result = encode_one(spec, args, model, transform, device, root)
        completed += result == "completed"
        skipped += result == "skipped"
        print(f"[{index}/{len(selected)}] {spec.slide_id}: {result} "
              f"in {time.perf_counter() - started:.1f} s", flush=True)
    print(f"Done: completed={completed}, skipped={skipped}, root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
