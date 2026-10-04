#!/usr/bin/env python3
"""Read-only comparison of source patch coordinates and an existing UNI2 bag.

Use before attempting augmentation when the baseline coordinate-order check
fails. This script does not modify the WSI, H5, PT or augmentation output.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from extract_features_dicom import read_coordinates, read_manifest


def describe_pair(name: str, source: np.ndarray | None, target: np.ndarray) -> None:
    if source is None:
        print(f"{name}: absent")
        return
    source = np.asarray(source)[:, :2]
    target = np.asarray(target)[:, :2]
    print(f"{name}: source_shape={source.shape}, baseline_shape={target.shape}")
    if source.shape != target.shape:
        return
    same_row = np.all(source == target, axis=1)
    print(f"{name}: same_order={bool(same_row.all())}, matching_rows={int(same_row.sum())}/{len(source)}")
    print(f"{name}: same_coordinate_multiset={Counter(map(tuple, source.tolist())) == Counter(map(tuple, target.tolist()))}")
    delta = source.astype(np.int64) - target.astype(np.int64)
    unique_delta, delta_counts = np.unique(delta, axis=0, return_counts=True)
    largest = np.argsort(delta_counts)[-3:][::-1]
    print(f"{name}: constant_translation={len(unique_delta) == 1}, distinct_translations={len(unique_delta)}")
    print(f"{name}: leading_translations={[(unique_delta[i].tolist(), int(delta_counts[i])) for i in largest]}")
    print(f"{name}: source_xy_range={source.min(axis=0).tolist()}..{source.max(axis=0).tolist()}")
    print(f"{name}: baseline_xy_range={target.min(axis=0).tolist()}..{target.max(axis=0).tolist()}")
    print(f"{name}: negative_baseline_rows={int(np.any(target < 0, axis=1).sum())}")
    for index in np.flatnonzero(~same_row)[:5]:
        print(f"{name}: first_difference[{int(index)}] source={source[index].tolist()} baseline={target[index].tolist()}")


def check_embeddings(spec: object, root: Path, baseline_dir: Path,
                     source_coords: np.ndarray, baseline_coords: np.ndarray,
                     patch_size: int, count: int, device_name: str) -> None:
    """Check which coordinate array recreates the baseline UNI2 embeddings."""
    import torch

    from extract_features_mrxs import MrxsPyramidReader
    from models import get_encoder
    from mrxs_pyramid import choose_mrxs

    if source_coords.shape != baseline_coords.shape:
        raise ValueError("Embedding comparison requires matching coordinate counts")
    indices = np.unique(np.linspace(0, len(source_coords) - 1,
                                    min(count, len(source_coords)), dtype=int))
    pt_path = baseline_dir / "pt_files" / f"{spec.slide_id}.pt"
    try:
        reference = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        reference = torch.load(pt_path, map_location="cpu")
    if not isinstance(reference, torch.Tensor) or reference.shape != (len(source_coords), 1536):
        raise ValueError(f"Invalid baseline tensor: {pt_path}")
    reference = reference[torch.from_numpy(indices)].float()
    device = torch.device(device_name)
    model, transform = get_encoder("uni_v2", target_img_size=224)
    model = model.eval().to(device)
    reader = MrxsPyramidReader(choose_mrxs(root / spec.slide_rel_path), 0)
    try:
        for name, coords in (("current_source_coords", source_coords),
                             ("baseline_h5_coords", baseline_coords)):
            predictions = []
            for start in range(0, len(indices), 4):
                batch_indices = indices[start:start + 4]
                images = [reader.read_patch(int(coords[i, 0]), int(coords[i, 1]),
                                            patch_size) for i in batch_indices]
                inputs = torch.stack([transform(image) for image in images]).to(device)
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                        predictions.append(model(inputs).detach().cpu().float())
            predicted = torch.cat(predictions)
            cosine = torch.nn.functional.cosine_similarity(predicted, reference, dim=1)
            error = (predicted - reference).abs()
            print(f"{name}_embedding_indices:", indices.tolist())
            print(f"{name}_embedding_mean_cosine:", float(cosine.mean()))
            print(f"{name}_embedding_min_cosine:", float(cosine.min()))
            print(f"{name}_embedding_mean_absolute_error:", float(error.mean()))
            print(f"{name}_embedding_cosines:", [round(float(value), 6) for value in cosine])
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--check-embeddings", type=int, default=0,
                        help="Re-encode this many spread-out patches under both coordinate arrays.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not 0 <= args.check_embeddings <= 32:
        parser.error("--check-embeddings must be 0..32")

    specs = read_manifest(args.feature_manifest, args.dataset_root)
    spec = next((item for item in specs if item.slide_id == args.slide_id), None)
    if spec is None:
        raise ValueError(f"Slide ID not found in manifest: {args.slide_id}")
    source = read_coordinates(spec.coords_path)
    baseline_h5 = args.baseline_dir / "h5_files" / f"{args.slide_id}.h5"
    print("slide_id:", args.slide_id)
    print("source_coordinate_file:", spec.coords_path)
    print("source_coordinate_level:", source.pyramid_level)
    print("source_patch_size:", source.patch_size)
    print("source_coordinate_count:", len(source.coords_level))
    print("baseline_feature_h5:", baseline_h5)
    if not baseline_h5.is_file():
        raise FileNotFoundError(baseline_h5)
    with h5py.File(baseline_h5, "r") as handle:
        print("baseline_h5_keys:", sorted(handle.keys()))
        for key in ("coords_path", "coords_source", "patch_level", "patch_size_level",
                    "source_patch_count", "encoded_patch_count", "patch_sampling",
                    "coordinate_storage_space", "source_mrxs_path"):
            if key in handle.attrs:
                print(f"baseline_{key}:", handle.attrs[key])
        if "coords" not in handle:
            raise ValueError("Baseline H5 lacks coords")
        baseline_coords = handle["coords"][:]
        print("baseline_feature_count:", handle["features"].shape[0] if "features" in handle else "absent")
        describe_pair("source_coords_vs_baseline_coords", source.coords_level, baseline_coords)
        describe_pair("source_level0_vs_baseline_coords", source.coords_level0, baseline_coords)
        if "coords_level" in handle:
            describe_pair("source_coords_vs_baseline_coords_level", source.coords_level,
                          handle["coords_level"][:])
    if args.check_embeddings:
        check_embeddings(spec, args.dataset_root, args.baseline_dir,
                         source.coords_level, baseline_coords, source.patch_size,
                         args.check_embeddings, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
