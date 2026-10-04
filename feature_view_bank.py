"""CLAM-compatible whole-bag view selection for pre-extracted UNI2 features.

Only the training split is wrapped. Validation/test keep Generic_Split and read
the original feature directory. The view RNG is private so it cannot advance
PyTorch's sampler/model RNG or Python's global RNG.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def coordinate_hashes(coords: Any) -> set[str]:
    """Hashes for the same integer XY rows under supported storage widths."""
    import numpy as np

    array = np.asarray(coords)
    if (array.ndim != 2 or array.shape[1] != 2
            or not np.issubdtype(array.dtype, np.integer)):
        raise ValueError("Coordinate array must be [patch,2] integer XY rows")
    return {hashlib.sha256(array.astype(dtype).tobytes()).hexdigest()
            for dtype in (np.int32, np.int64)}


class ViewSelector:
    def __init__(self, total_views: int, seed: int, original_probability: float = 0.5):
        if total_views not in (1, 2, 3, 5):
            raise ValueError("total_views must be 1, 2, 3, or 5")
        if not 0.0 <= original_probability <= 1.0:
            raise ValueError("original_probability must lie in [0, 1]")
        self.total_views = total_views
        self.original_probability = 1.0 if total_views == 1 else original_probability
        self.rng = random.Random(seed)

    def draw(self) -> int:
        if self.total_views == 1 or self.rng.random() < self.original_probability:
            return 0
        return 1 + self.rng.randrange(self.total_views - 1)


def feature_path(baseline_dir: Path, bank_root: Path | None,
                 policy: str, slide_id: str, view: int) -> Path:
    if view == 0:
        return baseline_dir / "pt_files" / f"{slide_id}.pt"
    if bank_root is None:
        raise ValueError("Augmented view requires bank_root")
    return bank_root / policy / f"view_{view}" / "pt_files" / f"{slide_id}.pt"


def validate_feature_bank(slide_rows: Any, *, baseline_dir: Path,
                          bank_root: Path | None, policy: str,
                          total_views: int) -> dict[str, int]:
    """Fail before training if any slide/view is missing or is a partial bag."""
    if total_views not in (1, 2, 3, 5):
        raise ValueError("total_views must be 1, 2, 3, or 5")
    if total_views > 1 and (bank_root is None or policy not in
                            ("geometry", "geometry_he_scale_light")):
        raise ValueError("Augmented conditions require a supported policy and bank root")
    if total_views == 1 and policy != "original":
        raise ValueError("1-view baseline must use policy='original'")
    checked = 0
    for row in slide_rows:
        sid = str(row["slide_id"])
        source = str(row.get("source_dataset", ""))
        stain = str(row.get("stain_group", ""))
        original = feature_path(baseline_dir, bank_root, policy, sid, 0)
        if not original.is_file():
            raise FileNotFoundError(f"Original feature missing: {original}")
        coordinate_sha: str | None = None
        patch_count: int | None = None
        bank_seed: int | None = None
        patch_size: int | None = None
        mrxs_h5_mismatch: bool | None = None
        source_coordinate_h5: Path | None = None
        for view in range(1, total_views):
            pt_path = feature_path(baseline_dir, bank_root, policy, sid, view)
            meta_path = (bank_root / policy / f"view_{view}" / "metadata" /
                         f"{sid}.json")
            if not pt_path.is_file() or not meta_path.is_file():
                raise FileNotFoundError(f"Complete view pair missing: {pt_path} / {meta_path}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta.get("slide_id") != sid or meta.get("view") != view
                    or meta.get("policy") != policy or meta.get("smoke_only") is not False
                    or meta.get("feature_dim") != 1536 or meta.get("encoder") != "uni_v2"
                    or meta.get("pyramid_level") != 0):
                raise ValueError(f"Wrong or partial feature metadata: {meta_path}")
            if source and meta.get("source_dataset") != source:
                raise ValueError(f"Source dataset differs: {meta_path}")
            if stain and meta.get("stain") != stain:
                raise ValueError(f"Stain family differs: {meta_path}")
            current_sha = meta.get("coordinate_sha256")
            current_count = meta.get("patch_count")
            current_seed = meta.get("bank_seed")
            current_size = meta.get("patch_size")
            current_mismatch = meta.get("baseline_h5_coords_mismatch", False)
            if not isinstance(current_sha, str) or not current_sha:
                raise ValueError(f"Coordinate hash absent: {meta_path}")
            if not isinstance(current_count, int) or current_count < 1:
                raise ValueError(f"Patch count invalid: {meta_path}")
            if not isinstance(current_seed, int) or not isinstance(current_size, int):
                raise ValueError(f"Bank seed or patch size invalid: {meta_path}")
            if not isinstance(current_mismatch, bool):
                raise ValueError(f"Baseline H5 mismatch flag invalid: {meta_path}")
            current_source_h5: Path | None = None
            if current_mismatch:
                samples = meta.get("original_reencoding_samples")
                min_cosine = meta.get("original_reencoding_min_cosine")
                mean_error = meta.get("original_reencoding_mean_absolute_error")
                source_h5 = meta.get("coordinate_h5")
                if (meta.get("source_dataset") != "mrxs13" or source != "mrxs13"
                        or not isinstance(samples, int) or samples < 8
                        or not isinstance(min_cosine, (int, float))
                        or not math.isfinite(min_cosine) or min_cosine < 0.999
                        or not isinstance(mean_error, (int, float))
                        or not math.isfinite(mean_error) or mean_error > 0.01
                        or not isinstance(source_h5, str) or not source_h5):
                    raise ValueError(f"MRXS coordinate mismatch lacks passing re-encoding QA: {meta_path}")
                current_source_h5 = Path(source_h5)
            if coordinate_sha is not None and current_sha != coordinate_sha:
                raise ValueError(f"Views do not share coordinates: {sid}")
            if patch_count is not None and current_count != patch_count:
                raise ValueError(f"Views do not share patch count: {sid}")
            if bank_seed is not None and current_seed != bank_seed:
                raise ValueError(f"Views do not share bank seed: {sid}")
            if patch_size is not None and current_size != patch_size:
                raise ValueError(f"Views do not share patch size: {sid}")
            if mrxs_h5_mismatch is not None and current_mismatch != mrxs_h5_mismatch:
                raise ValueError(f"Views disagree on baseline H5 coordinate status: {sid}")
            # QA values can vary slightly across GPU/batch executions. Each view
            # must independently pass the absolute gates above; floating-point
            # equality of separate re-encoding measurements is not required.
            if source_coordinate_h5 is not None and current_source_h5 != source_coordinate_h5:
                raise ValueError(f"Views disagree on MRXS source coordinates: {sid}")
            coordinate_sha, patch_count = current_sha, current_count
            bank_seed, patch_size = current_seed, current_size
            mrxs_h5_mismatch = current_mismatch
            source_coordinate_h5 = current_source_h5
            checked += 1
        baseline_h5 = baseline_dir / "h5_files" / f"{sid}.h5"
        if coordinate_sha is not None and baseline_h5.is_file():
            import h5py

            with h5py.File(baseline_h5, "r") as handle:
                if "coords" not in handle:
                    raise ValueError(f"Baseline H5 lacks coords: {baseline_h5}")
                coords = handle["coords"][:]
            baseline_matches = (len(coords) == patch_count
                                and coordinate_sha in coordinate_hashes(coords))
            if baseline_matches:
                if mrxs_h5_mismatch:
                    raise ValueError(f"MRXS mismatch metadata contradicts baseline H5: {sid}")
            elif not mrxs_h5_mismatch or source_coordinate_h5 is None:
                raise ValueError(f"Augmented bank does not match baseline coordinates: {sid}")
            else:
                if not source_coordinate_h5.is_file():
                    raise FileNotFoundError(f"MRXS source coordinates missing: {source_coordinate_h5}")
                with h5py.File(source_coordinate_h5, "r") as handle:
                    if "coords" not in handle:
                        raise ValueError(f"MRXS source H5 lacks coords: {source_coordinate_h5}")
                    source_coords = handle["coords"][:]
                if source_coords.ndim != 2 or source_coords.shape[1] < 2 or len(source_coords) != patch_count:
                    raise ValueError(f"MRXS source coordinate shape differs: {sid}")
                source_xy = source_coords[:, :2]
                # read_coordinates may normalize integer width before hashing.
                source_hashes = coordinate_hashes(source_xy)
                if coordinate_sha not in source_hashes:
                    raise ValueError(f"MRXS source coordinates differ from validated feature view: {sid}")
        elif mrxs_h5_mismatch:
            raise FileNotFoundError(f"Baseline H5 required for MRXS mismatch QA: {baseline_h5}")
    return {"slides": len(slide_rows), "augmented_view_files": checked}


def torch_feature_loader(path: Path) -> Any:
    import torch

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions without weights_only.
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"Feature bag is not a 2-D tensor: {path}")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"Feature bag is not floating-point: {path}")
    if tensor.shape[0] < 1 or tensor.shape[1] != 1536 or not torch.isfinite(tensor).all():
        raise ValueError(f"Feature bag is empty, non-1536-D, or non-finite: {path}")
    return tensor.float()


class FeatureViewTrainSplit:
    """Proxy Generic_Split; each __getitem__ call selects one complete view."""

    def __init__(self, base_split: Any, *, baseline_dir: Path,
                 bank_root: Path | None, policy: str, total_views: int,
                 seed: int, log_path: Path,
                 feature_loader: Callable[[Path], Any] = torch_feature_loader):
        self.base = base_split
        self.baseline_dir = baseline_dir
        self.bank_root = bank_root
        self.policy = policy
        self.total_views = total_views
        self.selector = ViewSelector(total_views, seed=seed)
        self.feature_loader = feature_loader
        self.slide_data = base_split.slide_data
        self.slide_cls_ids = base_split.slide_cls_ids
        self.selected_counts: Counter[int] = Counter()
        self.draw_count = 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("w", newline="", encoding="utf-8")
        self._log_writer = csv.writer(self._log_handle)
        self._log_writer.writerow(("draw_index", "slide_id", "label", "view", "policy"))

    def __len__(self) -> int:
        return len(self.base)

    def getlabel(self, index: int) -> Any:
        return self.base.getlabel(index)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        row = self.slide_data.iloc[index]
        sid = str(row["slide_id"])
        view = self.selector.draw()
        path = feature_path(self.baseline_dir, self.bank_root, self.policy, sid, view)
        tensor = self.feature_loader(path)
        label = row["label"]
        self._log_writer.writerow((self.draw_count, sid, label, view, self.policy))
        self._log_handle.flush()
        self.draw_count += 1
        self.selected_counts[view] += 1
        return tensor, label

    def close(self) -> None:
        if not self._log_handle.closed:
            self._log_handle.close()
