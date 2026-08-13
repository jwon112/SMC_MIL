#!/usr/bin/env python3
"""Verify CLAM feature bags against an encoding manifest without re-encoding."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feat-dir", type=Path, required=True)
    parser.add_argument("--expected-embed-dim", type=int, default=1536)
    parser.add_argument("--expected-level", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"slide_id", "slide_rel_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain: {', '.join(sorted(required))}")

    failures: list[dict[str, str]] = []
    valid = 0
    for row in rows:
        slide_id = row["slide_id"].strip()
        h5_path = args.feat_dir / "h5_files" / f"{slide_id}.h5"
        pt_path = args.feat_dir / "pt_files" / f"{slide_id}.pt"
        try:
            if not h5_path.is_file() or not pt_path.is_file():
                raise FileNotFoundError(f"missing h5={h5_path.is_file()} pt={pt_path.is_file()}")
            with h5py.File(h5_path, "r") as h5:
                if "features" not in h5 or "coords" not in h5:
                    raise ValueError("missing features or coords dataset")
                features = h5["features"]
                coords = h5["coords"]
                if features.ndim != 2 or features.shape[1] != args.expected_embed_dim:
                    raise ValueError(f"feature shape={features.shape}, expected (*, {args.expected_embed_dim})")
                if len(features) < 1 or len(coords) != len(features):
                    raise ValueError(f"feature/coords rows={len(features)}/{len(coords)}")
                if h5.attrs.get("slide_id", "") != slide_id:
                    raise ValueError(f"slide_id attr={h5.attrs.get('slide_id')!r}")
                if h5.attrs.get("slide_rel_path", "") != row["slide_rel_path"]:
                    raise ValueError("slide_rel_path attr differs from manifest")
                if args.expected_level is not None and int(h5.attrs.get("patch_level", -1)) != args.expected_level:
                    raise ValueError(f"patch_level={h5.attrs.get('patch_level')}, expected={args.expected_level}")
                feature_count = len(features)
            tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (feature_count, args.expected_embed_dim):
                raise ValueError(f"pt shape={getattr(tensor, 'shape', None)}, expected=({feature_count}, {args.expected_embed_dim})")
            valid += 1
            print(f"[OK] {slide_id}: {feature_count} x {args.expected_embed_dim}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"slide_id": slide_id, "error": str(exc)})
            print(f"[FAIL] {slide_id}: {exc}")

    report = args.feat_dir / "logs" / "feature_bag_verification.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", "error"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"Verified: {valid}/{len(rows)}; failures: {len(failures)}")
    print(f"Report: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
