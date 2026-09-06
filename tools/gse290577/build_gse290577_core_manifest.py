#!/usr/bin/env python3
"""Map Xenium sample bounds into the two source OME-TIFF images."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import openslide


def clean_grade(value: str) -> str:
    value = value.strip().replace(" ", "")
    return f"{value}R" if re.fullmatch(r"[0-3]", value) else value


def binary_labels(acr: str, amr: str) -> dict[str, str | int]:
    acr_match = re.fullmatch(r"([0-3])R?", acr, re.IGNORECASE)
    acr_number = int(acr_match.group(1)) if acr_match else None
    amr_match = re.search(r"p?AMR\s*([0-3])", amr, re.IGNORECASE)
    amr_number = int(amr_match.group(1)) if amr_match else None
    return {
        "acr_any_label": "" if acr_number is None else int(acr_number >= 1),
        "acr_high_label": "" if acr_number is None else int(acr_number >= 2),
        "amr_positive_label": "" if amr_number is None else int(amr_number >= 1),
        "any_rejection_label": "" if acr_number is None or amr_number is None else int(acr_number >= 1 or amr_number >= 1),
        "significant_rejection_label": "" if acr_number is None or amr_number is None else int(acr_number >= 2 or amr_number >= 1),
    }


def find_one(root: Path, pattern: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one match for {pattern}, found {len(paths)}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rds", type=Path)
    parser.add_argument("--bounds-csv", type=Path, help="Reuse bounds already exported by export_core_bounds.R.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--margin-um", type=float, default=75.0, help="Physical margin around cell-coordinate bounds.")
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundled_bounds = Path(__file__).with_name("resources") / "gse290577_core_bounds.csv"
    bounds_csv = args.bounds_csv.resolve() if args.bounds_csv else output / "gse290577_core_bounds.csv"
    if args.bounds_csv is None and args.rds is not None and shutil.which(args.rscript):
        r_script = Path(__file__).with_name("export_core_bounds.R")
        subprocess.run([args.rscript, str(r_script), str(args.rds), str(bounds_csv)], check=True)
    elif args.bounds_csv is None:
        if not bundled_bounds.is_file():
            raise RuntimeError("Rscript is unavailable and bundled core bounds are missing")
        bounds_csv = bundled_bounds
        print(f"[INFO] Rscript unavailable; using bundled public core bounds: {bounds_csv}")

    assets = root / "spatial_assets"
    source_map = {
        "slide1": find_one(assets, "*22213*ome.ome.tif"),
        "slide2": find_one(assets, "*22215*ome.ome.tif"),
    }
    matrix_map = {
        "slide1": np.loadtxt(find_one(assets, "*22213*matrix.csv"), delimiter=","),
        "slide2": np.loadtxt(find_one(assets, "*22215*matrix.csv"), delimiter=","),
    }
    dimensions: dict[str, tuple[int, int, float, float]] = {}
    for name, path in source_map.items():
        with openslide.OpenSlide(str(path)) as slide:
            dimensions[name] = (
                int(slide.dimensions[0]), int(slide.dimensions[1]),
                float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),
                float(slide.properties[openslide.PROPERTY_NAME_MPP_Y]),
            )

    with bounds_csv.open(newline="", encoding="utf-8-sig") as handle:
        bounds = list(csv.DictReader(handle))
    rows: list[dict[str, str | int | float]] = []
    for bound in bounds:
        xenium_slide = bound["xenium_slide"].strip().lower()
        if xenium_slide not in source_map:
            raise ValueError(f"Unknown Xenium slide: {bound['xenium_slide']}")
        inverse = np.linalg.inv(matrix_map[xenium_slide])
        source_w, source_h, mpp_x, mpp_y = dimensions[xenium_slide]
        # Seurat centroids are in micrometers. The affine matrix operates in
        # Xenium morphology pixels; derive their spacing from the matrix scale.
        affine_scale = float(np.sqrt(abs(np.linalg.det(matrix_map[xenium_slide][:2, :2]))))
        fixed_mpp = min(mpp_x, mpp_y) / affine_scale
        margin = args.margin_um
        fixed_corners = np.asarray([
            [(float(bound["fixed_x_min"]) - margin) / fixed_mpp, (float(bound["fixed_y_min"]) - margin) / fixed_mpp, 1],
            [(float(bound["fixed_x_max"]) + margin) / fixed_mpp, (float(bound["fixed_y_min"]) - margin) / fixed_mpp, 1],
            [(float(bound["fixed_x_min"]) - margin) / fixed_mpp, (float(bound["fixed_y_max"]) + margin) / fixed_mpp, 1],
            [(float(bound["fixed_x_max"]) + margin) / fixed_mpp, (float(bound["fixed_y_max"]) + margin) / fixed_mpp, 1],
        ])
        image_points = (inverse @ fixed_corners.T).T
        image_points = image_points[:, :2] / image_points[:, 2, None]
        x0 = max(0, int(np.floor(image_points[:, 0].min())))
        y0 = max(0, int(np.floor(image_points[:, 1].min())))
        x1 = min(source_w, int(np.ceil(image_points[:, 0].max())))
        y1 = min(source_h, int(np.ceil(image_points[:, 1].max())))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Empty mapped ROI for {bound['sample_id']}")
        sample_id = re.sub(r"[^A-Za-z0-9._-]+", "_", bound["sample_id"]).strip("_")
        slide_id = f"GSE290577_core_{sample_id}"
        acr, amr = clean_grade(bound["acr_grade"]), clean_grade(bound["amr_grade"])
        rows.append({
            "case_id": bound["patient_id"], "patient_id": bound["patient_id"],
            "biopsy_id": bound["sample_id"], "slide_id": slide_id,
            "source_type": "core", "source_rel_path": source_map[xenium_slide].relative_to(root).as_posix(),
            "processing_rel_path": f"cores/{slide_id}", "stain": "HE", "stain_group": "HE",
            "acr_grade": acr, "amr_grade": amr, "rejection_group": bound["rejection_group"],
            **binary_labels(acr, amr),
            "roi_x": x0, "roi_y": y0, "roi_width": x1 - x0, "roi_height": y1 - y0,
            "width": source_w, "height": source_h, "mpp_x": mpp_x, "mpp_y": mpp_y,
            "xenium_slide": xenium_slide, "cell_count": bound["cell_count"],
            "fixed_mpp": fixed_mpp,
            "biopsy_timing": bound["biopsy_timing"], "days_from_transplant": bound["days_from_transplant"],
        })
    inventory = output / "gse290577_core_inventory.csv"
    with inventory.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"[OK] core inventory: {inventory}")
    print(f"Patients: {len({row['patient_id'] for row in rows})}; cores: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
