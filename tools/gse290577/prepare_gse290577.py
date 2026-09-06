#!/usr/bin/env python3
"""Inventory GSE290577 IHC WSI and create conservative external labels."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import openslide


STAINS = {"HE": "HE", "CD3": "IHC", "CD8": "IHC", "CD68": "IHC"}


def parse_name(path: Path) -> tuple[str, str] | None:
    match = re.fullmatch(r"(ACR-[1-5]|AMR-[1-5]|Non-rejecting-[1-5])(?:_(CD3|CD8|CD68))?", path.stem)
    if not match:
        return None
    return match.group(1), match.group(2) or "HE"


def labels_for(biopsy_id: str) -> dict[str, str]:
    if biopsy_id.startswith("ACR-"):
        return {"acr_grade": "2R", "amr_grade": "", "rejection_group": "ACR"}
    if biopsy_id.startswith("AMR-"):
        index = int(biopsy_id.split("-")[1])
        return {
            "acr_grade": "0R" if index <= 2 else "1R",
            "amr_grade": "pAMR1-i" if index <= 3 else "pAMR2",
            "rejection_group": "AMR",
        }
    return {"acr_grade": "0R", "amr_grade": "pAMR0", "rejection_group": "non_rejecting"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    source_dir = root / "ihc_wsi" / "IHC_raw_images"
    rows: list[dict[str, str | int | float]] = []
    controls: list[str] = []
    for path in sorted(source_dir.glob("*.svs")):
        parsed = parse_name(path)
        if parsed is None:
            controls.append(path.relative_to(root).as_posix())
            continue
        biopsy_id, stain = parsed
        slide_id = f"GSE290577_{biopsy_id.replace('-', '_')}_{stain}"
        with openslide.OpenSlide(str(path)) as slide:
            mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            width, height = slide.dimensions
        labels = labels_for(biopsy_id)
        rows.append({
            "case_id": biopsy_id,
            "patient_id": biopsy_id,
            "biopsy_id": biopsy_id,
            "slide_id": slide_id,
            "source_type": "wsi",
            "source_rel_path": path.relative_to(root).as_posix(),
            "processing_rel_path": f"wsi/{slide_id}",
            "stain": stain,
            "stain_group": STAINS[stain],
            **labels,
            "acr_any_label": int(labels["acr_grade"] != "0R"),
            "acr_high_label": int(labels["acr_grade"] in {"2R", "3R"}),
            "amr_positive_label": "" if labels["amr_grade"] == "" else int(labels["amr_grade"] != "pAMR0"),
            "any_rejection_label": int(labels["acr_grade"] != "0R" or labels["amr_grade"] not in {"", "pAMR0"}),
            "significant_rejection_label": (
                1 if labels["acr_grade"] in {"2R", "3R"}
                else "" if labels["amr_grade"] == ""
                else int(labels["amr_grade"] != "pAMR0")
            ),
            "roi_x": 0, "roi_y": 0, "roi_width": width, "roi_height": height,
            "width": width, "height": height, "mpp_x": mpp_x, "mpp_y": mpp_y,
        })
    output.mkdir(parents=True, exist_ok=True)
    inventory = output / "gse290577_wsi_inventory.csv"
    subsets = {
        inventory: rows,
        output / "gse290577_wsi_he.csv": [row for row in rows if row["stain_group"] == "HE"],
        output / "gse290577_wsi_ihc.csv": [row for row in rows if row["stain_group"] == "IHC"],
    }
    for path, subset in subsets.items():
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(subset)
    (output / "excluded_controls.txt").write_text("\n".join(controls) + "\n", encoding="utf-8")
    print(f"[OK] WSI inventory: {inventory}")
    print(f"Biopsies: {len({row['biopsy_id'] for row in rows})}; slides: {len(rows)} (HE=15, IHC=45); excluded controls: {len(controls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
