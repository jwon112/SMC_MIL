#!/usr/bin/env python3
"""Build patient-grouped CLAM label CSVs from Sheet1 pathology labels.

The generated ``case_id`` is a stable pseudonym, not a raw patient identifier.
Each output row is one physical WSI bag; all bags from the same patient share a
case_id so CLAM's patient-level split logic keeps them in one fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TASKS = {
    "smc_acr_4class_0r_1r_2r_3r": {
        "source_column": "ACR 등급",
        "labels": {"0R": (0, "0R"), "1R": (1, "1R"), "2R": (2, "2R"), "3R": (3, "3R")},
    },
    "smc_acr_binary_0r_vs_1r2r3r": {
        "source_column": "ACR 등급",
        "labels": {"0R": (0, "0R"), "1R": (1, "1R/2R/3R"), "2R": (1, "1R/2R/3R"), "3R": (1, "1R/2R/3R")},
    },
    "smc_amr_binary_pamr0_vs_positive": {
        "source_column": "AMR 등급",
        "labels": {
            "pAMR0": (0, "pAMR0"),
            "pAMR1": (1, "pAMR1/pAMR1(I+)/pAMR2"),
            "pAMR1(I+)": (1, "pAMR1/pAMR1(I+)/pAMR2"),
            "pAMR2": (1, "pAMR1/pAMR1(I+)/pAMR2"),
        },
    },
}


@dataclass(frozen=True)
class Slide:
    source_dataset: str
    slide_rel_path: str
    slide_id: str
    event_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-xlsx", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--mrxs-root", type=Path, required=True)
    parser.add_argument("--quality-exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_csv"))
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def feature_slide_id(slide_rel_path: str) -> str:
    raw = "__".join(part for part in slide_rel_path.replace("\\", "/").split("/") if part)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(".")


def pseudonymous_case_id(patient_id: object) -> str:
    digest = hashlib.sha256(f"SMC_MIL_patient:{patient_id}".encode("utf-8")).hexdigest()[:16]
    return f"patient_{digest}"


def load_slides(dataset_root: Path, source_dataset: str) -> list[Slide]:
    qc_path = dataset_root / "_qc" / "mask_qc_labels.csv"
    if not qc_path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {qc_path}")

    slides: list[Slide] = []
    with qc_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rel_path = row["slide_rel_path"].strip().replace("\\", "/")
            if not rel_path:
                continue
            event_key = normalize(rel_path.split("/")[0])
            slides.append(Slide(source_dataset, rel_path, feature_slide_id(rel_path), event_key))
    return slides


def load_exclusions(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["slide_rel_path"].strip().replace("\\", "/").lower()
            for row in csv.DictReader(handle)
            if row.get("slide_rel_path", "").strip()
        }


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_excel(path, sheet_name="Sheet1")
    required = {"ID", "생검일자", "ACR 등급", "AMR 등급", "병리ID"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Sheet1 is missing columns: {sorted(missing)}")
    labels = labels.dropna(subset=["ID", "병리ID"]).copy()
    labels["event_key"] = labels["병리ID"].map(normalize)
    if labels["event_key"].duplicated().any():
        duplicates = labels.loc[labels["event_key"].duplicated(keep=False), "병리ID"].tolist()
        raise ValueError(f"Sheet1 pathology IDs must be unique; duplicates: {duplicates[:10]}")
    return labels.set_index("event_key", verify_integrity=True)


def task_rows(slides: list[Slide], labels: pd.DataFrame, task: dict[str, object]) -> list[dict[str, str | int]]:
    source_column = str(task["source_column"])
    label_map = dict(task["labels"])
    rows: list[dict[str, str | int]] = []
    for slide in slides:
        if slide.event_key not in labels.index:
            continue
        record = labels.loc[slide.event_key]
        grade = str(record[source_column]).strip()
        if grade not in label_map:
            continue
        numeric_label, label_text = label_map[grade]
        rows.append(
            {
                "case_id": pseudonymous_case_id(record["ID"]),
                "slide_id": slide.slide_id,
                "label": numeric_label,
                "label_text": label_text,
                "event_id": str(record["병리ID"]),
                "source_dataset": slide.source_dataset,
                "slide_rel_path": slide.slide_rel_path,
                "biopsy_date": pd.Timestamp(record["생검일자"]).date().isoformat(),
            }
        )
    return rows


def write_task(path: Path, rows: list[dict[str, str | int]]) -> None:
    fields = ["case_id", "slide_id", "label", "label_text", "event_id", "source_dataset", "slide_rel_path", "biopsy_date"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    labels = load_labels(args.label_xlsx)
    exclusions = load_exclusions(args.quality_exclusions)
    dicom_slides = [slide for slide in load_slides(args.dicom_root, "exp3") if slide.slide_rel_path.lower() not in exclusions]
    mrxs_slides = load_slides(args.mrxs_root, "mrxs13")
    slides = dicom_slides + mrxs_slides

    duplicate_ids = [slide_id for slide_id, count in Counter(slide.slide_id for slide in slides).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Feature slide_id collision(s): {duplicate_ids[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input slide bags: exp3={len(dicom_slides)}, mrxs13={len(mrxs_slides)}, excluded={len(exclusions)}")
    for task_name, task in TASKS.items():
        rows = task_rows(slides, labels, task)
        output_path = args.output_dir / f"{task_name}.csv"
        write_task(output_path, rows)
        counts = Counter(row["label"] for row in rows)
        print(
            f"[OK] {task_name}: bags={len(rows)}, events={len({row['event_id'] for row in rows})}, "
            f"patients={len({row['case_id'] for row in rows})}, class_bags={dict(sorted(counts.items()))} -> {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
