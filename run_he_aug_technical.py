#!/usr/bin/env python3
"""Generate and test four complete HE feature bags for fold-0 CLAM integration.

Copy to the SMC_MIL root and run `python run_he_aug_technical.py generate`,
then `preflight`, then `train`. This is not a performance experiment.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path("/home/jupyter/image_team/projects/SMC_MIL")
COHORT = Path("cohorts/he_manual_acr_v1/cohort.csv")
SPLIT = Path("cohorts/he_manual_acr_v1/splits_0.csv")
TECHNICAL_IDS = Path("cohorts/he_manual_acr_v1/technical_train_slides_0.txt")
BASELINE = Path("data/features/uni_v2/l0_0p25mpp_40x")
BANK = Path("data/features/uni_v2_aug/l0_0p25mpp_40x/fold_0")
RESULTS = Path("results/he_aug_technical_geometry_fold0")
STAIN = Path("dataset_csv/stain_aug_labels.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT)


def technical_sources() -> list[str]:
    for path in (COHORT, SPLIT, TECHNICAL_IDS, BASELINE / "pt_files", STAIN):
        if not (PROJECT / path).exists():
            raise FileNotFoundError(PROJECT / path)
    ids = [line.strip() for line in (PROJECT / TECHNICAL_IDS).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 4 or len(ids) != len(set(ids)):
        raise ValueError("Expected exactly four distinct technical train slides")
    cohort = {row["slide_id"]: row for row in read_csv(PROJECT / COHORT)}
    split = read_csv(PROJECT / SPLIT)
    train = {row["train"] for row in split if row["train"]}
    val = {row["val"] for row in split if row["val"]}
    if not set(ids) <= train or set(ids) & val:
        raise ValueError("Technical IDs are not all in fold-0 train")
    if not set(ids) <= cohort.keys():
        raise ValueError("Technical IDs are missing from cohort")
    if Counter(cohort[sid]["label"] for sid in ids) != {"0": 2, "1": 2}:
        raise ValueError("Technical IDs must contain two slides per ACR class")
    sources = {cohort[sid]["source_dataset"] for sid in ids}
    if not sources <= {"exp3", "mrxs13"}:
        raise ValueError(f"Unexpected source datasets: {sources}")
    print("Technical train slides:", [(sid, cohort[sid]["source_dataset"])
                                       for sid in ids], flush=True)
    return [source for source in ("exp3", "mrxs13") if source in sources]


def generate(sources: list[str]) -> None:
    for source in sources:
        if source == "exp3":
            source_args = [
                "--source", "dicom", "--source-dataset", "exp3",
                "--dataset-root", "/home/jupyter/data/image_team/exp3_inbox",
                "--feature-manifest",
                "/home/jupyter/data/image_team/exp3_inbox/_clam/dicom_feature_manifest.csv",
                "--check-original-patches", "8",
            ]
        else:
            source_args = [
                "--source", "mrxs", "--source-dataset", "mrxs13",
                "--dataset-root", "/home/jupyter/data/image_team/mrxs13_inbox",
                "--feature-manifest",
                "/home/jupyter/data/image_team/mrxs13_inbox/_clam/mrxs_feature_manifest_l0.csv",
                "--allow-mrxs-baseline-h5-coord-mismatch",
                "--check-original-patches", "16",
            ]
        command = [
            sys.executable, "augment_uni2_wsi.py", *source_args,
            "--baseline-dir", str(BASELINE),
            "--stain-csv", str(STAIN),
            "--task-csv", str(COHORT),
            "--split-csv", str(SPLIT),
            "--output-root", str(BANK),
            "--stain", "HE", "--label-source", "manual_only",
            "--policy", "geometry", "--num-views", "1",
            "--include-slide-ids", str(TECHNICAL_IDS),
            "--batch-size", "8", "--device", "cuda", "--amp",
        ]
        run(command)


def clam(stage: str) -> None:
    command = [
        sys.executable, "train_clam_feature_views.py",
        "--cohort-csv", str(COHORT), "--split-csv", str(SPLIT),
        "--baseline-dir", str(BASELINE), "--bank-root", str(BANK),
        "--policy", "geometry", "--total-views", "2", "--fold", "0",
        "--results-dir", str(RESULTS), "--technical-smoke",
        "--max-epochs", "1", "--deep-feature-check",
    ]
    if stage == "preflight":
        command.append("--preflight-only")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "preflight", "train"))
    args = parser.parse_args()
    if not PROJECT.is_dir():
        raise FileNotFoundError(PROJECT)
    sources = technical_sources()
    if args.stage == "generate":
        generate(sources)
    else:
        clam(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
