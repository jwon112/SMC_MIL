#!/usr/bin/env python3
"""Evaluate every compatible internal CLAM experiment on GSE290577."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


TASKS = {
    "acr_0r_vs_rest": "acr_any",
    "acr_high_grade": "acr_high",
    "amr_positive": "amr_positive",
    "any_rejection": "any_rejection",
    "significant_rejection": "significant_rejection",
}
SCALES = {
    ("l0", "0p25", "40x"): "0p25",
    ("l1", "0p50", "20x"): "0p5",
    ("l2", "1p00", "10x"): "1",
    ("l3", "2p00", "5x"): "2",
}
RESULT_PATTERN = re.compile(
    r"^smc_(?P<prefix>acr_0r_vs_rest|acr_high_grade|amr_positive|any_rejection|significant_rejection)_"
    r"(?P<level>l[0-3])_(?P<mpp>0p25|0p50|1p00|2p00)mpp_(?P<mag>40x|20x|10x|5x)_"
    r"uni2_clamsb(?P<variant>.+)_s1$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--work-root", type=Path, default=Path("/home/jupyter/data/image_team/GSE290577_work"))
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data/features/uni_v2/GSE290577"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/gse290577_external/all_internal_experiments"),
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include-task", action="append", choices=sorted(set(TASKS.values())))
    parser.add_argument("--include-variant", action="append", help="Only run variants containing this text; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover(args: argparse.Namespace) -> list[dict[str, object]]:
    experiments = []
    for path in sorted(args.results_root.iterdir()):
        checkpoints = list(path.glob("s_*_checkpoint.pt")) if path.is_dir() else []
        if not checkpoints or not (path / "summary.csv").is_file():
            continue
        match = RESULT_PATTERN.match(path.name)
        if not match:
            continue
        fields = match.groupdict()
        task = TASKS[fields["prefix"]]
        variant = fields["variant"].lstrip("_")
        fold_match = re.search(r"cv(?P<folds>[35])val", variant)
        if fold_match and len(checkpoints) != int(fold_match.group("folds")):
            continue
        if args.include_task and task not in args.include_task:
            continue
        if args.include_variant and not any(value in variant for value in args.include_variant):
            continue
        tag = SCALES[(fields["level"], fields["mpp"], fields["mag"])]
        experiments.append({
            "experiment": path.name,
            "checkpoint_dir": path,
            "checkpoint_count": len(checkpoints),
            "task": task,
            "variant": variant,
            "level": fields["level"],
            "mpp": fields["mpp"],
            "magnification": fields["mag"],
            "feature_tag": tag,
        })
    return experiments


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    script = Path(__file__).resolve().parents[2] / "evaluate_external_clam.py"
    experiments = discover(args)
    if not experiments:
        raise FileNotFoundError(f"No compatible checkpoint directories found under {args.results_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_root / "experiment_inventory.csv", experiments)
    failures = 0

    for index, experiment in enumerate(experiments, start=1):
        output_dir = args.output_root / str(experiment["experiment"])
        summary_path = output_dir / "external_summary.csv"
        if summary_path.is_file() and not args.overwrite:
            print(f"[SKIP {index}/{len(experiments)}] {experiment['experiment']}", flush=True)
            continue

        tag = str(experiment["feature_tag"])
        cohorts = [
            ("wsi_he", args.work_root / "manifests/gse290577_wsi_he.csv", args.feature_root / f"{tag}mpp/wsi"),
            ("wsi_ihc", args.work_root / "manifests/gse290577_wsi_ihc.csv", args.feature_root / f"{tag}mpp/wsi"),
        ]
        if tag == "0p25":
            cohorts.append(("core", args.work_root / "manifests/gse290577_core_inventory.csv", args.feature_root / "0p25mpp/core"))

        command = [
            sys.executable, str(script),
            "--checkpoint-dir", str(experiment["checkpoint_dir"]),
            "--task", str(experiment["task"]),
            "--output-dir", str(output_dir),
            "--embed-dim", "1536",
            "--threshold", str(args.threshold),
            "--bootstrap", str(args.bootstrap),
            "--device", args.device,
        ]
        for name, manifest, features in cohorts:
            command.extend(("--cohort", name, str(manifest), str(features)))

        print(f"[RUN {index}/{len(experiments)}] {experiment['experiment']}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            failures += 1
            print(f"[FAIL] exit_code={result.returncode}: {experiment['experiment']}", flush=True)

    combined = []
    for experiment in experiments:
        summary_path = args.output_root / str(experiment["experiment"]) / "external_summary.csv"
        if not summary_path.is_file():
            continue
        metadata = {key: value for key, value in experiment.items() if key != "checkpoint_dir"}
        for row in read_rows(summary_path):
            combined.append({**metadata, **row})
    write_rows(args.output_root / "all_external_summary.csv", combined)
    print(f"[DONE] experiments={len(experiments)} summaries={len(combined)} failures={failures}")
    print(f"Combined summary: {args.output_root / 'all_external_summary.csv'}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
