#!/usr/bin/env python3
"""Full-cohort exploratory HE augmentation comparison on patient-grouped 3-fold CV.

Copy to the SMC_MIL root. `stage1` builds all train-slide 2-view banks and
trains original/geometry/geometry+H-E-scale on every fold. `stage2 --policy ...`
adds 3/5-view comparisons for one policy. Validation always uses original bags.
This is exploratory CV, not an independent final performance estimate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path("/home/jupyter/image_team/projects/SMC_MIL")
COHORT = Path("cohorts/he_manual_acr_v1/cohort.csv")
SPLIT_DIR = Path("cohorts/he_manual_acr_v1")
BASELINE = Path("data/features/uni_v2/l0_0p25mpp_40x")
BANK_BASE = Path("/home/jupyter/data/image_team/uni_v2_aug/l0_0p25mpp_40x")
RESULTS = Path("results/he_manual_acr_aug_explore_v2_best")
STAIN = Path("dataset_csv/stain_aug_labels.csv")
POLICIES = ("geometry", "geometry_he_scale_light")
SOURCES = {
    "exp3": (
        "dicom", "/home/jupyter/data/image_team/exp3_inbox",
        "/home/jupyter/data/image_team/exp3_inbox/_clam/dicom_feature_manifest.csv",
    ),
    "mrxs13": (
        "mrxs", "/home/jupyter/data/image_team/mrxs13_inbox",
        "/home/jupyter/data/image_team/mrxs13_inbox/_clam/mrxs_feature_manifest_l0.csv",
    ),
}


def records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT, check=True)
    print(f"Completed in {time.perf_counter() - started:.1f} s", flush=True)


def validate_inputs() -> tuple[dict[str, dict[str, str]], dict[int, dict[str, set[str]]]]:
    data_mount = Path("/home/jupyter/data").resolve()
    if not BANK_BASE.is_absolute() or data_mount not in BANK_BASE.resolve().parents:
        raise ValueError(f"Augmentation bank must be beneath {data_mount}: {BANK_BASE}")
    if BANK_BASE.resolve() == (PROJECT / BASELINE).resolve():
        raise ValueError("Augmentation bank cannot overwrite the original feature directory")
    required = [COHORT, STAIN, BASELINE / "pt_files", Path("augment_uni2_wsi.py"),
                Path("feature_view_bank.py"), Path("train_clam_feature_views.py")]
    required += [SPLIT_DIR / f"splits_{fold}.csv" for fold in range(3)]
    missing = [str(PROJECT / path) for path in required if not (PROJECT / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    cohort_rows = records(PROJECT / COHORT)
    cohort = {row["slide_id"]: row for row in cohort_rows}
    if len(cohort_rows) != 84 or len(cohort) != 84:
        raise ValueError(f"Expected the frozen 84-slide manual HE cohort, got {len(cohort_rows)}")
    if {row["stain_group"] for row in cohort_rows} != {"HE"}:
        raise ValueError("This workflow requires the HE-only cohort")
    if not {row["source_dataset"] for row in cohort_rows} <= SOURCES.keys():
        raise ValueError("Cohort contains an unsupported source dataset")
    if not {row["label"] for row in cohort_rows} == {"0", "1"}:
        raise ValueError("Cohort lacks one ACR class")
    for sid in cohort:
        if not (PROJECT / BASELINE / "pt_files" / f"{sid}.pt").is_file():
            raise FileNotFoundError(f"Original feature missing: {sid}")
    splits: dict[int, dict[str, set[str]]] = {}
    seen_val: set[str] = set()
    for fold in range(3):
        split_rows = records(PROJECT / SPLIT_DIR / f"splits_{fold}.csv")
        train_list = [row["train"] for row in split_rows if row["train"]]
        val_list = [row["val"] for row in split_rows if row["val"]]
        train, val = set(train_list), set(val_list)
        if (len(train_list) != len(train) or len(val_list) != len(val)
                or train & val or train | val != cohort.keys() or seen_val & val):
            raise ValueError(f"Invalid fold-{fold} slide partition")
        train_patients = {cohort[sid]["case_id"] for sid in train}
        val_patients = {cohort[sid]["case_id"] for sid in val}
        if train_patients & val_patients:
            raise ValueError(f"Patient leakage in fold {fold}")
        if {cohort[sid]["label"] for sid in train} != {"0", "1"}:
            raise ValueError(f"Train fold {fold} lacks a class")
        if {cohort[sid]["label"] for sid in val} != {"0", "1"}:
            raise ValueError(f"Validation fold {fold} lacks a class")
        seen_val.update(val)
        splits[fold] = {"train": train, "val": val}
    if seen_val != cohort.keys():
        raise ValueError("Validation folds do not cover every cohort slide")
    return cohort, splits


def estimate_storage(cohort: dict[str, dict[str, str]],
                     splits: dict[int, dict[str, set[str]]],
                     policies: tuple[str, ...], views: range,
                     *, enforce: bool = True) -> int:
    total = 0
    for fold in range(3):
        for sid in splits[fold]["train"]:
            original_bytes = (PROJECT / BASELINE / "pt_files" / f"{sid}.pt").stat().st_size
            for policy in policies:
                for view in views:
                    augmented = (PROJECT / BANK_BASE / f"fold_{fold}" / policy /
                                 f"view_{view}" / "pt_files" / f"{sid}.pt")
                    if not augmented.is_file():
                        total += original_bytes
    target = PROJECT / BANK_BASE
    while not target.exists():
        target = target.parent
    free = shutil.disk_usage(target).free
    print(f"Approximate additional feature storage: {total / 2**30:.2f} GiB; "
          f"free on output filesystem: {free / 2**30:.2f} GiB", flush=True)
    if enforce and free < total * 1.2:
        raise RuntimeError("Insufficient free feature storage with 20% safety margin")
    return total


def generate(cohort: dict[str, dict[str, str]],
             splits: dict[int, dict[str, set[str]]],
             policies: tuple[str, ...], max_view: int) -> None:
    estimate_storage(cohort, splits, policies, range(1, max_view + 1))
    for fold in range(3):
        for policy in policies:
            for source_name, (source_type, root, manifest) in SOURCES.items():
                ids = [sid for sid in splits[fold]["train"]
                       if cohort[sid]["source_dataset"] == source_name]
                if not ids:
                    continue
                command = [
                    sys.executable, "augment_uni2_wsi.py",
                    "--source", source_type, "--source-dataset", source_name,
                    "--dataset-root", root, "--feature-manifest", manifest,
                    "--baseline-dir", str(BASELINE),
                    "--stain-csv", str(STAIN), "--task-csv", str(COHORT),
                    "--split-csv", str(SPLIT_DIR / f"splits_{fold}.csv"),
                    "--output-root", str(BANK_BASE / f"fold_{fold}"),
                    "--stain", "HE", "--label-source", "manual_only",
                    "--policy", policy, "--num-views", str(max_view),
                    "--check-original-patches", "16" if source_type == "mrxs" else "8",
                    "--batch-size", "8", "--device", "cuda", "--amp",
                ]
                if source_type == "mrxs":
                    command.append("--allow-mrxs-baseline-h5-coord-mismatch")
                print(f"Fold {fold}: {policy}, {source_name}, {len(ids)} train slides", flush=True)
                run(command)


def result_dir(policy: str, views: int, fold: int) -> Path:
    name = "original_1view" if policy == "original" else f"{policy}_{views}view"
    return RESULTS / name / f"fold_{fold}"


def evaluate(policy: str, views: int, epochs: int, protocol: str = "best_val_v2") -> None:
    for fold in range(3):
        destination = result_dir(policy, views, fold)
        report_path = PROJECT / destination / "view_run_report.json"
        prediction_path = PROJECT / destination / "val_event_predictions.csv"
        if report_path.is_file() and prediction_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("training_protocol", "legacy200") != protocol
                    or report.get("policy") != policy or report.get("total_views") != views
                    or report.get("fold") != fold or report.get("max_epochs") != epochs
                    or report.get("technical_smoke") is not False
                    or report.get("cohort_sha256") != sha256(PROJECT / COHORT)
                    or report.get("split_sha256") != sha256(
                        PROJECT / SPLIT_DIR / f"splits_{fold}.csv")):
                raise RuntimeError(f"Completed result metadata differs: {report_path}")
            if protocol == "best_val_v2" and not (report_path.parent / "best_checkpoint.pt").is_file():
                raise FileNotFoundError(f"Missing selected checkpoint: {report_path.parent}")
            print(f"Already complete: {destination}", flush=True)
            continue
        if (PROJECT / destination).exists() and any((PROJECT / destination).iterdir()):
            raise RuntimeError(f"Incomplete result directory; inspect before retry: {destination}")
        command = [
            sys.executable, "train_clam_feature_views.py",
            "--cohort-csv", str(COHORT),
            "--split-csv", str(SPLIT_DIR / f"splits_{fold}.csv"),
            "--baseline-dir", str(BASELINE),
            "--policy", policy, "--total-views", str(views),
            "--fold", str(fold), "--results-dir", str(destination),
            "--seed", "1", "--view-seed", "20260918",
            "--sampler", "weighted", "--max-epochs", str(epochs),
            "--training-protocol", protocol,
        ]
        if policy != "original":
            command.extend(["--bank-root", str(BANK_BASE / f"fold_{fold}")])
        run(command + ["--preflight-only", "--deep-feature-check"])
        run(command)


def summarize(conditions: list[tuple[str, int]], label: str) -> None:
    import pandas as pd
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    rows = []
    pooled_rows = []
    for policy, views in conditions:
        pieces = []
        for fold in range(3):
            path = PROJECT / result_dir(policy, views, fold) / "val_event_predictions.csv"
            report_path = path.parent / "view_run_report.json"
            if not path.is_file() or not report_path.is_file():
                raise FileNotFoundError(f"Incomplete condition: {path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(path)
            frame["fold"] = fold
            frame["condition"] = f"{policy}_{views}view"
            rows.append({
                "condition": f"{policy}_{views}view", "fold": fold,
                "events": int(report["events"]),
                "positive_events": int(report["positive_events"]),
                "event_average_precision": float(report["event_average_precision"]),
                "event_roc_auc": float(report["event_roc_auc"]),
                "event_brier": float(report["event_brier"]),
                "best_epoch": report.get("best_epoch"),
                "epochs_ran": report.get("epochs_ran", report.get("max_epochs")),
                "optimizer_steps": report.get("optimizer_steps", report.get("train_draws")),
            })
            pieces.append(frame)
        pooled = pd.concat(pieces, ignore_index=True)
        if pooled.duplicated(["case_id", "event_id"]).any():
            raise ValueError(f"Event appears in multiple validation folds: {policy}, {views}")
        pooled_rows.append({
            "condition": f"{policy}_{views}view", "events": len(pooled),
            "positive_events": int(pooled["label"].sum()),
            "pooled_event_average_precision": float(average_precision_score(
                pooled["label"], pooled["prob_positive"])),
            "pooled_event_roc_auc": float(roc_auc_score(
                pooled["label"], pooled["prob_positive"])),
            "pooled_event_brier": float(brier_score_loss(
                pooled["label"], pooled["prob_positive"])),
        })
    fold_table = pd.DataFrame(rows)
    baseline = fold_table[fold_table["condition"] == "original_1view"].set_index("fold")
    fold_table["delta_event_ap_vs_original"] = [
        0.0 if row.condition == "original_1view" else
        row.event_average_precision - baseline.loc[row.fold, "event_average_precision"]
        for row in fold_table.itertuples(index=False)
    ]
    destination = PROJECT / RESULTS
    destination.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(destination / f"{label}_fold_event_metrics.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(destination / f"{label}_pooled_event_metrics.csv", index=False)
    print("Fold-level event metrics (exploratory):", flush=True)
    print(fold_table.to_string(index=False), flush=True)
    print("Pooled OOF metrics (not independent final performance):", flush=True)
    print(pd.DataFrame(pooled_rows).to_string(index=False), flush=True)


def main() -> int:
    global RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "stage1", "prepare-stage2", "stage2", "summary"))
    parser.add_argument("--policy", choices=(*POLICIES, "both"),
                        help="Policy for stage2; 'both' is allowed only for prepare-stage2")
    parser.add_argument("--gpu", type=int,
                        help="Required for prepare-stage2: an idle GPU, separate from Stage1 training")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--training-protocol", choices=("legacy200", "best_val_v2"), default="best_val_v2")
    parser.add_argument("--reuse-features", action="store_true",
                        help="Skip extraction; existing complete feature banks are still validated before training.")
    args = parser.parse_args()
    if args.epochs < 1 or (args.stage in ("stage2", "prepare-stage2") and not args.policy):
        parser.error("Use positive --epochs and specify --policy for stage2")
    if args.stage not in ("stage2", "prepare-stage2", "audit") and args.policy:
        parser.error("--policy is only used for stage2, prepare-stage2 or audit")
    if args.policy == "both" and args.stage != "prepare-stage2":
        parser.error("--policy both is only for prepare-stage2; choose one policy for training")
    if args.stage == "prepare-stage2":
        if args.reuse_features or args.gpu is None or args.gpu < 0:
            parser.error("prepare-stage2 requires --gpu N and must not use --reuse-features")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Feature generation only; CUDA_VISIBLE_DEVICES={args.gpu}. "
              "Ensure this GPU is idle and do not launch another writer to this bank.", flush=True)
    elif args.gpu is not None:
        parser.error("--gpu is currently only supported for prepare-stage2")
    if not PROJECT.is_dir():
        raise FileNotFoundError(PROJECT)
    RESULTS = Path("results/he_manual_acr_aug_explore_v2_best" if args.training_protocol == "best_val_v2"
                   else "results/he_manual_acr_aug_explore_v1")
    if args.training_protocol == "best_val_v2" and not (PROJECT / "clam_view_training.py").is_file():
        raise FileNotFoundError(PROJECT / "clam_view_training.py")
    print(f"Training protocol: {args.training_protocol}; results: {RESULTS}", flush=True)
    cohort, splits = validate_inputs()
    if args.stage == "audit":
        original_bytes = sum((PROJECT / BASELINE / "pt_files" / f"{sid}.pt").stat().st_size
                             for sid in cohort)
        print(f"Manual HE cohort: {len(cohort)} slides; original UNI2 bags: "
              f"{original_bytes / 2**30:.2f} GiB", flush=True)
        for fold in range(3):
            train = splits[fold]["train"]
            counts = {source: sum(cohort[sid]["source_dataset"] == source for sid in train)
                      for source in SOURCES}
            print(f"Fold {fold}: train={len(train)}, val={len(splits[fold]['val'])}, "
                  f"train sources={counts}", flush=True)
        try:
            import h5py

            patch_counts = []
            for sid in cohort:
                h5_path = PROJECT / BASELINE / "h5_files" / f"{sid}.h5"
                if not h5_path.is_file():
                    break
                with h5py.File(h5_path, "r") as handle:
                    patch_counts.append(int(handle["features"].shape[0]))
            if len(patch_counts) == len(cohort):
                total_patches = sum(patch_counts)
                print(f"Original cohort patch rows: {total_patches:,}; "
                      f"stage1 approx {4 * total_patches:,} patch encodes; "
                      f"stage2 selected policy approx {6 * total_patches:,} more", flush=True)
        except (ImportError, KeyError, OSError) as exc:
            print(f"Patch-count audit unavailable: {exc}", flush=True)
        print("Stage1 remaining storage:", flush=True)
        estimate_storage(cohort, splits, POLICIES, range(1, 2), enforce=False)
        for policy in ((args.policy,) if args.policy else POLICIES):
            print(f"Stage2 remaining storage after stage1, policy={policy}:", flush=True)
            estimate_storage(cohort, splits, (policy,), range(2, 5), enforce=False)
        print("Runtime cannot be predicted from file size alone; measure full-slide "
              "extraction and CLAM epoch throughput on this server.", flush=True)
    elif args.stage == "prepare-stage2":
        from feature_view_bank import validate_feature_bank

        policies = POLICIES if args.policy == "both" else (args.policy,)
        # Require an already complete Stage1 bank; never repair an active reader's view_1.
        for fold in range(3):
            active = [cohort[sid] for sid in sorted(splits[fold]["train"])]
            for policy in policies:
                validate_feature_bank(active, baseline_dir=PROJECT / BASELINE,
                                      bank_root=BANK_BASE / f"fold_{fold}",
                                      policy=policy, total_views=2)
        generate(cohort, splits, policies, 4)
        for fold in range(3):
            active = [cohort[sid] for sid in sorted(splits[fold]["train"])]
            for policy in policies:
                validate_feature_bank(active, baseline_dir=PROJECT / BASELINE,
                                      bank_root=BANK_BASE / f"fold_{fold}",
                                      policy=policy, total_views=5)
        print("Stage2 feature-bank preparation complete. No CLAM training or evaluation was run. "
              "After Stage1 completes, select a policy and run stage2 --reuse-features.", flush=True)
    elif args.stage == "stage1":
        if not args.reuse_features:
            generate(cohort, splits, POLICIES, 1)
        for policy, views in (("original", 1), ("geometry", 2),
                              ("geometry_he_scale_light", 2)):
            evaluate(policy, views, args.epochs, args.training_protocol)
        summarize([("original", 1), ("geometry", 2),
                   ("geometry_he_scale_light", 2)], "stage1")
    elif args.stage == "stage2":
        for policy, views in (("original", 1), (args.policy, 2)):
            for fold in range(3):
                report_path = PROJECT / result_dir(policy, views, fold) / "view_run_report.json"
                if not report_path.is_file():
                    raise FileNotFoundError(f"Run stage1 first: {report_path}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if (report.get("training_protocol", "legacy200") != args.training_protocol
                        or report.get("max_epochs") != args.epochs
                        or report.get("policy") != policy
                        or report.get("total_views") != views
                        or report.get("fold") != fold
                        or report.get("technical_smoke") is not False
                        or report.get("cohort_sha256") != sha256(PROJECT / COHORT)
                        or report.get("split_sha256") != sha256(
                            PROJECT / SPLIT_DIR / f"splits_{fold}.csv")):
                    raise ValueError(f"Stage2 cannot reuse mismatched stage1 result: {report_path}")
        if not args.reuse_features:
            generate(cohort, splits, (args.policy,), 4)
        for views in (3, 5):
            evaluate(args.policy, views, args.epochs, args.training_protocol)
        summarize([("original", 1), (args.policy, 2),
                   (args.policy, 3), (args.policy, 5)], "stage2")
    else:
        conditions = [("original", 1), ("geometry", 2),
                      ("geometry_he_scale_light", 2)]
        if (PROJECT / result_dir("geometry", 3, 0)).is_dir():
            conditions.extend((("geometry", 3), ("geometry", 5)))
        if (PROJECT / result_dir("geometry_he_scale_light", 3, 0)).is_dir():
            conditions.extend((("geometry_he_scale_light", 3),
                               ("geometry_he_scale_light", 5)))
        summarize(conditions, "all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
