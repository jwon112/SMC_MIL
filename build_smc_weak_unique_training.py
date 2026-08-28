#!/usr/bin/env python3
"""Build train-only weak-label augmentations from unique date-linked WSI slides.

Gold pathology-ID labels and their existing 3-fold validation splits are kept
unchanged.  Only the train column receives weak slides, and a weak slide is
excluded from a fold whenever its inferred EHR patient is in that fold's
gold-validation set.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


TASKS = {
    "smc_acr_binary_0r_vs_1r2r3r": "smc_cv_acr_0r_vs_1r2r3r_standard3",
    "smc_acr_binary_0r1r_vs_2r3r": "smc_cv_acr_0r1r_vs_2r3r_standard3",
    "smc_amr_binary_pamr0_vs_positive": "smc_cv_amr_pamr0_vs_positive_standard3",
    "smc_any_rejection_binary": "smc_cv_any_rejection_standard3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-dates-csv", type=Path, required=True)
    parser.add_argument("--label-xlsx", type=Path, required=True)
    parser.add_argument("--ehr-xlsx", type=Path, required=True)
    parser.add_argument("--base-csv-dir", type=Path, default=Path("dataset_csv"))
    parser.add_argument("--base-splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=3)
    return parser.parse_args()


def normalize(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def feature_slide_id(slide_rel_path: str) -> str:
    raw = "__".join(part for part in slide_rel_path.replace("\\", "/").split("/") if part)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip(".")


def case_id(patient_id: object) -> str:
    digest = hashlib.sha256(f"SMC_MIL_patient:{patient_id}".encode("utf-8")).hexdigest()[:16]
    return f"patient_{digest}"


def read_gold(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    if frame.shape[1] < 5:
        raise ValueError("Gold workbook Sheet1 must have at least five columns")
    gold = pd.DataFrame({
        "gold_smc_id": pd.to_numeric(frame.iloc[:, 0], errors="coerce").astype("Int64"),
        "gold_biopsy_date": pd.to_datetime(frame.iloc[:, 1], errors="coerce").dt.normalize(),
        "pathology_key": frame.iloc[:, 4].map(normalize),
    }).dropna(subset=["gold_smc_id", "gold_biopsy_date"])
    return gold[gold.pathology_key.ne("")].drop_duplicates("pathology_key")


def read_ehr(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="biopsy_features")
    if frame.shape[1] < 8 or str(frame.columns[0]) != "biopsy_id":
        raise ValueError("Unexpected EHR biopsy_features layout")
    ehr = pd.DataFrame({
        "candidate_biopsy_id": frame.iloc[:, 0].astype(str).str.strip(),
        "candidate_smc_id": pd.to_numeric(frame.iloc[:, 4], errors="coerce").astype("Int64"),
        "candidate_biopsy_date": pd.to_datetime(frame.iloc[:, 5], errors="coerce").dt.normalize(),
        "acr_grade": frame.iloc[:, 6].astype(str).str.strip(),
        "amr_grade": frame.iloc[:, 7].astype(str).str.strip(),
    }).dropna(subset=["candidate_smc_id", "candidate_biopsy_date"])
    if ehr.candidate_biopsy_id.duplicated().any():
        raise ValueError("EHR biopsy_id must be unique")
    return ehr


def task_label(task: str, acr: str, amr: str) -> tuple[int, str] | None:
    if task == "smc_acr_binary_0r_vs_1r2r3r":
        mapping = {"0R": (0, "0R"), "1R": (1, "1R/2R/3R"), "2R": (1, "1R/2R/3R"), "3R": (1, "1R/2R/3R")}
    elif task == "smc_acr_binary_0r1r_vs_2r3r":
        mapping = {"0R": (0, "0R/1R"), "1R": (0, "0R/1R"), "2R": (1, "2R/3R"), "3R": (1, "2R/3R")}
    elif task == "smc_amr_binary_pamr0_vs_positive":
        mapping = {"pAMR0": (0, "pAMR0"), "pAMR1": (1, "pAMR-positive"), "pAMR1(I+)": (1, "pAMR-positive"), "pAMR2": (1, "pAMR-positive")}
        return mapping.get(amr)
    else:
        if acr not in {"0R", "1R", "2R", "3R"} or amr not in {"pAMR0", "pAMR1", "pAMR1(I+)", "pAMR2"}:
            return None
        return (int(acr != "0R" or amr != "pAMR0"), "ACR>=1R or AMR-positive" if acr != "0R" or amr != "pAMR0" else "ACR 0R and pAMR0")
    return mapping.get(acr)


def build_manifest(slides: pd.DataFrame, gold: pd.DataFrame, ehr: pd.DataFrame, window_days: int) -> pd.DataFrame:
    gold_keys = set(gold.pathology_key)
    slide_rows = slides[~slides.case_folder_key.isin(gold_keys)].copy()
    # Only pathology IDs that actually occur as an exp3 top-level WSI folder
    # are existing gold WSI anchors.  The workbook contains additional labeled
    # EHR events with no directly matched WSI folder; those remain new-event
    # candidates here.
    matched_gold = gold[gold.pathology_key.isin(set(slides.case_folder_key))]
    anchored = {(int(row.gold_smc_id), row.gold_biopsy_date) for row in matched_gold.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for slide in slide_rows.itertuples(index=False):
        scan_date = pd.Timestamp(slide.scan_date)
        candidates = ehr[(ehr.candidate_biopsy_date <= scan_date) & (ehr.candidate_biopsy_date >= scan_date - pd.Timedelta(days=window_days))]
        if len(candidates) != 1:
            continue
        candidate = candidates.iloc[0]
        anchored_candidate = (int(candidate.candidate_smc_id), candidate.candidate_biopsy_date) in anchored
        rows.append({
            "slide_id": feature_slide_id(slide.slide_rel_path),
            "slide_rel_path": slide.slide_rel_path,
            "source_dataset": "exp3",
            "case_id": case_id(candidate.candidate_smc_id),
            "candidate_biopsy_id": candidate.candidate_biopsy_id,
            "candidate_smc_id": int(candidate.candidate_smc_id),
            "candidate_biopsy_date": candidate.candidate_biopsy_date.date().isoformat(),
            "scan_date": scan_date.date().isoformat(),
            "scan_minus_biopsy_days": int((scan_date - candidate.candidate_biopsy_date).days),
            "acr_grade": candidate.acr_grade,
            "amr_grade": candidate.amr_grade,
            "weak_group": "possible_gold_extension" if anchored_candidate else "weak_unique_new",
            "candidate_is_gold_anchored": anchored_candidate,
        })
    return pd.DataFrame(rows).sort_values(["weak_group", "slide_id"]).reset_index(drop=True)


def write_augmented_task(task: str, manifest: pd.DataFrame, args: argparse.Namespace) -> None:
    base_path = args.base_csv_dir / f"{task}.csv"
    base = pd.read_csv(base_path)
    base["label_source"] = "gold_direct"
    base["weak_group"] = ""
    weak_rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        mapped = task_label(task, row.acr_grade, row.amr_grade)
        if mapped is None:
            continue
        label, label_text = mapped
        weak_rows.append({
            "case_id": row.case_id,
            "slide_id": row.slide_id,
            "label": label,
            "label_text": label_text,
            "event_id": row.candidate_biopsy_id,
            "source_dataset": row.source_dataset,
            "slide_rel_path": row.slide_rel_path,
            "biopsy_date": row.candidate_biopsy_date,
            "label_source": "weak_unique_date",
            "weak_group": row.weak_group,
        })
    weak = pd.DataFrame(weak_rows)
    combined = pd.concat([base, weak], ignore_index=True, sort=False)
    if combined.slide_id.duplicated().any():
        raise ValueError(f"Duplicate slide IDs in {task} augmentation")
    csv_dir = args.output_dir / "dataset_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    combined_path = csv_dir / f"{task}_weak_unique_0to{args.window_days}.csv"
    combined.to_csv(combined_path, index=False)

    split_dir = args.base_splits_dir / TASKS[task]
    output_splits = args.output_dir / "splits" / f"{TASKS[task]}_weak_unique_0to{args.window_days}"
    output_splits.mkdir(parents=True, exist_ok=True)
    base_cases = base.set_index("slide_id")["case_id"]
    report: list[dict[str, object]] = []
    for fold in range(3):
        split = pd.read_csv(split_dir / f"splits_{fold}.csv")
        val_ids = split["val"].dropna().astype(str).tolist()
        val_cases = set(base_cases.loc[val_ids])
        fold_weak = weak[~weak.case_id.isin(val_cases)]
        train_ids = split["train"].dropna().astype(str).tolist() + fold_weak.slide_id.tolist()
        out = pd.DataFrame({"train": pd.Series(train_ids, dtype="object"), "val": pd.Series(val_ids, dtype="object"), "test": pd.Series(dtype="object")})
        out.to_csv(output_splits / f"splits_{fold}.csv", index=False)
        report.append({"fold": fold, "base_train_bags": len(split["train"].dropna()), "weak_train_bags": len(fold_weak), "weak_unique_new": int((fold_weak.weak_group == "weak_unique_new").sum()), "possible_gold_extension": int((fold_weak.weak_group == "possible_gold_extension").sum()), "validation_bags_unchanged": len(val_ids)})
    pd.DataFrame(report).to_csv(output_splits / "fold_summary.csv", index=False)
    print(f"[OK] {task}: combined={len(combined)} weak={len(weak)} -> {combined_path}")
    print(pd.DataFrame(report).to_string(index=False))


def main() -> int:
    args = parse_args()
    if args.window_days < 0:
        raise ValueError("--window-days must be non-negative")
    slides = pd.read_csv(args.slide_dates_csv)
    required = {"case_folder_key", "slide_rel_path", "scan_date"}
    if missing := required.difference(slides.columns):
        raise ValueError(f"Slide date CSV missing columns: {sorted(missing)}")
    gold = read_gold(args.label_xlsx)
    ehr = read_ehr(args.ehr_xlsx)
    manifest = build_manifest(slides, gold, ehr, args.window_days)
    if manifest.empty:
        raise ValueError("No unique weak candidates found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / f"weak_unique_slide_manifest_0to{args.window_days}.csv", index=False)
    print("Weak manifest:")
    print(manifest.groupby(["weak_group", "acr_grade", "amr_grade"]).size().to_string())
    for task in TASKS:
        write_augmented_task(task, manifest, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
