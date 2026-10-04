#!/usr/bin/env python3
"""Audit or classify SMC WSI stain families from frozen UNI2-h feature bags.

The curated v2 manifest is the label source. The older classifier manifest is
used only to report version differences. Patient-grouped CV uses the case_id
mapping available in the existing gold-label ACR CSV; unmapped slides are never
put into that CV. The final model can use all quality-usable, currently labeled
slides, but its predictions on those training slides are marked in-sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASSES = ("HE", "IHC", "special_other")
DEFAULT_MANIFEST = Path(
    "/home/jupyter/data/image_team/labels/derived/"
    "wsi_curation_v2_final/slide_curation_manifest_curated.csv"
)
DEFAULT_OLD_MANIFEST = Path(
    "/home/jupyter/data/image_team/labels/derived/"
    "stain_classifier_dataset_v1/dataset_manifest.csv"
)
DEFAULT_FEATURE_DIR = Path("data/features/uni_v2/l1_0p50mpp_20x/pt_files")
DEFAULT_PATIENT_CSV = Path("dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--old-manifest", type=Path, default=DEFAULT_OLD_MANIFEST)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--patient-csv", type=Path, default=DEFAULT_PATIENT_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("results/stain_family_uni_v1"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", action="store_true",
                        help="Extract slide vectors, cross-validate, and save predictions.")
    return parser.parse_args()


def read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.slide_id.duplicated().any():
        raise ValueError(f"{path}: duplicate slide_id values")
    return frame


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    current = read_csv(args.manifest, {"slide_id", "stain_group", "include_quality_usable"})
    patients = read_csv(args.patient_csv, {"slide_id", "case_id"})
    current = current.merge(
        patients[["slide_id", "case_id"]], on="slide_id", how="left",
        validate="one_to_one", suffixes=("", "_from_gold"),
    )
    if "case_id_from_gold" in current:
        current["case_id"] = current["case_id_from_gold"]
        current = current.drop(columns="case_id_from_gold")
    current["case_id"] = current.case_id.fillna("").astype(str)
    current["usable"] = current.include_quality_usable.str.lower().isin(
        ("true", "1", "yes")
    )
    current["feature_exists"] = current.slide_id.map(
        lambda slide_id: (args.feature_dir / f"{slide_id}.pt").is_file()
    )

    audit: dict[str, object] = {
        "manifest": str(args.manifest),
        "feature_dir": str(args.feature_dir),
        "patient_csv": str(args.patient_csv),
        "slides": len(current),
        "stain_groups": current.stain_group.value_counts(dropna=False).to_dict(),
        "usable_slides": int(current.usable.sum()),
        "usable_by_stain_group": current.loc[current.usable, "stain_group"].value_counts().to_dict(),
        "usable_with_feature": int((current.usable & current.feature_exists).sum()),
        "usable_with_feature_by_stain_group": current.loc[
            current.usable & current.feature_exists, "stain_group"
        ].value_counts().to_dict(),
        "usable_known_with_feature": int((
            current.usable & current.feature_exists & current.stain_group.isin(CLASSES)
        ).sum()),
        "usable_unknown_with_feature": int((
            current.usable & current.feature_exists & current.stain_group.eq("unknown")
        ).sum()),
        "usable_with_patient_case_id": int((current.usable & current.case_id.ne("")).sum()),
        "cv_eligible_by_stain_group": current.loc[
            current.usable & current.feature_exists & current.case_id.ne("")
            & current.stain_group.isin(CLASSES), "stain_group"
        ].value_counts().to_dict(),
    }
    if args.old_manifest.is_file():
        old = read_csv(args.old_manifest, {"slide_id", "stain_group"})
        joined = current[["slide_id", "stain_group"]].merge(
            old[["slide_id", "stain_group"]], on="slide_id", how="inner",
            suffixes=("_v2", "_v1"), validate="one_to_one",
        )
        audit["v1_shared_slides"] = len(joined)
        audit["v1_v2_different_stain_groups"] = int((
            joined.stain_group_v2 != joined.stain_group_v1
        ).sum())
    return current, audit


def vector_from_pt(path: Path) -> np.ndarray:
    import torch  # Audit mode does not need PyTorch.

    try:
        bag = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch releases lack weights_only.
        bag = torch.load(path, map_location="cpu")
    if not isinstance(bag, torch.Tensor):
        raise TypeError(f"Expected tensor feature bag: {path}")
    if bag.ndim != 2 or bag.shape[0] == 0 or bag.shape[1] != 1536:
        raise ValueError(f"Expected nonempty [patches,1536] bag, got {tuple(bag.shape)}: {path}")
    features = bag.detach().float().numpy()
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite feature bag: {path}")
    return np.concatenate((features.mean(axis=0), features.std(axis=0))).astype(np.float32)


def model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000,
                           random_state=seed),
    )


def grouped_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
               folds: int, seed: int) -> tuple[np.ndarray, dict[str, object]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = np.full((len(y), len(CLASSES)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_reports = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(x, y, groups)):
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise RuntimeError(f"Patient leakage in fold {fold}")
        if set(y[train_idx]) != set(CLASSES):
            raise ValueError(f"Fold {fold} training data does not contain all classes")
        estimator = model(seed + fold)
        estimator.fit(x[train_idx], y[train_idx])
        probabilities = estimator.predict_proba(x[val_idx])
        for column, label in enumerate(estimator.classes_):
            out[val_idx, CLASSES.index(label)] = probabilities[:, column]
        fold_ids[val_idx] = fold
        fold_reports.append({
            "fold": fold,
            "train_slides": len(train_idx),
            "validation_slides": len(val_idx),
            "train_patients": len(set(groups[train_idx])),
            "validation_patients": len(set(groups[val_idx])),
            "validation_classes": pd.Series(y[val_idx]).value_counts().to_dict(),
        })
    if np.isnan(out).any() or (fold_ids < 0).any():
        raise RuntimeError("Incomplete cross-validation predictions")
    predicted = np.asarray(CLASSES)[out.argmax(axis=1)]
    report = {
        "cv_slides": len(y),
        "cv_patients": len(set(groups)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=CLASSES, average="macro")),
        "classification_report": classification_report(
            y, predicted, labels=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": CLASSES,
        "confusion_matrix": confusion_matrix(y, predicted, labels=CLASSES).tolist(),
        "folds": fold_reports,
    }
    return out, {"report": report, "fold_ids": fold_ids}


def run_train(args: argparse.Namespace, current: pd.DataFrame,
              audit: dict[str, object]) -> None:
    eligible = current[current.usable & current.feature_exists].copy().reset_index(drop=True)
    vectors: list[np.ndarray] = []
    good_rows: list[int] = []
    failures: list[dict[str, str]] = []
    for index, slide_id in enumerate(eligible.slide_id):
        try:
            vectors.append(vector_from_pt(args.feature_dir / f"{slide_id}.pt"))
            good_rows.append(index)
        except Exception as exc:
            failures.append({"slide_id": slide_id, "error": f"{type(exc).__name__}: {exc}"})
        if (index + 1) % 100 == 0:
            print(f"Read {index + 1}/{len(eligible)} feature bags", flush=True)
    if not vectors:
        raise ValueError("No readable UNI2-h feature bags")
    eligible = eligible.iloc[good_rows].reset_index(drop=True)
    x = np.stack(vectors)
    labels = eligible.stain_group.to_numpy()
    known = eligible.stain_group.isin(CLASSES).to_numpy()
    mapped = eligible.case_id.ne("").to_numpy()
    cv_indices = np.flatnonzero(known & mapped)
    if len(cv_indices) < args.folds * len(CLASSES):
        raise ValueError("Too few mapped, labeled slides for grouped CV")
    y_cv = labels[cv_indices]
    groups = eligible.case_id.to_numpy()[cv_indices]
    oof, info = grouped_cv(x[cv_indices], y_cv, groups, args.folds, args.seed)

    final = model(args.seed)
    final.fit(x[known], labels[known])
    final_probs = final.predict_proba(x)
    columns = [list(final.classes_).index(label) for label in CLASSES]
    probs = final_probs[:, columns].copy()
    probs[cv_indices] = oof
    predicted = np.asarray(CLASSES)[probs.argmax(axis=1)]

    predictions = eligible[["slide_id", "stain_group", "case_id"]].copy()
    for field in ("stain_source", "stain_confidence", "stain_signature", "source_dataset"):
        if field in eligible:
            predictions[field] = eligible[field]
    predictions["model_prediction"] = predicted
    predictions["prob_HE"] = probs[:, 0]
    predictions["prob_IHC"] = probs[:, 1]
    predictions["prob_other"] = probs[:, 2]
    predictions["max_probability"] = probs.max(axis=1)
    predictions["prediction_type"] = np.where(
        known, "final_fit_in_sample", "final_fit_unlabeled"
    )
    predictions.loc[cv_indices, "prediction_type"] = "patient_grouped_oof"
    predictions["fold"] = pd.Series(pd.NA, index=predictions.index, dtype="Int64")
    predictions.loc[cv_indices, "fold"] = info["fold_ids"]
    predictions["disagrees_with_current"] = (
        predictions.prediction_type.eq("patient_grouped_oof")
        & predictions.stain_group.ne(predictions.model_prediction)
    )
    predictions["needs_review"] = (
        predictions.stain_group.eq("unknown") | predictions.disagrees_with_current
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "stain_predictions.csv", index=False)
    review = predictions[predictions.needs_review].sort_values("max_probability")
    review.to_csv(args.output_dir / "review_queue.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "unreadable_features.csv", index=False)
    missing = current[current.usable & ~current.feature_exists]
    missing[["slide_id", "stain_group"]].to_csv(
        args.output_dir / "missing_features.csv", index=False
    )
    joblib.dump(final, args.output_dir / "stain_classifier.joblib")
    audit.update(info["report"])
    audit["readable_feature_bags"] = len(eligible)
    audit["unreadable_feature_bags"] = len(failures)
    audit["review_queue_slides"] = len(review)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved model, CV metrics, predictions, and review queue to {args.output_dir}")
    print(f"Patient-grouped CV macro-F1: {audit['macro_f1']:.3f}")
    print(f"Review queue: {len(review)} slides (not automatically relabeled)")


def main() -> None:
    args = arguments()
    current, audit = prepare(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.train:
        run_train(args, current, audit)
    else:
        print("Audit only. Add --train to fit the classifier and save outputs.")


if __name__ == "__main__":
    main()
