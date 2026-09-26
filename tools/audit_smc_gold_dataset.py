#!/usr/bin/env python3
"""Create a reproducible bias, multiplicity, and split audit for the SMC gold cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TASK_FILES = {
    "acr_high": "smc_acr_binary_0r1r_vs_2r3r.csv",
    "amr_positive": "smc_amr_binary_pamr0_vs_positive.csv",
    "significant_rejection": "smc_significant_rejection_binary.csv",
}
SCALES = {
    "40x": "l0_0p25mpp_40x",
    "20x": "l1_0p50mpp_20x",
    "10x": "l2_1p00mpp_10x",
    "5x": "l3_2p00mpp_5x",
}
BASE_COLUMNS = {
    "case_id", "slide_id", "event_id", "label", "source_dataset", "biopsy_date",
}
TRUE_VALUES = {"1", "true", "yes", "y"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, default=Path("dataset_csv"))
    parser.add_argument("--curation-manifest", type=Path)
    parser.add_argument(
        "--feature-root",
        type=Path,
        help="Optional UNI v2 root containing l0_0p25mpp_40x/pt_files and the other scales.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, choices=(3, 5), default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 11, 21, 31, 41])
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def as_bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def read_task(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"case_id": str, "slide_id": str, "event_id": str, "source_dataset": str},
    )
    missing = BASE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.slide_id.duplicated().any():
        raise ValueError(f"Duplicate slide_id in {path}")
    frame["label"] = pd.to_numeric(frame.label, errors="raise").astype(int)
    if set(frame.label.unique()).difference({0, 1}):
        raise ValueError(f"Non-binary labels in {path}")
    frame["biopsy_date"] = pd.to_datetime(frame.biopsy_date, errors="raise").dt.normalize()
    return frame


def derive_significant(acr: pd.DataFrame, amr: pd.DataFrame) -> pd.DataFrame:
    keys = ["slide_id", "case_id", "event_id", "source_dataset", "biopsy_date"]
    joined = acr[keys + ["label"]].merge(
        amr[keys + ["label"]], on=keys, how="inner", validate="one_to_one", suffixes=("_acr", "_amr")
    )
    if len(joined) != len(acr) or len(joined) != len(amr):
        raise ValueError("Cannot derive significant rejection: ACR and AMR slide cohorts differ")
    joined["label"] = (joined.label_acr.eq(1) | joined.label_amr.eq(1)).astype(int)
    joined["label_text"] = "ACR>=2R or AMR-positive"
    return joined.drop(columns=["label_acr", "label_amr"])


def load_tasks(label_dir: Path) -> dict[str, pd.DataFrame]:
    tasks: dict[str, pd.DataFrame] = {}
    for task in ("acr_high", "amr_positive"):
        path = label_dir / TASK_FILES[task]
        if not path.is_file():
            raise FileNotFoundError(path)
        tasks[task] = read_task(path)
    significant_path = label_dir / TASK_FILES["significant_rejection"]
    tasks["significant_rejection"] = (
        read_task(significant_path)
        if significant_path.is_file()
        else derive_significant(tasks["acr_high"], tasks["amr_positive"])
    )
    reference = set(tasks["acr_high"].slide_id)
    for task, frame in tasks.items():
        if set(frame.slide_id) != reference:
            raise ValueError(f"{task} does not contain the same gold slides as acr_high")
    return tasks


def load_curation(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"slide_id": str}).fillna("")
    if "slide_id" not in frame:
        raise ValueError("Curation manifest requires slide_id")
    if frame.slide_id.duplicated().any():
        raise ValueError("Curation manifest contains duplicate slide_id")
    return frame


def normalize_stain(value: object) -> str:
    observed = str(value).strip().upper()
    if observed in {"HE", "H&E"}:
        return "HE"
    if observed == "IHC":
        return "IHC"
    if observed in {"OTHER", "SPECIAL", "SPECIAL_OTHER", "SPECIAL STAIN"}:
        return "other"
    return "unknown"


def slide_metadata(reference: pd.DataFrame, curation: pd.DataFrame | None) -> pd.DataFrame:
    slides = reference[["case_id", "event_id", "slide_id", "source_dataset", "biopsy_date"]].copy()
    if curation is None:
        return slides
    optional = [
        "stain_group", "include_quality_usable", "include_quality_clean", "quality_status_final",
        "patch_count_l0", "tissue_ratio", "sharpness_score", "tissue_luminance_mean",
        "tissue_luminance_std", "tissue_saturation_mean", "grid_periodicity_score",
    ]
    columns = ["slide_id", *[column for column in optional if column in curation]]
    slides = slides.merge(curation[columns], on="slide_id", how="left", validate="one_to_one")
    if "stain_group" in slides:
        slides["stain_group"] = slides.stain_group.map(normalize_stain)
    return slides


def unique_or_error(values: pd.Series, name: str) -> object:
    observed = values.dropna().unique()
    if len(observed) != 1:
        raise ValueError(f"Event has inconsistent {name}: {observed.tolist()}")
    return observed[0]


def build_event_metadata(slides: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quality_numeric = [
        "patch_count_l0", "tissue_ratio", "sharpness_score", "tissue_luminance_mean",
        "tissue_luminance_std", "tissue_saturation_mean", "grid_periodicity_score",
    ]
    for event_id, group in slides.groupby("event_id", sort=False):
        row: dict[str, object] = {
            "event_id": event_id,
            "case_id": unique_or_error(group.case_id, "case_id"),
            "biopsy_date": unique_or_error(group.biopsy_date, "biopsy_date"),
            "source_dataset": "+".join(sorted(group.source_dataset.dropna().astype(str).unique())),
            "slides_per_event": len(group),
        }
        if "stain_group" in group:
            counts = group.stain_group.value_counts()
            for stain in ("HE", "IHC", "other", "unknown"):
                row[f"{stain}_slides"] = int(counts.get(stain, 0))
            present = [stain for stain in ("HE", "IHC", "other", "unknown") if counts.get(stain, 0)]
            row["stain_combination"] = "+".join(present)
        for column in quality_numeric:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = values.mean()
            if column == "patch_count_l0":
                row["patch_count_l0_total"] = values.sum(min_count=1)
        rows.append(row)
    events = pd.DataFrame(rows)
    events["biopsy_date"] = pd.to_datetime(events.biopsy_date)
    events = events.sort_values(["case_id", "biopsy_date", "event_id"]).reset_index(drop=True)
    events["event_index"] = events.groupby("case_id").cumcount() + 1
    events["patient_event_count"] = events.groupby("case_id").event_id.transform("size")
    previous = events.groupby("case_id").biopsy_date.shift()
    events["days_since_previous"] = (events.biopsy_date - previous).dt.days
    events["biopsy_year"] = events.biopsy_date.dt.year
    events["biopsy_month"] = events.biopsy_date.dt.month
    return events


def attach_labels(event_metadata: pd.DataFrame, tasks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for task, slides in tasks.items():
        labels = slides.groupby("event_id", as_index=False).agg(
            case_id_label=("case_id", "first"), label=("label", "first"), label_nunique=("label", "nunique")
        )
        if labels.label_nunique.ne(1).any():
            raise ValueError(f"{task} has inconsistent labels within an event")
        merged = event_metadata.merge(labels.drop(columns="label_nunique"), on="event_id", validate="one_to_one")
        if merged.case_id.ne(merged.case_id_label).any():
            raise ValueError(f"{task} case_id mismatch after event merge")
        result[task] = merged.drop(columns="case_id_label")
    return result


def dataset_table1(tasks: dict[str, pd.DataFrame], task_events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in TASK_FILES:
        slides = tasks[task]
        events = task_events[task]
        patient_labels = events.groupby("case_id").label.max()
        for level, total, positives in (
            ("patient", len(patient_labels), int(patient_labels.sum())),
            ("event", len(events), int(events.label.sum())),
            ("slide", len(slides), int(slides.label.sum())),
        ):
            rows.append({
                "task": task, "analysis_level": level, "total": total, "positive": positives,
                "negative": total - positives, "positive_fraction": positives / total,
            })
    return pd.DataFrame(rows)


def event_multiplicity(task_events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for task, events in task_events.items():
        frame = events.copy()
        frame.insert(0, "task", task)
        event_prevalence = float(frame.label.mean())
        slide_prevalence = float(
            frame.loc[frame.label.eq(1), "slides_per_event"].sum() / frame.slides_per_event.sum()
        )
        frame["event_positive_fraction"] = event_prevalence
        frame["slide_positive_fraction"] = slide_prevalence
        frame["positive_exposure_ratio_slide_vs_event"] = slide_prevalence / event_prevalence
        frame["event_slide_sampling_multiplier"] = frame.slides_per_event / frame.slides_per_event.mean()
        frame["event_weight"] = 1.0
        frame["slide_weight"] = frame.slides_per_event.astype(float)
        frame["patient_balanced_weight"] = 1.0 / frame.patient_event_count
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def numeric_confounder(task: str, frame: pd.DataFrame, variable: str) -> dict[str, object] | None:
    values = pd.to_numeric(frame[variable], errors="coerce")
    positive = values[frame.label.eq(1)].dropna()
    negative = values[frame.label.eq(0)].dropna()
    if positive.empty or negative.empty:
        return None
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
    smd = (positive.mean() - negative.mean()) / pooled if pooled and np.isfinite(pooled) else np.nan
    return {
        "task": task, "variable": variable, "variable_type": "numeric", "level": "positive_vs_negative",
        "n": len(positive) + len(negative), "positive_events": len(positive),
        "positive_rate": np.nan, "overall_positive_rate": frame.label.mean(), "rate_ratio": np.nan,
        "negative_mean": negative.mean(), "positive_mean": positive.mean(),
        "negative_median": negative.median(), "positive_median": positive.median(),
        "standardized_mean_difference": smd,
    }


def confounder_audit(task_events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    categorical_candidates = ["source_dataset", "biopsy_year", "stain_combination"]
    numeric_candidates = [
        "slides_per_event", "patient_event_count", "event_index", "days_since_previous",
        "HE_slides", "IHC_slides", "other_slides", "unknown_slides",
        "patch_count_l0_total", "patch_count_l0_mean", "tissue_ratio_mean", "sharpness_score_mean",
        "tissue_luminance_mean_mean", "tissue_luminance_std_mean", "tissue_saturation_mean_mean",
        "grid_periodicity_score_mean",
    ]
    for task, frame in task_events.items():
        overall = frame.label.mean()
        for variable in categorical_candidates:
            if variable not in frame:
                continue
            values = frame[variable].fillna("missing").astype(str)
            table = frame.assign(_level=values).groupby("_level", dropna=False).label.agg(["size", "sum"])
            for level, observed in table.iterrows():
                rate = observed["sum"] / observed["size"]
                rows.append({
                    "task": task, "variable": variable, "variable_type": "categorical", "level": level,
                    "n": int(observed["size"]), "positive_events": int(observed["sum"]),
                    "positive_rate": rate, "overall_positive_rate": overall,
                    "rate_ratio": rate / overall if overall else np.nan,
                })
        for variable in numeric_candidates:
            if variable in frame and (row := numeric_confounder(task, frame, variable)) is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def patient_folds(frame: pd.DataFrame, folds: int, seed: int) -> list[tuple[int, set[str], set[str]]]:
    patients = frame.groupby("case_id", as_index=False).label.max()
    if patients.label.value_counts().min() < folds:
        raise ValueError(f"Not enough positive patients for {folds}-fold CV")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for fold, (train_index, val_index) in enumerate(splitter.split(patients.case_id, patients.label)):
        train = set(patients.iloc[train_index].case_id)
        val = set(patients.iloc[val_index].case_id)
        if train & val:
            raise RuntimeError("Patient leakage in generated audit folds")
        result.append((fold, train, val))
    return result


def split_audit(task_events: dict[str, pd.DataFrame], folds: int, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for task, frame in task_events.items():
        for seed in seeds:
            validation_patients: set[str] = set()
            for fold, train_cases, val_cases in patient_folds(frame, folds, seed):
                validation_patients.update(val_cases)
                for split, cases in (("train", train_cases), ("val", val_cases)):
                    selected = frame[frame.case_id.isin(cases)]
                    patient_labels = selected.groupby("case_id").label.max()
                    rows.append({
                        "task": task, "seed": seed, "fold": fold, "split": split,
                        "patients": len(patient_labels), "positive_patients": int(patient_labels.sum()),
                        "events": len(selected), "positive_events": int(selected.label.sum()),
                        "slides": int(selected.slides_per_event.sum()),
                        "positive_slides": int(selected.loc[selected.label.eq(1), "slides_per_event"].sum()),
                        "patient_leakage": bool(train_cases & val_cases),
                    })
            if validation_patients != set(frame.case_id):
                raise RuntimeError(f"{task}/seed{seed}: validation folds do not cover all patients")
    return pd.DataFrame(rows)


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "events": len(labels), "positive_events": int(labels.sum()), "prevalence": labels.mean(),
        "auroc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "sensitivity": sensitivity, "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
    }


def available_feature_sets(frame: pd.DataFrame) -> dict[str, tuple[list[str], list[str]]]:
    candidates = {
        "acquisition": (["biopsy_year", "biopsy_month"], ["source_dataset"]),
        "multiplicity": (["slides_per_event", "patient_event_count", "event_index", "days_since_previous"], []),
        "stain_only": (["HE_slides", "IHC_slides", "other_slides", "unknown_slides"], ["stain_combination"]),
        "quality_only": ([
            "patch_count_l0_total", "patch_count_l0_mean", "tissue_ratio_mean", "sharpness_score_mean",
            "tissue_luminance_mean_mean", "tissue_luminance_std_mean", "tissue_saturation_mean_mean",
            "grid_periodicity_score_mean",
        ], []),
    }
    available: dict[str, tuple[list[str], list[str]]] = {}
    for name, (numeric, categorical) in candidates.items():
        numeric = [column for column in numeric if column in frame and frame[column].notna().any()]
        categorical = [column for column in categorical if column in frame and frame[column].notna().any()]
        if numeric or categorical:
            available[name] = (numeric, categorical)
    all_numeric = list(dict.fromkeys(column for numeric, _ in available.values() for column in numeric))
    all_categorical = list(dict.fromkeys(column for _, categorical in available.values() for column in categorical))
    available["all_metadata"] = (all_numeric, all_categorical)
    return available


def metadata_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000)),
    ])


def metadata_only_baseline(
    task_events: dict[str, pd.DataFrame], folds: int, seeds: list[int], threshold: float
) -> pd.DataFrame:
    rows = []
    for task, frame in task_events.items():
        feature_sets = available_feature_sets(frame)
        for feature_set, (numeric, categorical) in feature_sets.items():
            seed_predictions = []
            for seed in seeds:
                oof = []
                for _, train_cases, val_cases in patient_folds(frame, folds, seed):
                    train = frame[frame.case_id.isin(train_cases)].copy()
                    val = frame[frame.case_id.isin(val_cases)].copy()
                    model = metadata_model(numeric, categorical)
                    patient_weights = 1.0 / train.groupby("case_id").event_id.transform("size")
                    patient_weights = patient_weights / patient_weights.mean()
                    model.fit(
                        train[numeric + categorical], train.label,
                        classifier__sample_weight=patient_weights.to_numpy(),
                    )
                    probability = model.predict_proba(val[numeric + categorical])[:, 1]
                    oof.append(pd.DataFrame({
                        "event_id": val.event_id, "label": val.label, "probability": probability,
                    }))
                predictions = pd.concat(oof, ignore_index=True).sort_values("event_id")
                if predictions.event_id.duplicated().any() or len(predictions) != len(frame):
                    raise RuntimeError(f"Incomplete metadata OOF predictions: {task}/{feature_set}/seed{seed}")
                rows.append({
                    "task": task, "feature_set": feature_set, "aggregation": "seed_oof", "seed": seed,
                    "folds": folds, "numeric_features": "+".join(numeric),
                    "categorical_features": "+".join(categorical),
                    **classification_metrics(predictions.label.to_numpy(), predictions.probability.to_numpy(), threshold),
                })
                predictions["seed"] = seed
                seed_predictions.append(predictions)
            combined = pd.concat(seed_predictions, ignore_index=True)
            ensemble = combined.groupby("event_id", as_index=False).agg(
                label=("label", "first"), probability=("probability", "mean"), seeds=("seed", "nunique")
            )
            if not ensemble.seeds.eq(len(seeds)).all():
                raise RuntimeError(f"Incomplete metadata seed ensemble: {task}/{feature_set}")
            rows.append({
                "task": task, "feature_set": feature_set, "aggregation": "seed_ensemble", "seed": "ensemble",
                "folds": folds, "numeric_features": "+".join(numeric),
                "categorical_features": "+".join(categorical),
                **classification_metrics(ensemble.label.to_numpy(), ensemble.probability.to_numpy(), threshold),
            })
    return pd.DataFrame(rows)


def cohort_flow(
    tasks: dict[str, pd.DataFrame], curation: pd.DataFrame | None, feature_root: Path | None
) -> pd.DataFrame:
    reference = tasks["acr_high"]
    rows: list[dict[str, object]] = []

    def add(stage: str, parent: str, frame: pd.DataFrame, definition: str) -> None:
        rows.append({
            "stage": stage, "parent_stage": parent, "slides": frame.slide_id.nunique() if "slide_id" in frame else np.nan,
            "events": frame.event_id.nunique() if "event_id" in frame else frame.event_key.nunique() if "event_key" in frame else np.nan,
            "patients": frame.case_id.nunique() if "case_id" in frame else np.nan,
            "definition": definition,
        })

    gold_parent = ""
    if curation is not None:
        add("wsi_inventory", "", curation, "All slides in the curated WSI inventory")
        gold_parent = "wsi_inventory"
        if "gold_pathology_id_match" in curation:
            exact = curation[as_bool(curation.gold_pathology_id_match)]
            add("exact_pathology_match", "wsi_inventory", exact, "Filename pathology ID exactly matched to the label workbook")
            gold_parent = "exact_pathology_match"
            if "include_quality_usable" in exact:
                usable = exact[as_bool(exact.include_quality_usable)]
                add("quality_usable_exact_match", "exact_pathology_match", usable, "Exact matches retained by final quality curation")
                gold_parent = "quality_usable_exact_match"
    add("gold_analysis_cohort", gold_parent, reference, "Gold task CSV after label matching and exclusions")
    if feature_root is not None:
        gold_ids = set(reference.slide_id)
        for scale, directory in SCALES.items():
            pt_root = feature_root / directory / "pt_files"
            if not pt_root.is_dir():
                continue
            available = reference[reference.slide_id.isin({path.stem for path in pt_root.glob("*.pt")} & gold_ids)]
            add(f"feature_available_{scale}", "gold_analysis_cohort", available, f"Gold slides with {scale} UNI v2 feature bags")
    return pd.DataFrame(rows)


def main() -> int:
    args = arguments()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must not contain duplicates")
    tasks = load_tasks(args.label_dir)
    curation = load_curation(args.curation_manifest)
    slides = slide_metadata(tasks["acr_high"], curation)
    event_metadata = build_event_metadata(slides)
    task_events = attach_labels(event_metadata, tasks)
    outputs = {
        "cohort_flow.csv": cohort_flow(tasks, curation, args.feature_root),
        "dataset_table1.csv": dataset_table1(tasks, task_events),
        "event_multiplicity_audit.csv": event_multiplicity(task_events),
        "confounder_audit.csv": confounder_audit(task_events),
        "split_audit.csv": split_audit(task_events, args.folds, args.seeds),
        "metadata_only_baseline.csv": metadata_only_baseline(task_events, args.folds, args.seeds, args.threshold),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)
        print(f"[OK] {filename}: {len(frame)} rows")
    print(f"[DONE] SMC gold dataset audit -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
