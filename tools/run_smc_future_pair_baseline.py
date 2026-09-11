#!/usr/bin/env python3
"""Run a patient-grouped future-biopsy baseline with frozen UNI patch features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


VARIANTS = ("time_only", "image_only", "image_plus_time")


def normalize(value: object) -> str:
    import re

    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--patient-folds", type=Path, required=True)
    parser.add_argument("--gold-csv", type=Path, required=True)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Scale feature directory containing pt_files/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.001, 0.01, 0.1, 1.0))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def tensor_from_pt(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        for key in ("features", "feature", "embeddings"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor in {path}, got {type(value)}")
    if value.shape[0] == 0:
        raise ValueError(f"Empty feature tensor: {path}")
    return value.float()


def build_event_embeddings(
    pairs: pd.DataFrame,
    gold_csv: Path,
    feature_dir: Path,
    cache_path: Path,
    force: bool,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    required_keys = sorted(pairs.source_event_key.astype(str).unique())
    if cache_path.is_file() and not force:
        cached = np.load(cache_path, allow_pickle=False)
        keys = cached["event_keys"].astype(str).tolist()
        embeddings = cached["embeddings"].astype(np.float32)
        if keys == required_keys:
            audit = pd.DataFrame(
                {
                    "source_event_key": keys,
                    "slides": cached["slide_counts"].astype(int),
                    "patches": cached["patch_counts"].astype(int),
                }
            )
            return dict(zip(keys, embeddings)), audit

    bags = pd.read_csv(gold_csv, dtype={"slide_id": str})
    required = {"event_id", "slide_id"}
    missing = required.difference(bags.columns)
    if missing:
        raise ValueError(f"Gold CSV is missing columns: {sorted(missing)}")
    bags["event_key"] = bags.event_id.map(normalize)
    event_slides = bags.groupby("event_key").slide_id.apply(list).to_dict()
    pt_dir = feature_dir / "pt_files"
    if not pt_dir.is_dir():
        raise FileNotFoundError(pt_dir)

    embeddings: list[np.ndarray] = []
    slide_counts: list[int] = []
    patch_counts: list[int] = []
    expected_dim: int | None = None
    for index, event_key in enumerate(required_keys, start=1):
        slides = event_slides.get(event_key)
        if not slides:
            raise KeyError(f"No slides found for source event {event_key}")
        slide_means: list[np.ndarray] = []
        event_patches = 0
        for slide_id in slides:
            path = pt_dir / f"{slide_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            tensor = tensor_from_pt(path)
            if expected_dim is None:
                expected_dim = int(tensor.shape[1])
            elif tensor.shape[1] != expected_dim:
                raise ValueError(f"Feature dimension mismatch in {path}: {tensor.shape[1]}")
            slide_means.append(tensor.mean(dim=0).numpy())
            event_patches += int(tensor.shape[0])
        # Equal slide weighting prevents a large tissue area from defining the event alone.
        embeddings.append(np.mean(slide_means, axis=0, dtype=np.float64).astype(np.float32))
        slide_counts.append(len(slides))
        patch_counts.append(event_patches)
        if index % 25 == 0 or index == len(required_keys):
            print(f"[EMBED] {index}/{len(required_keys)} source events", flush=True)

    matrix = np.stack(embeddings)
    np.savez_compressed(
        cache_path,
        event_keys=np.asarray(required_keys),
        embeddings=matrix,
        slide_counts=np.asarray(slide_counts),
        patch_counts=np.asarray(patch_counts),
    )
    audit = pd.DataFrame(
        {
            "source_event_key": required_keys,
            "slides": slide_counts,
            "patches": patch_counts,
        }
    )
    return dict(zip(required_keys, matrix)), audit


def patient_equal_weights(groups: np.ndarray) -> np.ndarray:
    counts = pd.Series(groups).value_counts()
    return np.asarray([1.0 / counts[group] for group in groups], dtype=float)


def balanced_training_weights(groups: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = patient_equal_weights(groups)
    positive_mass = float(weights[labels == 1].sum())
    negative_mass = float(weights[labels == 0].sum())
    if positive_mass <= 0 or negative_mass <= 0:
        raise ValueError("A training partition contains only one pair-label class")
    target_mass = float(pd.Series(groups).nunique()) / 2.0
    weights[labels == 1] *= target_mass / positive_mass
    weights[labels == 0] *= target_mass / negative_mass
    return weights


def weighted_metrics(
    frame: pd.DataFrame,
    threshold: float | None,
    probability_column: str = "probability",
) -> dict[str, float | int]:
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    groups = frame.case_id.astype(str).to_numpy()
    weights = patient_equal_weights(groups)
    predictions = (
        frame.prediction.to_numpy(dtype=int)
        if threshold is None
        else probabilities >= threshold
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    weighted_tn, weighted_fp, weighted_fn, weighted_tp = confusion_matrix(
        labels, predictions, labels=[0, 1], sample_weight=weights
    ).ravel()
    sensitivity = weighted_tp / (weighted_tp + weighted_fn) if weighted_tp + weighted_fn else np.nan
    specificity = weighted_tn / (weighted_tn + weighted_fp) if weighted_tn + weighted_fp else np.nan
    return {
        "pairs": len(frame),
        "patients": int(frame.case_id.nunique()),
        "positive_pairs": int(labels.sum()),
        "positive_patients": int(frame.loc[frame.label.eq(1), "case_id"].nunique()),
        "raw_prevalence": float(labels.mean()),
        "patient_weighted_prevalence": float(np.average(labels, weights=weights)),
        "auroc": float(roc_auc_score(labels, probabilities, sample_weight=weights)),
        "pr_auc": float(average_precision_score(labels, probabilities, sample_weight=weights)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "threshold": float(threshold) if threshold is not None else np.nan,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def choose_threshold(frame: pd.DataFrame) -> float:
    labels = frame.label.to_numpy(dtype=int)
    probabilities = frame.probability.to_numpy(dtype=float)
    weights = patient_equal_weights(frame.case_id.astype(str).to_numpy())
    candidates = np.unique(
        np.r_[np.nextafter(probabilities.min(), -np.inf), probabilities, np.nextafter(probabilities.max(), np.inf)]
    )
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    for threshold in candidates:
        prediction = probabilities >= threshold
        score = float(f1_score(labels, prediction, sample_weight=weights, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(
            labels, prediction, labels=[0, 1], sample_weight=weights
        ).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        key = (score, min(sensitivity, specificity), -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_x: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_x)
    scaled_test = scaler.transform(test_x)
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=3000,
        random_state=seed,
    )
    model.fit(
        scaled_train,
        train_y,
        sample_weight=balanced_training_weights(train_groups, train_y),
    )
    return model.predict_proba(scaled_test)[:, 1]


def variant_matrix(image: np.ndarray, log_days: np.ndarray, variant: str) -> np.ndarray:
    if variant == "time_only":
        return log_days[:, None]
    if variant == "image_only":
        return image
    if variant == "image_plus_time":
        return np.column_stack([image, log_days])
    raise KeyError(variant)


def select_c_and_threshold(
    x: np.ndarray,
    pairs: pd.DataFrame,
    train_indices: np.ndarray,
    c_values: list[float],
    inner_folds: int,
    seed: int,
) -> tuple[float, float, list[dict[str, float]]]:
    train = pairs.iloc[train_indices]
    patient_labels = train.groupby("case_id", as_index=False).label.max()
    class_counts = patient_labels.label.value_counts()
    n_splits = min(inner_folds, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Not enough positive patients for inner patient CV")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    candidates: list[dict[str, float]] = []

    for c_value in c_values:
        probabilities = np.full(len(train_indices), np.nan)
        for inner_train, inner_test in splitter.split(patient_labels.case_id, patient_labels.label):
            fit_cases = set(patient_labels.iloc[inner_train].case_id)
            test_cases = set(patient_labels.iloc[inner_test].case_id)
            local_fit = np.flatnonzero(train.case_id.isin(fit_cases).to_numpy())
            local_test = np.flatnonzero(train.case_id.isin(test_cases).to_numpy())
            probabilities[local_test] = fit_predict(
                x[train_indices[local_fit]],
                pairs.iloc[train_indices[local_fit]].label.to_numpy(int),
                pairs.iloc[train_indices[local_fit]].case_id.astype(str).to_numpy(),
                x[train_indices[local_test]],
                c_value,
                seed,
            )
        if np.isnan(probabilities).any():
            raise RuntimeError("Incomplete inner OOF probabilities")
        inner = train[["case_id", "label"]].copy()
        inner["probability"] = probabilities
        threshold = choose_threshold(inner)
        result = weighted_metrics(inner, threshold)
        candidates.append(
            {
                "C": float(c_value),
                "threshold": threshold,
                "inner_auroc": float(result["auroc"]),
                "inner_pr_auc": float(result["pr_auc"]),
                "inner_f1": float(
                    f1_score(
                        inner.label,
                        inner.probability.ge(threshold),
                        sample_weight=patient_equal_weights(inner.case_id.astype(str).to_numpy()),
                        zero_division=0,
                    )
                ),
            }
        )
    selected = max(candidates, key=lambda row: (row["inner_pr_auc"], row["inner_auroc"], -row["C"]))
    return selected["C"], selected["threshold"], candidates


def cluster_bootstrap(
    frame: pd.DataFrame,
    threshold: float | None,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    patient_ids = frame.case_id.astype(str).unique()
    values = {name: [] for name in ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy")}
    grouped = {case_id: part for case_id, part in frame.groupby(frame.case_id.astype(str))}
    for _ in range(repetitions):
        sampled_parts = []
        for draw, case_id in enumerate(rng.choice(patient_ids, len(patient_ids), replace=True)):
            part = grouped[str(case_id)].copy()
            part["case_id"] = f"bootstrap_{draw}"
            sampled_parts.append(part)
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if sampled.label.nunique() < 2:
            continue
        result = weighted_metrics(sampled, threshold)
        for name in values:
            value = float(result[name])
            if np.isfinite(value):
                values[name].append(value)
    intervals: dict[str, float] = {}
    for name, samples in values.items():
        if samples:
            intervals[f"{name}_ci_low"] = float(np.quantile(samples, 0.025))
            intervals[f"{name}_ci_high"] = float(np.quantile(samples, 0.975))
    return intervals


def summarize_predictions(
    predictions: pd.DataFrame,
    threshold_by_fold: dict[int, float],
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    predictions = predictions.copy()
    predictions["selected_threshold"] = predictions.fold.map(threshold_by_fold)
    predictions["prediction"] = [
        int(probability >= threshold_by_fold[int(fold)])
        for probability, fold in zip(predictions.probability, predictions.fold)
    ]
    for fold, part in predictions.groupby("fold"):
        fold_rows.append({"fold": int(fold), **weighted_metrics(part, None)})

    # Ranking metrics use OOF probabilities; operating metrics preserve each fold's
    # threshold selected without seeing that outer fold.
    pooled = weighted_metrics(predictions, None)
    pooled.update(cluster_bootstrap(predictions, None, bootstrap_repetitions, seed))

    horizon = pd.cut(
        predictions.delta_days,
        bins=[0, 30, 90, 180, np.inf],
        labels=["1-30d", "31-90d", "91-180d", ">180d"],
    )
    horizon_rows = []
    for name, part in predictions.assign(horizon=horizon).groupby("horizon", observed=True):
        if part.label.nunique() < 2:
            continue
        horizon_rows.append({"horizon": str(name), **weighted_metrics(part, None)})

    return pd.DataFrame(fold_rows), pd.DataFrame([pooled]), pd.DataFrame(horizon_rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pair_manifest, dtype={"case_id": str, "source_event_key": str})
    assignments = pd.read_csv(args.patient_folds, dtype={"case_id": str})
    required = {"pair_id", "case_id", "source_event_key", "target_event_key", "delta_days", "label"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Pair manifest is missing columns: {sorted(missing)}")
    if assignments.held_out_fold.nunique() != args.folds:
        raise ValueError("Patient fold file does not match --folds")
    fold_lookup = assignments.set_index("case_id").held_out_fold
    pairs["fold"] = pairs.case_id.map(fold_lookup)
    if pairs.fold.isna().any():
        raise ValueError("At least one pair patient has no fold assignment")
    pairs["fold"] = pairs.fold.astype(int)

    embedding_map, embedding_audit = build_event_embeddings(
        pairs,
        args.gold_csv,
        args.feature_dir,
        args.output_dir / "event_embeddings.npz",
        args.force_embeddings,
    )
    image = np.stack([embedding_map[key] for key in pairs.source_event_key.astype(str)])
    log_days = np.log1p(pairs.delta_days.to_numpy(float))
    embedding_audit.to_csv(args.output_dir / "event_embedding_audit.csv", index=False)

    all_predictions = []
    all_target_predictions = []
    all_fold_metrics = []
    all_pooled_metrics = []
    all_horizon_metrics = []
    all_selections = []
    for variant_index, variant in enumerate(VARIANTS):
        print(f"[RUN] {variant}", flush=True)
        x = variant_matrix(image, log_days, variant)
        variant_predictions = []
        threshold_by_fold: dict[int, float] = {}
        for fold in range(args.folds):
            train_indices = np.flatnonzero(pairs.fold.ne(fold).to_numpy())
            test_indices = np.flatnonzero(pairs.fold.eq(fold).to_numpy())
            train_cases = set(pairs.iloc[train_indices].case_id)
            test_cases = set(pairs.iloc[test_indices].case_id)
            if train_cases & test_cases:
                raise RuntimeError("Patient leakage between outer folds")
            c_value, threshold, candidates = select_c_and_threshold(
                x,
                pairs,
                train_indices,
                list(args.c_values),
                args.inner_folds,
                args.seed + variant_index * 100 + fold,
            )
            probability = fit_predict(
                x[train_indices],
                pairs.iloc[train_indices].label.to_numpy(int),
                pairs.iloc[train_indices].case_id.astype(str).to_numpy(),
                x[test_indices],
                c_value,
                args.seed + variant_index * 100 + fold,
            )
            part = pairs.iloc[test_indices].copy()
            part["fold"] = fold
            part["probability"] = probability
            variant_predictions.append(part)
            threshold_by_fold[fold] = threshold
            for candidate in candidates:
                all_selections.append(
                    {
                        "model_variant": variant,
                        "fold": fold,
                        "selected": int(candidate["C"] == c_value),
                        **candidate,
                    }
                )
            print(f"[FOLD {fold}] C={c_value:g} threshold={threshold:.4f}", flush=True)

        predictions = pd.concat(variant_predictions, ignore_index=True)
        if len(predictions) != len(pairs) or predictions.pair_id.nunique() != len(pairs):
            raise RuntimeError("Outer OOF predictions do not cover each pair exactly once")
        fold_metrics, pooled_metrics, horizon_metrics = summarize_predictions(
            predictions,
            threshold_by_fold,
            args.bootstrap,
            args.seed + variant_index,
        )
        target_predictions = predictions.groupby(
            ["case_id", "target_event_key", "target_event_id", "target_biopsy_date", "label", "fold"],
            as_index=False,
        ).agg(
            probability=("probability", "mean"),
            source_pairs=("pair_id", "size"),
            minimum_delta_days=("delta_days", "min"),
            maximum_delta_days=("delta_days", "max"),
        )
        target_predictions["selected_threshold"] = target_predictions.fold.map(threshold_by_fold)
        target_predictions["prediction"] = target_predictions.probability.ge(
            target_predictions.selected_threshold
        ).astype(int)
        target_pooled = weighted_metrics(target_predictions, None)
        target_pooled["pairs"] = np.nan
        target_pooled["positive_pairs"] = np.nan
        target_pooled.update(
            cluster_bootstrap(
                target_predictions,
                None,
                args.bootstrap,
                args.seed + variant_index + 1000,
            )
        )
        target_pooled["target_events"] = len(target_predictions)
        target_pooled["positive_target_events"] = int(target_predictions.label.sum())

        predictions.insert(0, "model_variant", variant)
        target_predictions.insert(0, "model_variant", variant)
        fold_metrics.insert(0, "model_variant", variant)
        pooled_metrics.insert(0, "model_variant", variant)
        pooled_metrics.insert(1, "evaluation_unit", "pair")
        target_pooled_frame = pd.DataFrame([target_pooled])
        target_pooled_frame.insert(0, "model_variant", variant)
        target_pooled_frame.insert(1, "evaluation_unit", "target_event")
        horizon_metrics.insert(0, "model_variant", variant)
        all_predictions.append(predictions)
        all_target_predictions.append(target_predictions)
        all_fold_metrics.append(fold_metrics)
        all_pooled_metrics.extend([pooled_metrics, target_pooled_frame])
        all_horizon_metrics.append(horizon_metrics)

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    prediction_frame.to_csv(args.output_dir / "pair_oof_predictions.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_target_predictions, ignore_index=True).to_csv(
        args.output_dir / "target_event_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(
        args.output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pooled_frame = pd.concat(all_pooled_metrics, ignore_index=True)
    pooled_frame.to_csv(args.output_dir / "pooled_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_horizon_metrics, ignore_index=True).to_csv(
        args.output_dir / "horizon_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_selections).to_csv(
        args.output_dir / "inner_model_selection.csv", index=False, encoding="utf-8-sig"
    )
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pair_manifest": str(args.pair_manifest),
                "patient_folds": str(args.patient_folds),
                "gold_csv": str(args.gold_csv),
                "feature_dir": str(args.feature_dir),
                "folds": args.folds,
                "inner_folds": args.inner_folds,
                "c_values": args.c_values,
                "seed": args.seed,
                "event_pooling": "mean patches per slide, then equal-weight mean across slides",
                "training_weighting": "patient-total-one, then equal pair-label mass",
            },
            handle,
            indent=2,
        )
    print("\n" + pooled_frame.to_string(index=False))
    print(f"[OK] Future-pair baseline: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
