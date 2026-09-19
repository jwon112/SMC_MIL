#!/usr/bin/env python3
"""Evaluate repeated-seed ensembles and fixed multiscale probability fusion."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


EXPERIMENT_PATTERN = re.compile(
    r"^smc_(?P<task>acr_high_grade|amr_positive|significant_rejection)_"
    r"(?P<level>l[0-3])_(?P<mpp>0p25|0p50|1p00|2p00)mpp_"
    r"(?P<scale>40x|20x|10x|5x)_uni2_clamsb_"
    r"cv(?P<folds>[35])val_s(?P<seed>[0-9]+)$"
)
TASK_NAMES = {
    "acr_high_grade": "acr_high",
    "amr_positive": "amr_positive",
    "significant_rejection": "significant_rejection",
}
SCALE_ORDER = ("40x", "20x", "10x", "5x")
DEFAULT_FUSIONS = ("40x+20x", "40x+5x", "40x+20x+10x+5x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("results/gse290577_external/gold5_repeated"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smc_prediction_ensembles"),
    )
    parser.add_argument("--folds", type=int, choices=(3, 5), default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=(1, 11, 21, 31, 41))
    parser.add_argument(
        "--fusion",
        action="append",
        help="Plus-separated scales, e.g. 40x+20x. Repeatable; defaults to three prespecified fusions.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def positive_probability(value: object) -> float:
    probabilities = np.asarray(value, dtype=float).reshape(-1)
    if probabilities.size != 2:
        raise ValueError(f"Expected two class probabilities, got {probabilities.shape}")
    return float(probabilities[1])


def parse_experiment(name: str, folds: int, seeds: set[int]) -> dict[str, object] | None:
    match = EXPERIMENT_PATTERN.match(name)
    if match is None:
        return None
    fields = match.groupdict()
    seed = int(fields["seed"])
    if int(fields["folds"]) != folds or seed not in seeds:
        return None
    return {
        "experiment": name,
        "task": TASK_NAMES[fields["task"]],
        "scale": fields["scale"],
        "seed": seed,
    }


def read_internal(results_root: Path, folds: int, seeds: set[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_dir in sorted(results_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        info = parse_experiment(experiment_dir.name, folds, seeds)
        if info is None:
            continue
        result_paths = sorted(experiment_dir.glob("split_*_results.pkl"))
        if len(result_paths) != folds:
            raise RuntimeError(
                f"Expected {folds} split result files in {experiment_dir}, found {len(result_paths)}"
            )
        for result_path in result_paths:
            fold = int(result_path.stem.split("_")[1])
            with result_path.open("rb") as handle:
                result = pickle.load(handle)
            for slide_id, item in result.items():
                rows.append({
                    **info,
                    "cohort": "internal_cv",
                    "unit_id": str(slide_id),
                    "group_id": str(slide_id),
                    "fold": fold,
                    "label": int(item["label"]),
                    "probability": positive_probability(item["prob"]),
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No repeated-CV prediction files found under {results_root}")
    duplicate_columns = ["task", "scale", "seed", "unit_id"]
    if frame.duplicated(duplicate_columns).any():
        raise RuntimeError("Duplicate internal OOF prediction for a task/scale/seed/slide")
    return frame


def read_external(external_root: Path, folds: int, seeds: set[int]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not external_root.is_dir():
        return pd.DataFrame()
    seen: set[tuple[str, str]] = set()
    for prediction_path in sorted(external_root.rglob("*_predictions.csv")):
        experiment = prediction_path.parent.name
        info = parse_experiment(experiment, folds, seeds)
        if info is None:
            continue
        cohort = prediction_path.name.removesuffix("_predictions.csv")
        key = (experiment, cohort)
        if key in seen:
            raise RuntimeError(f"Duplicate external prediction file for {experiment}/{cohort}")
        seen.add(key)
        frame = pd.read_csv(prediction_path, dtype={"slide_id": str})
        required = {"slide_id", "label", "probability"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {prediction_path}")
        group = None
        for column in ("patient_id", "case_id", "biopsy_id"):
            if column in frame and frame[column].notna().any():
                group = frame[column].fillna(frame["slide_id"]).astype(str)
                break
        if group is None:
            group = frame["slide_id"].astype(str)
        rows.append(pd.DataFrame({
            **{key: value for key, value in info.items()},
            "cohort": cohort,
            "unit_id": frame["slide_id"].astype(str),
            "group_id": group,
            "fold": -1,
            "label": frame["label"].astype(int),
            "probability": frame["probability"].astype(float),
        }))
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    duplicate_columns = ["task", "scale", "seed", "cohort", "unit_id"]
    if result.duplicated(duplicate_columns).any():
        raise RuntimeError("Duplicate external prediction for a task/scale/seed/cohort/slide")
    return result


def validate_fusions(raw_fusions: list[str] | None) -> list[tuple[str, ...]]:
    values = raw_fusions or list(DEFAULT_FUSIONS)
    parsed: list[tuple[str, ...]] = []
    for value in values:
        scales = tuple(part.strip() for part in value.split("+") if part.strip())
        if len(scales) < 2 or len(set(scales)) != len(scales):
            raise ValueError(f"Fusion must contain at least two unique scales: {value}")
        unknown = set(scales).difference(SCALE_ORDER)
        if unknown:
            raise ValueError(f"Unknown scales in fusion {value}: {sorted(unknown)}")
        canonical = tuple(scale for scale in SCALE_ORDER if scale in scales)
        if canonical not in parsed:
            parsed.append(canonical)
    return parsed


def assert_complete(frame: pd.DataFrame, expected_seeds: set[int], fusions: list[tuple[str, ...]]) -> None:
    for (task, cohort), group in frame.groupby(["task", "cohort"], sort=False):
        units = set(group["unit_id"])
        available = set(group["scale"].unique())
        for scale in SCALE_ORDER:
            if scale not in available:
                continue
            scale_group = group[group["scale"].eq(scale)]
            found_seeds = set(scale_group["seed"].unique())
            if found_seeds != expected_seeds:
                raise RuntimeError(
                    f"Incomplete seeds for {task}/{cohort}/{scale}: "
                    f"expected={sorted(expected_seeds)}, found={sorted(found_seeds)}"
                )
            for seed, seed_group in scale_group.groupby("seed"):
                if set(seed_group["unit_id"]) != units:
                    raise RuntimeError(f"Unit mismatch for {task}/{cohort}/{scale}/seed{seed}")
        label_counts = group.groupby("unit_id")["label"].nunique()
        if (label_counts != 1).any():
            raise RuntimeError(f"Label disagreement across predictions for {task}/{cohort}")


def assert_inventory(frame: pd.DataFrame, scope: str, expected_seeds: set[int]) -> None:
    tasks = tuple(TASK_NAMES.values())
    if scope == "internal_oof":
        expected = {
            (task, "internal_cv", scale, seed)
            for task in tasks for scale in SCALE_ORDER for seed in expected_seeds
        }
    else:
        expected = {
            (task, cohort, scale, seed)
            for task in tasks
            for cohort in ("wsi_he", "wsi_ihc")
            for scale in SCALE_ORDER
            for seed in expected_seeds
        }
        expected.update({
            (task, "core", "40x", seed)
            for task in tasks for seed in expected_seeds
        })
    actual = set(
        frame[["task", "cohort", "scale", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    missing = sorted(expected.difference(actual))
    if missing:
        preview = ", ".join("/".join(map(str, item)) for item in missing[:10])
        raise RuntimeError(f"Missing {len(missing)} required {scope} combinations: {preview}")


def aggregate_predictions(
    source: pd.DataFrame,
    group_columns: list[str],
    method: str,
    scales: tuple[str, ...],
    seeds_used: int,
) -> pd.DataFrame:
    metadata = source.groupby(group_columns, as_index=False).agg(
        label=("label", "first"),
        label_versions=("label", "nunique"),
        group_id=("group_id", "first"),
        probability=("probability", "mean"),
        component_predictions=("probability", "size"),
    )
    if (metadata["label_versions"] != 1).any():
        raise RuntimeError(f"Label disagreement while constructing {method}")
    metadata = metadata.drop(columns="label_versions")
    metadata["method"] = method
    metadata["scales"] = "+".join(scales)
    metadata["scale_count"] = len(scales)
    metadata["seeds_used"] = seeds_used
    return metadata


def build_strategies(
    frame: pd.DataFrame,
    seeds: set[int],
    fusions: list[tuple[str, ...]],
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    base_groups = ["task", "cohort", "unit_id"]

    for (_, _), cohort_frame in frame.groupby(["task", "cohort"], sort=False):
        available = set(cohort_frame["scale"].unique())
        for scale in SCALE_ORDER:
            if scale not in available:
                continue
            subset = cohort_frame[cohort_frame["scale"].eq(scale)]
            outputs.append(aggregate_predictions(
                subset,
                ["task", "cohort", "seed", "unit_id"],
                "single_seed",
                (scale,),
                1,
            ))
            outputs.append(aggregate_predictions(
                subset, base_groups, "seed_ensemble", (scale,), len(seeds)
            ))

        for fusion in fusions:
            if not set(fusion).issubset(available):
                continue
            subset = cohort_frame[cohort_frame["scale"].isin(fusion)]
            per_seed = aggregate_predictions(
                subset,
                ["task", "cohort", "seed", "unit_id"],
                "scale_fusion_per_seed",
                fusion,
                1,
            )
            outputs.append(per_seed)
            outputs.append(aggregate_predictions(
                subset, base_groups, "seed_scale_ensemble", fusion, len(seeds)
            ))

    return pd.concat(outputs, ignore_index=True)


def calculate_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float | int]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "n": len(frame),
        "positive_n": int(labels.sum()),
        "negative_n": int((labels == 0).sum()),
        "auroc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(labels, probabilities)) if labels.sum() else np.nan,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": threshold,
    }


def summarize(predictions: pd.DataFrame, scope: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["task", "cohort", "method", "scales", "scale_count", "seeds_used"]
    if "seed" in predictions:
        keys.append("seed")
    for values, group in predictions.groupby(keys, dropna=False, sort=False):
        info = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        rows.append({"evaluation_scope": scope, **info, **calculate_metrics(group, threshold)})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")
    seeds = set(args.seeds)
    if len(seeds) != len(args.seeds):
        raise ValueError("--seeds contains duplicate values")
    fusions = validate_fusions(args.fusion)

    internal = read_internal(args.results_root, args.folds, seeds)
    external = read_external(args.external_root, args.folds, seeds)
    sources = [("internal_oof", internal)]
    if not external.empty:
        sources.append(("external_test", external))
    else:
        print(f"[WARN] no external prediction files found under {args.external_root}")

    prediction_outputs = []
    metric_outputs = []
    for scope, source in sources:
        assert_inventory(source, scope, seeds)
        assert_complete(source, seeds, fusions)
        predictions = build_strategies(source, seeds, fusions)
        predictions.insert(0, "evaluation_scope", scope)
        prediction_outputs.append(predictions)
        metric_outputs.append(summarize(predictions, scope, args.threshold))

    all_predictions = pd.concat(prediction_outputs, ignore_index=True)
    all_metrics = pd.concat(metric_outputs, ignore_index=True)
    sort_columns = ["evaluation_scope", "task", "cohort", "method", "scales"]
    if "seed" in all_metrics:
        sort_columns.append("seed")
    all_predictions = all_predictions.sort_values(sort_columns + ["unit_id"])
    all_metrics = all_metrics.sort_values(sort_columns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "ensemble_predictions.csv"
    metric_path = args.output_dir / "ensemble_metrics.csv"
    all_predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    all_metrics.to_csv(metric_path, index=False, encoding="utf-8-sig")

    seed_summary = all_metrics[all_metrics["method"].eq("seed_ensemble")]
    fusion_summary = all_metrics[all_metrics["method"].eq("seed_scale_ensemble")]
    seed_summary.to_csv(args.output_dir / "seed_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    fusion_summary.to_csv(args.output_dir / "multiscale_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
    reference_40x = seed_summary[seed_summary["scales"].eq("40x")]
    if not fusion_summary.empty and not reference_40x.empty:
        metric_columns = ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy")
        reference_columns = ["evaluation_scope", "task", "cohort"] + list(metric_columns)
        fusion_gain = fusion_summary.merge(
            reference_40x[reference_columns],
            on=["evaluation_scope", "task", "cohort"],
            how="left",
            suffixes=("", "_40x_seed_ensemble"),
            validate="many_to_one",
        )
        for metric in metric_columns:
            fusion_gain[f"delta_{metric}_vs_40x"] = (
                fusion_gain[metric] - fusion_gain[f"{metric}_40x_seed_ensemble"]
            )
        fusion_gain.to_csv(
            args.output_dir / "multiscale_gain_vs_40x.csv", index=False, encoding="utf-8-sig"
        )

    per_seed = all_metrics[all_metrics["method"].eq("scale_fusion_per_seed")]
    if not per_seed.empty:
        metric_columns = ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy")
        single_40x = all_metrics[
            all_metrics["method"].eq("single_seed") & all_metrics["scales"].eq("40x")
        ]
        paired = per_seed.merge(
            single_40x[["evaluation_scope", "task", "cohort", "seed", *metric_columns]],
            on=["evaluation_scope", "task", "cohort", "seed"],
            how="left",
            suffixes=("", "_40x"),
            validate="many_to_one",
        )
        for metric in metric_columns:
            paired[f"delta_{metric}_vs_40x"] = paired[metric] - paired[f"{metric}_40x"]
        paired.to_csv(
            args.output_dir / "multiscale_per_seed_gain_vs_40x.csv", index=False, encoding="utf-8-sig"
        )
        repeated = per_seed.groupby(
            ["evaluation_scope", "task", "cohort", "scales"], as_index=False
        ).agg(
            seeds_completed=("seed", "nunique"),
            n=("n", "first"),
            positive_n=("positive_n", "first"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in metric_columns
                for stat in ("mean", "std", "min", "max")
            },
        )
        repeated.to_csv(
            args.output_dir / "multiscale_per_seed_summary.csv", index=False, encoding="utf-8-sig"
        )

    single_seed = all_metrics[all_metrics["method"].eq("single_seed")]
    if not single_seed.empty:
        metric_columns = ("auroc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy", "accuracy")
        baseline = single_seed.groupby(
            ["evaluation_scope", "task", "cohort", "scales"], as_index=False
        ).agg(
            seeds_completed=("seed", "nunique"),
            **{
                f"single_seed_{metric}_{stat}": (metric, stat)
                for metric in metric_columns
                for stat in ("mean", "std")
            },
        )
        ensemble_columns = ["evaluation_scope", "task", "cohort", "scales"] + list(metric_columns)
        comparison = baseline.merge(
            seed_summary[ensemble_columns],
            on=["evaluation_scope", "task", "cohort", "scales"],
            how="left",
            validate="one_to_one",
        )
        for metric in metric_columns:
            comparison.rename(columns={metric: f"seed_ensemble_{metric}"}, inplace=True)
            comparison[f"delta_{metric}_vs_seed_mean"] = (
                comparison[f"seed_ensemble_{metric}"] - comparison[f"single_seed_{metric}_mean"]
            )
        comparison.to_csv(
            args.output_dir / "seed_ensemble_gain.csv", index=False, encoding="utf-8-sig"
        )

    print(f"[OK] internal raw predictions: {len(internal):,}")
    print(f"[OK] external raw predictions: {len(external):,}")
    print(f"[OK] ensemble predictions: {prediction_path}")
    print(f"[OK] ensemble metrics: {metric_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
