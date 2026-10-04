#!/usr/bin/env python3
"""Run one exploratory CLAM fold with draw-level UNI2 feature-view selection.

Copy alongside feature_view_bank.py and clam_view_training.py to SMC_MIL.
The best_val_v2 protocol uses the existing CLAM model with an explicit training
loop; legacy200 calls the repository's original loop without modifying it.
Validation is always original-only; the reported fold is NOT an independent
final assessment after choosing a policy on these same folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from feature_view_bank import FeatureViewTrainSplit, validate_feature_bank


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path,
                        help="Fold-specific output root of augment_uni2_wsi.py")
    parser.add_argument("--policy", choices=("original", "geometry", "geometry_he_scale_light"),
                        required=True)
    parser.add_argument("--total-views", type=int, choices=(1, 2, 3, 5), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--view-seed", type=int, default=20260918)
    parser.add_argument("--sampler", choices=("weighted", "uniform"), default="weighted")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reg", type=float, default=1e-5)
    parser.add_argument("--drop-out", type=float, default=0.25)
    parser.add_argument("--model-size", choices=("small", "big"), default="small")
    parser.add_argument("--bag-weight", type=float, default=0.7)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--no-inst-cluster", action="store_true")
    parser.add_argument("--early-stopping", action="store_true",
                        help="Exploratory only: may give unequal training budgets across conditions.")
    parser.add_argument("--training-protocol", choices=("legacy200", "best_val_v2"), default="legacy200")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--early-stop-min-epoch", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--technical-smoke", action="store_true",
                        help="Use two full bags per class in train and val; metrics are only for integration QA.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--deep-feature-check", action="store_true",
                        help="Load every original/augmented train tensor and val tensor before training.")
    args = parser.parse_args()
    if args.total_views == 1 and (args.policy != "original" or args.bank_root is not None):
        parser.error("1-view baseline requires --policy original and no --bank-root")
    if args.total_views > 1 and (args.policy == "original" or args.bank_root is None):
        parser.error("Augmented conditions require --bank-root and geometry or H/E-scale policy")
    if args.max_epochs < 1 or args.fold < 0 or args.B < 1:
        parser.error("Epochs and B must be positive and fold non-negative")
    if args.split_csv.name != f"splits_{args.fold}.csv":
        parser.error("--split-csv filename must match --fold")
    if args.bank_root is not None and args.bank_root.name != f"fold_{args.fold}":
        parser.error("--bank-root must end in the selected fold_N directory")
    if args.technical_smoke and not args.preflight_only and args.max_epochs != 1:
        parser.error("--technical-smoke training requires --max-epochs 1")
    if (args.lr <= 0 or args.reg < 0 or not 0 <= args.drop_out < 1
            or not 0 <= args.bag_weight <= 1 or args.grad_clip <= 0
            or not 0 < args.lr_factor < 1 or not 0 <= args.min_lr <= args.lr
            or args.lr_patience < 0 or args.early_stop_patience < 1
            or args.early_stop_min_epoch < 1 or args.early_stop_min_delta < 0):
        parser.error("Invalid optimization or stopping parameters")
    if args.training_protocol == "best_val_v2":
        args.early_stopping = not args.technical_smoke
    if args.early_stopping and args.max_epochs <= 1 and not args.technical_smoke:
        parser.error("Early stopping is not meaningful for a one-epoch technical test")
    return args


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cohort_split(cohort: pd.DataFrame, split: pd.DataFrame) -> tuple[list[str], list[str]]:
    required = {"slide_id", "case_id", "event_id", "label", "source_dataset", "stain_group"}
    if required - set(cohort.columns):
        raise ValueError(f"Cohort lacks columns: {sorted(required - set(cohort.columns))}")
    if cohort["slide_id"].eq("").any() or cohort["slide_id"].duplicated().any():
        raise ValueError("Cohort has blank or duplicate slide_id")
    if not cohort["label"].isin(("0", "1")).all():
        raise ValueError("Cohort must have binary string labels 0 and 1")
    if not {"train", "val", "test"}.issubset(split.columns):
        raise ValueError("Split requires train, val and test columns")
    if split["test"].ne("").any():
        raise ValueError("This runner uses validation-only CV; test must be empty")
    train = [sid for sid in split["train"] if sid]
    val = [sid for sid in split["val"] if sid]
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Duplicate slide within a split")
    if set(train) & set(val):
        raise ValueError("Train/validation slide overlap")
    if set(train) | set(val) != set(cohort["slide_id"]):
        raise ValueError("Train/validation must partition exactly this fixed cohort")
    indexed = cohort.set_index("slide_id")
    if set(indexed.loc[train, "case_id"]) & set(indexed.loc[val, "case_id"]):
        raise ValueError("Patient leakage across train/validation")
    for name, ids in (("train", train), ("val", val)):
        if set(indexed.loc[ids, "label"]) != {"0", "1"}:
            raise ValueError(f"{name} lacks one ACR class")
    if cohort.groupby(["case_id", "event_id"])["label"].nunique().gt(1).any():
        raise ValueError("An event has conflicting ACR labels")
    return train, val


def small_balanced(split: object, *, per_class: int = 2) -> object:
    """Return a Generic_Split with full bags, never partial patch features."""
    from dataset_modules.dataset_generic import Generic_Split

    slide_data = split.slide_data
    pieces = []
    for label in (0, 1):
        piece = slide_data[slide_data["label"] == label].head(per_class)
        if len(piece) < per_class:
            raise ValueError(f"Technical smoke needs {per_class} slides in class {label}")
        pieces.append(piece)
    selected = pd.concat(pieces, ignore_index=True)
    return Generic_Split(selected, data_dir=split.data_dir,
                         num_classes=split.num_classes, use_h5=False)


def event_predictions(results: dict, cohort: pd.DataFrame,
                      val_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    indexed = cohort.set_index("slide_id")
    rows = []
    if set(results) != set(val_ids):
        raise ValueError("CLAM returned a different validation slide set")
    for sid in val_ids:
        output = results[sid]
        score = float(np.asarray(output["prob"]).reshape(-1)[1])
        record = indexed.loc[sid]
        if (int(output["label"]) != int(record["label"])
                or not np.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError(f"Invalid prediction/label for {sid}")
        rows.append({"slide_id": sid, "case_id": record["case_id"],
                     "event_id": record["event_id"], "label": int(record["label"]),
                     "prob_positive": score})
    slides = pd.DataFrame(rows)
    events = slides.groupby(["case_id", "event_id"], as_index=False).agg(
        label=("label", "first"), prob_positive=("prob_positive", "mean"),
        slide_count=("slide_id", "size"))
    if slides.groupby(["case_id", "event_id"])["label"].nunique().gt(1).any():
        raise ValueError("Conflicting ACR label among slides in one event")
    metrics = {
        "slides": len(slides),
        "events": len(events),
        "positive_events": int(events["label"].sum()),
        "event_average_precision": float(average_precision_score(events["label"], events["prob_positive"])),
        "event_roc_auc": float(roc_auc_score(events["label"], events["prob_positive"])),
        "event_brier": float(brier_score_loss(events["label"], events["prob_positive"])),
        "aggregation": "mean positive-class probability of same-stain slides within (case_id,event_id)",
    }
    return slides, events, metrics


def main() -> int:
    args = arguments()
    cohort = pd.read_csv(args.cohort_csv, dtype=str, keep_default_na=False)
    split = pd.read_csv(args.split_csv, dtype=str, keep_default_na=False)
    train_ids, val_ids = validate_cohort_split(cohort, split)
    if args.policy == "geometry_he_scale_light" and set(cohort["stain_group"]) != {"HE"}:
        raise ValueError("H/E-scale policy requires an HE-only cohort")
    all_paths = [args.baseline_dir / "pt_files" / f"{sid}.pt" for sid in train_ids + val_ids]
    missing = [str(path) for path in all_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing original bags, first 10: {missing[:10]}")

    import torch
    from dataset_modules.dataset_generic import Generic_MIL_Dataset
    from feature_view_bank import torch_feature_loader
    if args.training_protocol == "best_val_v2":
        from clam_view_training import fit_best_checkpoint
        from models.model_clam import CLAM_SB  # Verify server model import during preflight.
    else:
        from utils.core_utils import train

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = Generic_MIL_Dataset(csv_path=str(args.cohort_csv),
                                  data_dir=str(args.baseline_dir),
                                  shuffle=False, seed=args.seed, print_info=True,
                                  label_dict={0: 0, 1: 1}, patient_strat=False,
                                  ignore=[])
    train_split, val_split, test_split = dataset.return_splits(
        from_id=False, csv_path=str(args.split_csv))
    if test_split is not None or train_split is None or val_split is None:
        raise ValueError("Expected a nonempty train/val split and no test split")
    if set(train_split.slide_data["slide_id"]) != set(train_ids) or set(
            val_split.slide_data["slide_id"]) != set(val_ids):
        raise ValueError("CLAM's split reader dropped slides")
    if args.technical_smoke:
        train_split = small_balanced(train_split)
        val_split = small_balanced(val_split)
        print("TECHNICAL SMOKE: four full train bags and four full val bags; ignore metrics")

    active_train = train_split.slide_data.to_dict("records")
    bank_audit = validate_feature_bank(active_train, baseline_dir=args.baseline_dir,
                                       bank_root=args.bank_root, policy=args.policy,
                                       total_views=args.total_views)
    if args.deep_feature_check:
        for row in active_train:
            sid = str(row["slide_id"])
            reference_shape = tuple(torch_feature_loader(
                args.baseline_dir / "pt_files" / f"{sid}.pt").shape)
            for view in range(1, args.total_views):
                bag = torch_feature_loader(args.bank_root / args.policy /
                                           f"view_{view}" / "pt_files" / f"{sid}.pt")
                if tuple(bag.shape) != reference_shape:
                    raise ValueError(f"Bag shape differs across views: {sid}, view={view}")
        for sid in val_split.slide_data["slide_id"]:
            torch_feature_loader(args.baseline_dir / "pt_files" / f"{sid}.pt")
    print(f"Preflight OK: {bank_audit}, val_original_slides={len(val_split)}")
    if args.preflight_only:
        return 0

    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty results directory: {args.results_dir}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    wrapped = FeatureViewTrainSplit(
        train_split, baseline_dir=args.baseline_dir, bank_root=args.bank_root,
        policy=args.policy, total_views=args.total_views,
        seed=args.view_seed + args.fold, log_path=args.results_dir / "train_view_draws.csv")
    args.n_classes = 2
    args.embed_dim = 1536
    args.results_dir = str(args.results_dir)
    args.bag_loss = "ce"
    args.inst_loss = "ce"
    args.model_type = "clam_sb"
    args.subtyping = False
    args.weighted_sample = args.sampler == "weighted"
    args.opt = "adam"
    args.lr_scheduler = "none"
    if args.training_protocol == "legacy200":
        args.min_lr = 0.0
    args.no_val = False
    args.cv_validation = True
    args.testing = False
    args.log_data = False
    args.use_maqw = False
    args.use_ddpm_denoise = False
    training_report = {"training_protocol": args.training_protocol}
    try:
        if args.training_protocol == "best_val_v2":
            results, slide_auc, training_report = fit_best_checkpoint(wrapped, val_split, args)
        else:
            results, slide_auc, _, _, _ = train(
                (wrapped, val_split, None), args.fold, args, rank=0, world_size=1,
                local_rank=0)
    finally:
        wrapped.close()
    if args.technical_smoke:
        val_ids = val_split.slide_data["slide_id"].tolist()
    slides, events, event_metrics = event_predictions(results, cohort, val_ids)
    result_dir = Path(args.results_dir)
    slides.to_csv(result_dir / "val_slide_predictions.csv", index=False)
    events.to_csv(result_dir / "val_event_predictions.csv", index=False)
    report = {
        "technical_smoke": args.technical_smoke,
        "exploratory_cv_not_independent_final_test": True,
        "fold": args.fold,
        "policy": args.policy,
        "total_views": args.total_views,
        "original_probability": 1.0 if args.total_views == 1 else 0.5,
        "sampler": args.sampler,
        "max_epochs": args.max_epochs,
        "early_stopping": args.early_stopping,
        "train_draws": wrapped.draw_count,
        "view_exposure_counts": dict(sorted(wrapped.selected_counts.items())),
        "slide_roc_auc": float(slide_auc),
        "cohort_sha256": sha256(args.cohort_csv),
        "split_sha256": sha256(args.split_csv),
        "view_bank_preflight": bank_audit,
        **event_metrics,
        **training_report,
    }
    (result_dir / "view_run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
