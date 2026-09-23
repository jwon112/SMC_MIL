#!/usr/bin/env python3
"""Build event-level stain-aware MIL manifests from curated gold slide labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from utils.event_mil import normalize_stain_group

TASKS = {"acr_high": "smc_acr_binary_0r1r_vs_2r3r", "amr_positive": "smc_amr_binary_pamr0_vs_positive", "significant_rejection": "smc_significant_rejection_binary"}
SPLIT_NAMES = {"acr_high": "smc_cv_acr_0r1r_vs_2r3r", "amr_positive": "smc_cv_amr_pamr0_vs_positive", "significant_rejection": "smc_cv_significant_rejection"}

def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-dir", type=Path, required=True); p.add_argument("--curation-manifest", type=Path, required=True)
    p.add_argument("--split-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--exclude-slide-ids-csv", type=Path, help="Optional CSV with a slide_id column to omit for stain-label sensitivity analysis.")
    p.add_argument("--folds", type=int, choices=(3, 5), default=5); p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    return p.parse_args()

def curated(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, dtype={"slide_id": str}).fillna("")
    need = {"slide_id", "stain_group", "include_quality_usable"}
    if missing := need - set(x): raise ValueError(f"Curation manifest missing: {sorted(missing)}")
    if x.slide_id.duplicated().any(): raise ValueError("Duplicate slide_id in curation manifest")
    x["stain_group"] = x.stain_group.map(normalize_stain_group)
    return x.loc[x.include_quality_usable.astype(str).str.lower().eq("true"), ["slide_id", "stain_group"]]

def split_events(slides: pd.DataFrame, path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, dtype=str); chunks = []
    for name in ("train", "val"):
        ids = set(source[name].dropna()); part = slides.loc[slides.slide_id.isin(ids), ["event_id"]].drop_duplicates().copy(); part["split"] = name; chunks.append(part)
    out = pd.concat(chunks, ignore_index=True)
    if (out.groupby("event_id").split.nunique() > 1).any(): raise ValueError(f"Event crosses a patient split: {path}")
    return out

def build(task: str, args: argparse.Namespace, stains: pd.DataFrame) -> None:
    base = TASKS[task]; labels = pd.read_csv(args.label_dir / f"{base}.csv", dtype={"slide_id": str, "event_id": str, "case_id": str}).fillna("")
    need = {"slide_id", "event_id", "case_id", "label", "label_text"}
    if missing := need - set(labels): raise ValueError(f"{task} label CSV missing: {sorted(missing)}")
    slides = labels.merge(stains, how="left", on="slide_id", validate="one_to_one"); slides = slides.loc[slides.stain_group.notna()].copy()
    if (slides.groupby("event_id").label.nunique() > 1).any(): raise ValueError(f"{task}: inconsistent labels in event")
    cols = ["event_id", "case_id", "label", "label_text"] + (["biopsy_date"] if "biopsy_date" in slides else [])
    events = slides[cols].drop_duplicates("event_id").sort_values("event_id")
    root = args.output_root / task; root.mkdir(parents=True, exist_ok=True); events.to_csv(root / "events.csv", index=False)
    slides[["event_id", "slide_id", "stain_group"]].sort_values(["event_id", "slide_id"]).to_csv(root / "event_slides.csv", index=False)
    out = root / "splits"; out.mkdir(exist_ok=True); report = []
    for seed in args.seeds:
        suffix = "" if seed == 1 else f"_seed{seed}"
        split_root = args.split_root / f"{SPLIT_NAMES[task]}_standard{args.folds}{suffix}"
        if not split_root.is_dir(): raise FileNotFoundError(f"Missing source split directory: {split_root}")
        seed_out = out / f"seed{seed}"; seed_out.mkdir(exist_ok=True)
        for fold in range(args.folds):
            mapped = split_events(slides, split_root / f"splits_{fold}.csv")
            saved = {}
            for part in ("train", "val"): saved[part] = mapped.loc[mapped.split.eq(part), "event_id"].sort_values().reset_index(drop=True)
            pd.DataFrame(saved).to_csv(seed_out / f"splits_{fold}.csv", index=False)
            for part, ids in saved.items():
                event_part = events.loc[events.event_id.isin(ids)]; report.append({"seed": seed, "fold": fold, "split": part, "events": len(event_part), "positive_events": int(event_part.label.sum()), "slides": int(slides.event_id.isin(ids).sum())})
    pd.DataFrame(report).to_csv(out / "fold_summary.csv", index=False)
    combinations = slides.groupby("event_id").stain_group.agg(lambda x: "+".join(sorted(set(x))))
    print(f"[OK] {task}: events={len(events)}, positive={int(events.label.sum())}, slides={len(slides)} -> {root}")
    print("Stain combinations:\n" + combinations.value_counts().to_string()); print(pd.DataFrame(report).to_string(index=False))

def main() -> int:
    args = arguments(); stains = curated(args.curation_manifest)
    if args.exclude_slide_ids_csv:
        excluded = pd.read_csv(args.exclude_slide_ids_csv, dtype={"slide_id": str})
        if "slide_id" not in excluded:
            raise ValueError("Exclusion CSV requires a slide_id column")
        excluded_ids = set(excluded.slide_id.dropna().astype(str))
        before = len(stains)
        stains = stains.loc[~stains.slide_id.isin(excluded_ids)].copy()
        print(f"[INFO] excluded review slides: {before - len(stains)} matched / {len(excluded_ids)} requested")
    for task in args.tasks: build(task, args, stains)
    return 0
if __name__ == "__main__": raise SystemExit(main())
