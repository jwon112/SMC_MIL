#!/usr/bin/env python3
"""Combine slide-to-event and event-MIL summaries into one ablation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = (
    "auroc",
    "pr_auc",
    "pr_auc_lift",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide-summary", type=Path, required=True)
    parser.add_argument("--event-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    slide = pd.read_csv(args.slide_summary)
    event = pd.read_csv(args.event_summary)
    required_slide = {"task", "scale", *METRICS}
    required_event = {"task", "scale", "mode", *METRICS}
    if missing := required_slide.difference(slide.columns):
        raise ValueError(f"Slide summary missing columns: {sorted(missing)}")
    if missing := required_event.difference(event.columns):
        raise ValueError(f"Event summary missing columns: {sorted(missing)}")

    slide = slide[["task", "scale", *METRICS]].copy()
    slide["approach"] = "slide_clam_event_mean"
    event = event[["task", "scale", "mode", *METRICS]].copy()
    event["approach"] = event["mode"].map({
        "agnostic": "event_mil_agnostic",
        "aware_nomask": "event_mil_stain_aware_no_mask",
        "aware": "event_mil_stain_aware",
    })
    if event.approach.isna().any():
        raise ValueError(f"Unsupported event modes: {sorted(event.loc[event.approach.isna(), 'mode'].unique())}")
    event = event.drop(columns="mode")
    combined = pd.concat([slide, event], ignore_index=True)

    baseline = slide.set_index(["task", "scale"])
    for metric in METRICS:
        reference = combined[["task", "scale"]].merge(
            baseline[[metric]].rename(columns={metric: "baseline"}).reset_index(),
            on=["task", "scale"],
            how="left",
            validate="many_to_one",
        )
        combined[f"delta_{metric}_vs_slide"] = combined[metric] - reference["baseline"]

    order = {"slide_clam_event_mean": 0, "event_mil_agnostic": 1, "event_mil_stain_aware_no_mask": 2, "event_mil_stain_aware": 3}
    combined["_order"] = combined.approach.map(order)
    combined = combined.sort_values(["task", "scale", "_order"]).drop(columns="_order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
