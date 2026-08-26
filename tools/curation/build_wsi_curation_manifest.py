#!/usr/bin/env python3
"""Prepare quality and stain curation manifests for exp3 DICOM and MRXS WSI.

The script only triages slides for review. It never automatically excludes a
slide because image-quality heuristics and filename-based stain rules are not
reliable enough to be final clinical decisions.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_smc_training_labels import (
    feature_slide_id,
    load_exclusions,
    load_labels,
    normalize,
)


STAIN_GROUPS = {"HE", "IHC", "special_other", "unknown"}
IHC_MARKERS = {
    "C1Q", "C3D", "C4D", "CD3", "CD4", "CD8", "CD20", "CD31", "CD34",
    "CD45", "CD56", "CD68", "CD79A", "CMV", "IGG", "IGM", "KI67", "P53",
    "SV40",
}
SPECIAL_STAINS = {"AFB", "CONGO", "EVG", "GMS", "MASS", "PAS", "SILVER", "TRICHROME"}
COLOR_FEATURE_COLUMNS = [
    "color_mean_r", "color_mean_g", "color_mean_b",
    "color_std_r", "color_std_g", "color_std_b",
    *[f"color_hue_bin_{index}" for index in range(8)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path, required=True)
    parser.add_argument("--mrxs-root", type=Path, required=True)
    parser.add_argument("--label-xlsx", type=Path, default=None,
                        help="Optional Sheet1 workbook used only to mark pathology-ID gold matches.")
    parser.add_argument("--quality-exclusions", type=Path, default=None,
                        help="Existing user-reviewed exclusions to preserve as known exclusions.")
    parser.add_argument("--stain-map", type=Path, default=None,
                        help="Optional reviewed stain_signature map produced by this script.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality-tail-frac", type=float, default=0.02,
                        help="Tail fraction per quality metric to flag for manual review (default: 0.02).")
    parser.add_argument("--audit-unflagged-count", type=int, default=50,
                        help="Random unflagged slides exported for quality-control audit (default: 50).")
    parser.add_argument("--stain-clusters", type=int, default=24,
                        help="Number of thumbnail-color clusters for grouped stain review (default: 24).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--export-quality-previews", action="store_true",
                        help="Export resized thumbnail PNGs for the quality review queue.")
    parser.add_argument("--export-stain-cluster-previews", action="store_true",
                        help="Export thumbnail montages for stain color-cluster review.")
    return parser.parse_args()


def read_qc_slides(dataset_root: Path, source_dataset: str) -> list[dict[str, object]]:
    qc_path = dataset_root / "_qc" / "mask_qc_labels.csv"
    if not qc_path.is_file():
        raise FileNotFoundError(f"QC manifest not found: {qc_path}")

    slides: list[dict[str, object]] = []
    with qc_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rel_path = row.get("slide_rel_path", "").strip().replace("\\", "/")
            if not rel_path:
                continue
            event_key = normalize(rel_path.split("/")[0])
            slides.append({
                "source_dataset": source_dataset,
                "dataset_root": str(dataset_root),
                "slide_rel_path": rel_path,
                "slide_id": feature_slide_id(rel_path),
                "event_key": event_key,
                "qc_status": row.get("qc_status", "").strip(),
            })
    return slides


def stain_signature(slide_rel_path: str, event_key: str) -> str:
    """Remove a leading event identifier while preserving the slide's stain suffix."""
    leaf = slide_rel_path.replace("\\", "/").split("/")[-1]
    target = normalize(event_key)
    observed = ""
    for index, char in enumerate(leaf):
        if char.isalnum():
            observed += char.upper()
        if observed == target:
            return leaf[index + 1:].lstrip(" -_.") or leaf
        if not target.startswith(observed):
            break
    return leaf


def automatic_stain_group(signature: str) -> tuple[str, str, str, str]:
    upper = signature.upper()
    tokens = set(re.findall(r"[A-Z]+[0-9+]*", upper))
    if "HE" in tokens or "H&E" in upper or re.search(r"(?:^|[^A-Z])H\s*E(?:$|[^A-Z])", upper):
        return "HE", "HE", "filename_rule", "high"
    marker_hits = sorted(token for token in tokens if token in IHC_MARKERS)
    if marker_hits:
        return "IHC", marker_hits[0], "filename_rule", "high"
    special_hits = sorted(token for token in tokens if token in SPECIAL_STAINS)
    if special_hits:
        return "special_other", special_hits[0], "filename_rule", "high"
    return "unknown", "", "none", "unknown"


def load_stain_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Stain map not found: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"stain_signature", "stain_group"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Stain map missing columns: {sorted(missing)}")
    mapped: dict[str, dict[str, str]] = {}
    for row in frame.to_dict("records"):
        signature = row["stain_signature"].strip()
        group = row["stain_group"].strip()
        if not signature or not group:
            continue
        if group not in STAIN_GROUPS:
            raise ValueError(f"Unsupported stain_group '{group}' for signature '{signature}'")
        mapped[signature] = {
            "stain_group": group,
            "stain_raw": row.get("stain_raw", "").strip(),
            "stain_confidence": row.get("stain_confidence", "manual").strip() or "manual",
            "stain_note": row.get("stain_note", "").strip(),
        }
    return mapped


def image_metrics(slide_dir: Path) -> dict[str, object]:
    thumbnail_path = slide_dir / "atlaspatch" / "thumbnail.png"
    base = {
        "thumbnail_path": str(thumbnail_path),
        "thumbnail_state": "ok",
        "thumbnail_width": np.nan,
        "thumbnail_height": np.nan,
        "luminance_mean": np.nan,
        "luminance_std": np.nan,
        "saturation_mean": np.nan,
        "bright_fraction": np.nan,
        "dark_fraction": np.nan,
        "sharpness_score": np.nan,
        "tissue_mask_path": "",
        "tissue_ratio": np.nan,
        **{column: np.nan for column in COLOR_FEATURE_COLUMNS},
    }
    if not thumbnail_path.is_file():
        base["thumbnail_state"] = "missing"
        return base
    try:
        with Image.open(thumbnail_path) as source:
            image = source.convert("RGB")
            image.thumbnail((1600, 1600))
            array = np.asarray(image, dtype=np.float32)
    except Exception as exc:
        base["thumbnail_state"] = f"unreadable:{type(exc).__name__}"
        return base

    base["thumbnail_width"], base["thumbnail_height"] = image.size
    luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    max_channel = array.max(axis=-1)
    min_channel = array.min(axis=-1)
    saturation = np.divide(max_channel - min_channel, max_channel, out=np.zeros_like(max_channel), where=max_channel > 0)
    gradients = np.concatenate([np.diff(luminance, axis=0).ravel(), np.diff(luminance, axis=1).ravel()])
    base.update({
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "saturation_mean": float(saturation.mean()),
        "bright_fraction": float((luminance >= 245).mean()),
        "dark_fraction": float((luminance <= 15).mean()),
        "sharpness_score": float(gradients.var()) if gradients.size else 0.0,
    })
    tissue_like = luminance < 245
    if tissue_like.sum() < 64:
        tissue_like = np.ones(luminance.shape, dtype=bool)
    values = (array[tissue_like] / 255.0).reshape(-1, 3)
    red, green, blue = values[:, 0], values[:, 1], values[:, 2]
    maximum = values.max(axis=1)
    minimum = values.min(axis=1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green)
    blue_max = nonzero & (maximum == blue)
    hue[red_max] = ((green[red_max] - blue[red_max]) / delta[red_max]) % 6
    hue[green_max] = ((blue[green_max] - red[green_max]) / delta[green_max]) + 2
    hue[blue_max] = ((red[blue_max] - green[blue_max]) / delta[blue_max]) + 4
    hue /= 6.0
    pixel_saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue_hist, _ = np.histogram(hue, bins=8, range=(0.0, 1.0), weights=pixel_saturation)
    hue_hist = hue_hist / hue_hist.sum() if hue_hist.sum() else hue_hist
    color_values = [
        float(red.mean()), float(green.mean()), float(blue.mean()),
        float(red.std()), float(green.std()), float(blue.std()),
        *[float(value) for value in hue_hist],
    ]
    base.update(dict(zip(COLOR_FEATURE_COLUMNS, color_values, strict=True)))

    for filename in ("tissue_mask_manual.png", "tissue_mask.png"):
        mask_path = slide_dir / "atlaspatch" / filename
        if not mask_path.is_file():
            continue
        try:
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L"), dtype=np.uint8)
            base["tissue_mask_path"] = str(mask_path)
            base["tissue_ratio"] = float((mask > 0).mean())
            break
        except Exception:
            continue
    return base


def add_quality_triage(frame: pd.DataFrame, tail_fraction: float, audit_count: int, seed: int) -> pd.DataFrame:
    if not 0.0 < tail_fraction < 0.25:
        raise ValueError("--quality-tail-frac must be between 0 and 0.25")
    frame = frame.copy()
    metric_columns = ["tissue_ratio", "sharpness_score", "luminance_mean", "saturation_mean"]
    valid = frame[frame.thumbnail_state.eq("ok")]
    quantiles = {
        column: valid[column].dropna().quantile([tail_fraction, 1.0 - tail_fraction]).to_dict()
        for column in metric_columns
    }
    flags: list[str] = []
    for row in frame.itertuples(index=False):
        row_flags: list[str] = []
        if row.thumbnail_state != "ok":
            row_flags.append("thumbnail_" + str(row.thumbnail_state))
        else:
            if pd.notna(row.tissue_ratio) and row.tissue_ratio <= quantiles["tissue_ratio"][tail_fraction]:
                row_flags.append("low_tissue_ratio_tail")
            if pd.notna(row.sharpness_score) and row.sharpness_score <= quantiles["sharpness_score"][tail_fraction]:
                row_flags.append("low_sharpness_tail")
            if (
                pd.notna(row.luminance_mean)
                and pd.notna(row.saturation_mean)
                and row.luminance_mean >= quantiles["luminance_mean"][1.0 - tail_fraction]
                and row.saturation_mean <= quantiles["saturation_mean"][tail_fraction]
            ):
                row_flags.append("bright_desaturated_tail")
        flags.append(";".join(row_flags))
    frame["quality_auto_flags"] = flags
    frame["quality_auto_priority"] = np.where(
        frame["quality_auto_flags"].eq(""), "none",
        np.where(frame["quality_auto_flags"].str.count(";").add(1).ge(2), "high", "medium"),
    )
    frame.loc[frame["known_quality_exclusion"], "quality_auto_priority"] = "known_exclusion"
    frame["quality_review_reason"] = frame["quality_auto_flags"]
    frame["quality_audit_sample"] = False
    eligible_for_audit = frame[
        frame["quality_auto_priority"].eq("none") & ~frame["known_quality_exclusion"]
    ]
    if audit_count > 0 and not eligible_for_audit.empty:
        selected = eligible_for_audit.sample(n=min(audit_count, len(eligible_for_audit)), random_state=seed).index
        frame.loc[selected, "quality_audit_sample"] = True
        frame.loc[selected, "quality_review_reason"] = "random_audit"
    return frame


def export_quality_previews(queue: pd.DataFrame, output_dir: Path) -> None:
    image_dir = output_dir / "quality_review_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for rank, row in enumerate(queue.itertuples(index=False), start=1):
        source = Path(row.thumbnail_path)
        if not source.is_file():
            continue
        filename = f"{rank:04d}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', row.slide_id)}.png"
        destination = image_dir / filename
        try:
            with Image.open(source) as image:
                preview = image.convert("RGB")
                preview.thumbnail((1400, 1400))
                preview.save(destination)
            index_rows.append({"rank": rank, "slide_id": row.slide_id, "preview_file": filename})
        except Exception:
            continue
    pd.DataFrame(index_rows).to_csv(output_dir / "quality_preview_index.csv", index=False)


def add_color_clusters(frame: pd.DataFrame, n_clusters: int, seed: int) -> pd.DataFrame:
    if n_clusters < 2:
        raise ValueError("--stain-clusters must be at least 2")
    frame = frame.copy()
    frame["stain_color_cluster"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    valid = frame.dropna(subset=COLOR_FEATURE_COLUMNS)
    if len(valid) < 2:
        return frame
    cluster_count = min(n_clusters, len(valid))
    features = StandardScaler().fit_transform(valid[COLOR_FEATURE_COLUMNS])
    clusters = KMeans(n_clusters=cluster_count, n_init=20, random_state=seed).fit_predict(features)
    frame.loc[valid.index, "stain_color_cluster"] = clusters
    return frame


def export_cluster_montages(frame: pd.DataFrame, output_dir: Path, seed: int) -> None:
    image_dir = output_dir / "stain_cluster_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for cluster, group in frame.dropna(subset=["stain_color_cluster"]).groupby("stain_color_cluster"):
        candidates = group[group.thumbnail_state.eq("ok")]
        if candidates.empty:
            continue
        count = min(12, len(candidates))
        selection = candidates.iloc[rng.choice(len(candidates), size=count, replace=False)]
        tiles = []
        for row in selection.itertuples(index=False):
            try:
                with Image.open(row.thumbnail_path) as source:
                    tile = source.convert("RGB")
                    tile.thumbnail((300, 230))
                    canvas = Image.new("RGB", (310, 260), "white")
                    canvas.paste(tile, ((310 - tile.width) // 2, 4))
                    ImageDraw.Draw(canvas).text((5, 238), str(row.slide_id)[:42], fill="black")
                    tiles.append(canvas)
            except Exception:
                continue
        if not tiles:
            continue
        columns = 4
        rows = int(np.ceil(len(tiles) / columns))
        montage = Image.new("RGB", (columns * 310, rows * 260), "white")
        for index, tile in enumerate(tiles):
            montage.paste(tile, ((index % columns) * 310, (index // columns) * 260))
        montage.save(image_dir / f"cluster_{int(cluster):02d}.jpg", quality=92)


def main() -> int:
    args = parse_args()
    labels = load_labels(args.label_xlsx) if args.label_xlsx else None
    exclusions = load_exclusions(args.quality_exclusions) if args.quality_exclusions else set()
    stain_map = load_stain_map(args.stain_map)
    slides = read_qc_slides(args.dicom_root, "exp3") + read_qc_slides(args.mrxs_root, "mrxs13")
    if len({slide["slide_id"] for slide in slides}) != len(slides):
        raise ValueError("slide_id collision across curation inputs")

    records: list[dict[str, object]] = []
    for slide in slides:
        root = Path(str(slide["dataset_root"]))
        rel_path = str(slide["slide_rel_path"])
        signature = stain_signature(rel_path, str(slide["event_key"]))
        auto_group, auto_raw, auto_source, auto_confidence = automatic_stain_group(signature)
        override = stain_map.get(signature)
        if override:
            stain_group = override["stain_group"]
            stain_raw = override["stain_raw"] or auto_raw
            stain_source = "signature_map"
            stain_confidence = override["stain_confidence"]
            stain_note = override["stain_note"]
        else:
            stain_group = auto_group
            stain_raw = auto_raw
            stain_source = auto_source
            stain_confidence = auto_confidence
            stain_note = ""
        known_exclusion = slide["source_dataset"] == "exp3" and rel_path.lower() in exclusions
        record = {
            **slide,
            "known_quality_exclusion": known_exclusion,
            "gold_pathology_id_match": bool(labels is not None and slide["event_key"] in labels.index),
            "stain_signature": signature,
            "stain_group_auto": auto_group,
            "stain_raw_auto": auto_raw,
            "stain_source_auto": auto_source,
            "stain_confidence_auto": auto_confidence,
            "stain_group": stain_group,
            "stain_raw": stain_raw,
            "stain_source": stain_source,
            "stain_confidence": stain_confidence,
            "stain_note": stain_note,
            "needs_stain_signature_review": stain_group == "unknown" or stain_confidence != "high",
            **image_metrics(root / rel_path),
        }
        records.append(record)

    frame = add_color_clusters(pd.DataFrame(records), args.stain_clusters, args.seed)
    frame = add_quality_triage(frame, args.quality_tail_frac, args.audit_unflagged_count, args.seed)
    frame["quality_manual_status"] = ""
    frame["quality_manual_reason"] = ""
    frame["quality_reviewer"] = ""
    frame["quality_reviewed_at"] = ""
    frame["curation_note"] = ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "slide_curation_manifest.csv", index=False)

    quality_queue = frame[
        (~frame["known_quality_exclusion"])
        & ((frame["quality_auto_priority"] != "none") | frame["quality_audit_sample"])
    ].copy()
    quality_queue = quality_queue.sort_values(["quality_auto_priority", "quality_review_reason", "slide_id"])
    quality_queue.to_csv(args.output_dir / "quality_review_queue.csv", index=False)

    signatures = (
        frame.groupby("stain_signature", dropna=False)
        .agg(
            slides=("slide_id", "count"),
            source_datasets=("source_dataset", lambda values: ";".join(sorted(set(values)))),
            stain_group_auto=("stain_group_auto", "first"),
            stain_raw_auto=("stain_raw_auto", "first"),
            stain_confidence_auto=("stain_confidence_auto", "first"),
            example_slide_rel_path=("slide_rel_path", "first"),
            stain_group=("stain_group", "first"),
            stain_raw=("stain_raw", "first"),
        )
        .reset_index()
    )
    signatures["stain_confidence"] = ""
    signatures["stain_note"] = ""
    signatures["needs_review"] = signatures["stain_group"].eq("unknown") | ~signatures["stain_confidence_auto"].eq("high")
    signatures.sort_values(["needs_review", "slides", "stain_signature"], ascending=[False, False, True]).to_csv(
        args.output_dir / "stain_signature_review.csv", index=False
    )

    clustered = frame.dropna(subset=["stain_color_cluster"]).copy()
    if not clustered.empty:
        cluster_rows = []
        for cluster, group in clustered.groupby("stain_color_cluster", sort=True):
            high_confidence = group[group["stain_confidence"].eq("high")]
            high_groups = Counter(high_confidence["stain_group"])
            cluster_rows.append({
                "stain_color_cluster": int(cluster),
                "slides": len(group),
                "high_confidence_HE": high_groups["HE"],
                "high_confidence_IHC": high_groups["IHC"],
                "high_confidence_special_other": high_groups["special_other"],
                "high_confidence_unknown": high_groups["unknown"],
                "example_slide_rel_paths": ";".join(group["slide_rel_path"].head(5)),
                "stain_group": "",
                "stain_confidence": "",
                "stain_note": "",
            })
        pd.DataFrame(cluster_rows).to_csv(args.output_dir / "stain_color_cluster_review.csv", index=False)

    if args.export_quality_previews:
        export_quality_previews(quality_queue, args.output_dir)
    if args.export_stain_cluster_previews:
        export_cluster_montages(frame, args.output_dir, args.seed)

    print(f"Slides: {len(frame)}")
    print(f"Gold pathology-ID matched: {int(frame.gold_pathology_id_match.sum())}")
    print(f"Known quality exclusions: {int(frame.known_quality_exclusion.sum())}")
    print(f"Quality review queue: {len(quality_queue)}")
    print(f"Stain signatures needing mapping: {int(signatures.needs_review.sum())}")
    print(f"Stain color clusters: {int(frame.stain_color_cluster.nunique())}")
    print(f"Manifest: {args.output_dir / 'slide_curation_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
