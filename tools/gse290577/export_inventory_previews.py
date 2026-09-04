#!/usr/bin/env python3
"""Export ROI previews and an HTML index for visual alignment/QC."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openslide_mpp import read_roi_thumbnail  # noqa: E402


def integer(row: dict[str, str], name: str) -> int:
    return int(float(row.get(name, "0") or 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=768)
    parser.add_argument("--fallback-mpp", type=float)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with args.inventory.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]
    previews = args.output_dir / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, row in enumerate(rows, 1):
        roi = (integer(row, "roi_x"), integer(row, "roi_y"), integer(row, "roi_width"), integer(row, "roi_height"))
        image, _, _ = read_roi_thumbnail(args.dataset_root / row["source_rel_path"], roi, args.max_size, args.fallback_mpp)
        filename = f"{row['slide_id']}.jpg"
        image.save(previews / filename, quality=90)
        caption = " | ".join(filter(None, [row.get("patient_id"), row.get("biopsy_id"), row.get("stain"), row.get("acr_grade"), row.get("amr_grade")]))
        cards.append(f'<figure><img src="previews/{html.escape(filename)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        print(f"[OK] [{index}/{len(rows)}] {row['slide_id']}")
    page = """<!doctype html><meta charset="utf-8"><title>GSE290577 inventory QC</title>
<style>body{font-family:Arial,sans-serif;margin:16px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}figure{margin:0;border:1px solid #bbb;padding:8px}img{width:100%;height:260px;object-fit:contain;background:white}figcaption{font-size:12px;overflow-wrap:anywhere;margin-top:6px}</style>
<h1>GSE290577 inventory QC</h1><main>""" + "".join(cards) + "</main>"
    index = args.output_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    print(f"Index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
