#!/usr/bin/env python3
"""Extract page-preserving Markdown and optional PNG previews from PDFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_render_pages(page_count: int, mode: str) -> list[int]:
    if mode == "none":
        return []
    if mode == "all":
        return list(range(page_count))
    candidates = {0, page_count // 2, page_count - 1}
    return sorted(page for page in candidates if 0 <= page < page_count)


def extract_one(
    pdf_path: Path,
    output_dir: Path,
    render_dir: Path | None,
    render_mode: str,
    dpi: int,
) -> dict[str, object]:
    document = fitz.open(pdf_path)
    page_texts: list[str] = []
    empty_pages: list[int] = []
    for page_index, page in enumerate(document):
        text = page.get_text("text", sort=True).strip()
        if len(text) < 20:
            empty_pages.append(page_index + 1)
        page_texts.append(text)

    metadata = document.metadata or {}
    title = (metadata.get("title") or pdf_path.stem).strip()
    digest = sha256_file(pdf_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# {title}",
        "",
        f"- Source PDF: `../paper_origin/{pdf_path.name}`",
        f"- SHA-256: `{digest}`",
        f"- PDF pages: {document.page_count}",
        f"- Extracted UTC: {generated_at}",
        "- Extraction: PyMuPDF sorted text; no OCR or formula repair",
        "",
    ]
    for page_number, text in enumerate(page_texts, start=1):
        parts.extend(
            [
                f"<!-- PDF_PAGE: {page_number} -->",
                "",
                f"## PDF page {page_number}",
                "",
                text,
                "",
            ]
        )
    markdown_path.write_text("\n".join(parts), encoding="utf-8")

    rendered: list[str] = []
    if render_dir is not None:
        paper_render_dir = render_dir / pdf_path.stem
        paper_render_dir.mkdir(parents=True, exist_ok=True)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in selected_render_pages(document.page_count, render_mode):
            output_path = paper_render_dir / f"page-{page_index + 1:03d}.png"
            document[page_index].get_pixmap(matrix=matrix, alpha=False).save(output_path)
            rendered.append(output_path.as_posix())

    report = {
        "paper_id": pdf_path.stem,
        "source_pdf": pdf_path.as_posix(),
        "source_sha256": digest,
        "markdown": markdown_path.as_posix(),
        "pages": document.page_count,
        "characters": sum(len(text) for text in page_texts),
        "empty_or_near_empty_pages": empty_pages,
        "rendered_pages": rendered,
    }
    document.close()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reference/extracted_text"),
    )
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument(
        "--render",
        choices=("none", "sample", "all"),
        default="none",
        help="Render no pages, first/middle/last pages, or every page.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    for pdf_path in args.pdfs:
        if not pdf_path.is_file():
            print(f"ERROR missing PDF: {pdf_path}", file=sys.stderr)
            return 1
        report = extract_one(
            pdf_path,
            args.output_dir,
            args.render_dir,
            args.render,
            args.dpi,
        )
        reports.append(report)
        print(
            f"OK {report['paper_id']}: pages={report['pages']} "
            f"chars={report['characters']} empty={report['empty_or_near_empty_pages']}"
        )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"documents": reports}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
