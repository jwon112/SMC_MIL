#!/usr/bin/env python3
"""Download open-access PMC PDFs from the official PMC Open Data bucket.

Usage:
    python reference/tools/fetch_pmc_pdf.py \
        lipkova2022_crane=PMC9353336 \
        seraphin2023_cardiac_ssl=PMC10232288

The script resolves the newest versioned object for each PMCID and writes the
PDF as ``<paper_id>.pdf``. It does not update the project manifest because
bibliographic and access metadata require human review.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SMC-MIL-literature-ingest/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def resolve_pdf_key(pmcid: str) -> str:
    normalized = pmcid.upper()
    if not re.fullmatch(r"PMC\d+", normalized):
        raise ValueError(f"invalid PMCID: {pmcid}")
    query = urllib.parse.urlencode({"list-type": "2", "prefix": f"{normalized}."})
    xml_bytes = request_bytes(f"{BUCKET}/?{query}")
    root = ET.fromstring(xml_bytes)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    candidates: list[tuple[int, str]] = []
    pattern = re.compile(
        rf"^{re.escape(normalized)}\.(\d+)/{re.escape(normalized)}\.\1\.pdf$"
    )
    for element in root.findall("s3:Contents/s3:Key", namespace):
        key = element.text or ""
        match = pattern.fullmatch(key)
        if match:
            candidates.append((int(match.group(1)), key))
    if not candidates:
        raise FileNotFoundError(f"no open-data PDF found for {normalized}")
    return max(candidates)[1]


def parse_item(value: str) -> tuple[str, str]:
    try:
        paper_id, pmcid = value.split("=", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected PAPER_ID=PMCID") from exc
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", paper_id):
        raise argparse.ArgumentTypeError(f"invalid paper_id: {paper_id}")
    if not re.fullmatch(r"PMC\d+", pmcid.upper()):
        raise argparse.ArgumentTypeError(f"invalid PMCID: {pmcid}")
    return paper_id, pmcid.upper()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", nargs="+", type=parse_item, metavar="PAPER_ID=PMCID")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reference/paper_origin"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for paper_id, pmcid in args.items:
        output_path = args.output_dir / f"{paper_id}.pdf"
        if output_path.exists() and not args.overwrite:
            print(f"SKIP {paper_id}: {output_path} already exists")
            continue
        try:
            key = resolve_pdf_key(pmcid)
            url = f"{BUCKET}/{urllib.parse.quote(key, safe='/')}"
            data = request_bytes(url)
            if not data.startswith(b"%PDF-"):
                raise ValueError(f"downloaded object is not a PDF: {url}")
            output_path.write_bytes(data)
            print(
                f"OK {paper_id}: pmcid={pmcid} bytes={len(data)} "
                f"sha256={sha256_bytes(data)} source={url}"
            )
        except Exception as exc:  # report all requested downloads before failing
            failures += 1
            print(f"ERROR {paper_id}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
