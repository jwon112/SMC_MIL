#!/usr/bin/env python3
"""Validate the local literature repository and its cross-references.

The validator intentionally uses only the Python standard library so it can be
run before a project environment is installed. JSON Schema files document the
full record contract; this script checks the invariants most likely to break
retrieval or provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REFERENCE_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REFERENCE_ROOT / "indexes"
CARD_DIR = REFERENCE_ROOT / "cards"
EVIDENCE_DIR = REFERENCE_ROOT / "evidence"

CARD_REQUIRED = {
    "paper_id",
    "title",
    "authors",
    "year",
    "source_pdf",
    "source_text",
    "study_context",
    "main_findings",
    "limitations",
    "does_not_establish",
    "relevance_to_project",
    "tags",
    "source_locators",
    "review_status",
    "last_reviewed",
}
ATOM_REQUIRED = {
    "atom_id",
    "claim_id",
    "paper_id",
    "statement",
    "evidence_type",
    "supports",
    "does_not_establish",
    "method_or_result",
    "source_locator",
    "confidence",
    "review_status",
    "tags",
}
TOPIC_REQUIRED = {
    "topic_id",
    "description",
    "query_terms",
    "preferred_evidence_types",
    "paper_ids",
    "claim_ids",
    "fallback_topics",
}
VALID_CONFIDENCE = {"direct", "qualified", "unresolved"}
VALID_REVIEW_STATUS = {"draft", "reviewed", "needs_update"}
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected an object in {path}:{line_number}")
        value["_record_source"] = f"{path.name}:{line_number}"
        records.append(value)
    return records


def collect_records() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paper_index = load_json(INDEX_DIR / "papers.json").get("papers", [])
    claims = load_json(INDEX_DIR / "claims.json").get("claims", [])
    topics = load_json(INDEX_DIR / "topics.json").get("topics", [])
    cards = [load_json(path) for path in sorted(CARD_DIR.glob("*.json"))]
    atoms: list[dict[str, Any]] = []
    for path in sorted(EVIDENCE_DIR.glob("*.jsonl")):
        atoms.extend(load_jsonl(path))
    return paper_index, claims, topics, cards, atoms


def duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return repeated


def missing_keys(record: dict[str, Any], required: set[str]) -> set[str]:
    return required - record.keys()


def resolve_card_source(card_path: Path, source: str) -> Path:
    return (card_path.parent / source).resolve()


def validate(strict_sources: bool = False) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        paper_index, claims, topics, cards, atoms = collect_records()
    except ValueError as exc:
        return [str(exc)], [], {}

    paper_ids = {entry.get("paper_id") for entry in paper_index if entry.get("paper_id")}
    claim_ids = {entry.get("claim_id") for entry in claims if entry.get("claim_id")}
    topic_ids = {entry.get("topic_id") for entry in topics if entry.get("topic_id")}
    card_ids = {card.get("paper_id") for card in cards if card.get("paper_id")}
    card_map = {card.get("paper_id"): card for card in cards}
    claim_map = {claim.get("claim_id"): claim for claim in claims}
    manifest = load_json(REFERENCE_ROOT / "paper_origin" / "manifest.json")
    sources = manifest.get("papers", [])
    source_map = {s.get("paper_id"): s for s in sources}
    for value in duplicates([s.get("paper_id", "") for s in sources]):
        errors.append(f"duplicate source manifest ID: {value}")
    atom_ids = {atom.get("atom_id") for atom in atoms if atom.get("atom_id")}

    id_groups = {
        "paper index": [str(entry.get("paper_id", "")) for entry in paper_index],
        "claim": [str(entry.get("claim_id", "")) for entry in claims],
        "topic": [str(entry.get("topic_id", "")) for entry in topics],
        "card": [str(card.get("paper_id", "")) for card in cards],
        "atom": [str(atom.get("atom_id", "")) for atom in atoms],
    }
    for label, ids in id_groups.items():
        for repeated in sorted(duplicates(ids)):
            errors.append(f"duplicate {label} id: {repeated}")

    indexed_card_paths: set[str] = set()
    for entry in paper_index:
        missing = missing_keys(entry, {"paper_id", "title", "card_path", "review_status"})
        if missing:
            errors.append(
                f"paper index entry {entry.get('paper_id', '<missing>')} lacks {sorted(missing)}"
            )
            continue
        if entry["review_status"] not in VALID_REVIEW_STATUS:
            errors.append(f"invalid paper review_status for {entry['paper_id']}")
        if card_map.get(entry["paper_id"], {}).get("review_status") != entry["review_status"]:
            errors.append(f"index/card review status differs: {entry['paper_id']}")
        card_path = REFERENCE_ROOT / entry["card_path"]
        indexed_card_paths.add(card_path.resolve().as_posix())
        if not card_path.is_file():
            errors.append(f"indexed card does not exist: {entry['card_path']}")

    for path in sorted(CARD_DIR.glob("*.json")):
        card = load_json(path)
        label = card.get("paper_id", path.name)
        missing = missing_keys(card, CARD_REQUIRED)
        if missing:
            errors.append(f"card {label} lacks {sorted(missing)}")
            continue
        if path.stem != card["paper_id"]:
            errors.append(f"card filename/id mismatch: {path.name} vs {card['paper_id']}")
        if card["review_status"] not in VALID_REVIEW_STATUS:
            errors.append(f"invalid card review_status for {label}")
        if card["paper_id"] not in paper_ids:
            errors.append(f"card is absent from papers index: {label}")
        if path.resolve().as_posix() not in indexed_card_paths:
            errors.append(f"card path is absent from papers index: {path.name}")
        for source_key in ("source_pdf", "source_text"):
            source_path = resolve_card_source(path, card[source_key])
            if not source_path.is_file():
                message = f"missing {source_key} for {label}: {card[source_key]}"
                (errors if strict_sources else warnings).append(message)
        source = source_map.get(label)
        if not source or not all(source.get(k) for k in ("filename", "sha256", "source_url", "provenance", "access_note")):
            errors.append(f"card {label} lacks complete source manifest provenance")
        elif strict_sources:
            pdf_path = resolve_card_source(path, card["source_pdf"])
            if pdf_path.name != source["filename"]:
                errors.append(f"manifest/card filename differs: {label}")
            if pdf_path.is_file() and hashlib.sha256(pdf_path.read_bytes()).hexdigest() != source["sha256"]:
                errors.append(f"source PDF hash mismatch: {label}")

    if paper_ids != card_ids:
        for paper_id in sorted(paper_ids - card_ids):
            errors.append(f"paper index has no card: {paper_id}")
        for paper_id in sorted(card_ids - paper_ids):
            errors.append(f"card has no paper index entry: {paper_id}")

    evidence_ids_from_claims: set[str] = set()
    for claim in claims:
        missing = missing_keys(
            claim,
            {"claim_id", "statement", "evidence_ids", "review_status", "tags"},
        )
        if missing:
            errors.append(
                f"claim {claim.get('claim_id', '<missing>')} lacks {sorted(missing)}"
            )
            continue
        if claim["review_status"] not in VALID_REVIEW_STATUS:
            errors.append(f"invalid claim review_status for {claim['claim_id']}")
        evidence_ids_from_claims.update(claim["evidence_ids"])

    for atom in atoms:
        label = atom.get("atom_id", atom.get("_record_source", "<missing>"))
        missing = missing_keys(atom, ATOM_REQUIRED)
        if missing:
            errors.append(f"atom {label} lacks {sorted(missing)}")
            continue
        if atom["paper_id"] not in paper_ids:
            errors.append(f"atom {label} references unknown paper {atom['paper_id']}")
        if atom["claim_id"] not in claim_ids:
            errors.append(f"atom {label} references unknown claim {atom['claim_id']}")
        if atom["confidence"] not in VALID_CONFIDENCE:
            errors.append(f"invalid confidence for atom {label}")
        if atom["review_status"] not in VALID_REVIEW_STATUS:
            errors.append(f"invalid review_status for atom {label}")
        locator = atom.get("source_locator", {})
        if not locator.get("page") or not locator.get("section"):
            errors.append(f"atom {label} lacks an exact page/section locator")
        claim = claim_map.get(atom["claim_id"], {})
        if label not in claim.get("evidence_ids", []):
            errors.append(f"atom {label} not indexed by its own claim")
        if atom["review_status"] == "reviewed":
            if card_map.get(atom["paper_id"], {}).get("review_status") != "reviewed":
                errors.append(f"reviewed atom {label} has non-reviewed paper")
            if claim.get("review_status") != "reviewed":
                errors.append(f"reviewed atom {label} has non-reviewed claim")

    for missing_atom_id in sorted(evidence_ids_from_claims - atom_ids):
        errors.append(f"claim index references unknown atom: {missing_atom_id}")
    for unindexed_atom_id in sorted(atom_ids - evidence_ids_from_claims):
        errors.append(f"evidence atom is absent from claims index: {unindexed_atom_id}")
    atom_map = {a["atom_id"]: a for a in atoms if "atom_id" in a}
    for claim in claims:
        for atom_id in claim.get("evidence_ids", []):
            if atom_id in atom_map and atom_map[atom_id].get("claim_id") != claim.get("claim_id"):
                errors.append(f"claim {claim['claim_id']} references atom assigned to another claim: {atom_id}")

    for topic in topics:
        label = topic.get("topic_id", "<missing>")
        missing = missing_keys(topic, TOPIC_REQUIRED)
        if missing:
            errors.append(f"topic {label} lacks {sorted(missing)}")
            continue
        for paper_id in topic["paper_ids"]:
            if paper_id not in paper_ids:
                errors.append(f"topic {label} references unknown paper {paper_id}")
        for claim_id in topic["claim_ids"]:
            if claim_id not in claim_ids:
                errors.append(f"topic {label} references unknown claim {claim_id}")
        for fallback_id in topic["fallback_topics"]:
            if fallback_id not in topic_ids:
                errors.append(f"topic {label} references unknown fallback {fallback_id}")

    counts = {
        "papers": len(paper_index),
        "cards": len(cards),
        "claims": len(claims),
        "topics": len(topics),
        "atoms": len(atoms),
        "reviewed_atoms": sum(
            atom.get("review_status") == "reviewed"
            and atom.get("confidence") in {"direct", "qualified"}
            for atom in atoms
        ),
    }
    from coverage import validate_facets
    try:
        errors.extend(validate_facets(paper_ids))
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"cannot validate facets/queue: {exc}")
    return errors, warnings, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-sources",
        action="store_true",
        help="Treat a missing local PDF or extracted-text source as an error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors, warnings, counts = validate(strict_sources=args.strict_sources)
    if counts:
        rendered_counts = ", ".join(f"{key}={value}" for key, value in counts.items())
        print(f"REFERENCE: {rendered_counts}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print("STATUS: FAIL")
        return 1
    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
