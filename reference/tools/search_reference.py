#!/usr/bin/env python3
"""Search reviewed literature evidence using transparent topic routing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REFERENCE_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REFERENCE_ROOT / "indexes"
CARD_DIR = REFERENCE_ROOT / "cards"
EVIDENCE_DIR = REFERENCE_ROOT / "evidence"
REVIEWED_CONFIDENCE = {"direct", "qualified"}
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_atoms() -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for path in sorted(EVIDENCE_DIR.glob("*.jsonl")):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                atoms.append(json.loads(raw_line))
    return atoms


def normalize(text: str) -> str:
    # Unicode-aware alphanumeric tokens keep Korean and English queries on the
    # same transparent lexical path without hard-coding a language range.
    return " ".join(re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE))


def tokens(text: str) -> set[str]:
    return {token for token in normalize(text).split() if len(token) > 1}


def phrase_score(query: str, phrase: str) -> int:
    normalized_query = normalize(query)
    normalized_phrase = normalize(phrase)
    if not normalized_phrase:
        return 0
    if normalized_phrase in normalized_query:
        return 3 + len(normalized_phrase.split())
    return len(tokens(normalized_query) & tokens(normalized_phrase))


def select_topics(query: str, topics: list[dict[str, Any]], top_k: int = 3):
    ranked: list[tuple[int, dict[str, Any]]] = []
    query_tokens = tokens(query)
    for topic in topics:
        score = len(query_tokens & tokens(topic["description"]))
        score += sum(phrase_score(query, term) for term in topic["query_terms"])
        if score:
            ranked.append((score, topic))
    ranked.sort(key=lambda item: (-item[0], item[1]["topic_id"]))
    return ranked[:top_k]


def search(query: str, include_drafts: bool, top_k: int, filters=None):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    papers = load_json(INDEX_DIR / "papers.json").get("papers", [])
    claims = load_json(INDEX_DIR / "claims.json").get("claims", [])
    topics = load_json(INDEX_DIR / "topics.json").get("topics", [])
    atoms = load_atoms()
    cards = {
        card["paper_id"]: card
        for card in (load_json(path) for path in sorted(CARD_DIR.glob("*.json")))
    }
    paper_by_id = {paper["paper_id"]: paper for paper in papers}
    claim_by_id = {claim["claim_id"]: claim for claim in claims}
    facets = {f["paper_id"]: f for f in load_json(INDEX_DIR / "paper_facets.json")["papers"]}
    filters = filters or {}

    selected_topics = select_topics(query, topics)
    paper_route_scores: dict[str, int] = {}
    claim_route_scores: dict[str, int] = {}
    for route_score, topic in selected_topics:
        for paper_id in topic.get("paper_ids", []):
            paper_route_scores[paper_id] = (
                paper_route_scores.get(paper_id, 0) + route_score
            )
        for claim_id in topic.get("claim_ids", []):
            claim_route_scores[claim_id] = (
                claim_route_scores.get(claim_id, 0) + route_score
            )
    preferred_types = {
        evidence_type
        for _, topic in selected_topics
        for evidence_type in topic.get("preferred_evidence_types", [])
    }
    query_tokens = tokens(query)
    ranked_atoms: list[tuple[int, dict[str, Any]]] = []

    for atom in atoms:
        facet = facets.get(atom.get("paper_id"), {})
        if any(value not in facet.get(dimension, []) for dimension, value in filters.items()):
            continue
        if not include_drafts and not (
            atom.get("review_status") == "reviewed"
            and atom.get("confidence") in REVIEWED_CONFIDENCE
            and cards.get(atom.get("paper_id"), {}).get("review_status") == "reviewed"
            and paper_by_id.get(atom.get("paper_id"), {}).get("review_status") == "reviewed"
            and claim_by_id.get(atom.get("claim_id"), {}).get("review_status") == "reviewed"
        ):
            continue
        text = " ".join(
            [
                atom.get("statement", ""),
                atom.get("supports", ""),
                atom.get("does_not_establish", ""),
                atom.get("method_or_result", ""),
                atom.get("evidence_type", ""),
                " ".join(atom.get("tags", [])),
                cards.get(atom.get("paper_id"), {}).get("title", ""),
            ]
        )
        lexical_score = len(query_tokens & tokens(text))
        score = lexical_score
        # Evidence shared by multiple matched topics receives cumulative route
        # weight (for example, cardiac pathology plus external validation).
        score += claim_route_scores.get(atom.get("claim_id", ""), 0)
        score += paper_route_scores.get(atom.get("paper_id", ""), 0)
        if atom.get("evidence_type") in preferred_types:
            score += 2
        if score and (selected_topics or lexical_score):
            ranked_atoms.append((score, atom))

    ranked_atoms.sort(key=lambda item: (-item[0], item[1]["atom_id"]))
    # A comparison query must not be filled entirely by one selected topic.
    diverse = []
    seen = set()
    for _, topic in selected_topics:
        for item in ranked_atoms:
            if item[1]["atom_id"] not in seen and item[1]["claim_id"] in topic["claim_ids"]:
                diverse.append(item)
                seen.add(item[1]["atom_id"])
                break
        if len(diverse) == top_k:
            break
    for item in ranked_atoms:
        if len(diverse) >= top_k:
            break
        if item[1]["atom_id"] not in seen:
            diverse.append(item)
            seen.add(item[1]["atom_id"])
    return selected_topics, diverse, cards, paper_by_id, claim_by_id


def render(query: str, result) -> None:
    selected_topics, ranked_atoms, cards, paper_by_id, claim_by_id = result
    print(f"QUERY: {query}")
    if selected_topics:
        print(
            "TOPICS: "
            + ", ".join(
                f"{topic['topic_id']}({score})" for score, topic in selected_topics
            )
        )
    else:
        print("TOPICS: no configured route matched")
    artifacts = sorted({p for _, t in selected_topics for p in t.get("linked_local_artifacts", [])})
    if artifacts:
        print("LOCAL_ARTIFACTS: " + ", ".join(artifacts))

    if not ranked_atoms:
        print("RESULT: no matching reviewed evidence atoms")
        return

    for score, atom in ranked_atoms:
        paper = cards.get(atom["paper_id"], paper_by_id.get(atom["paper_id"], {}))
        claim = claim_by_id.get(atom["claim_id"], {})
        locator = atom["source_locator"]
        print(f"\n[{score:02d}] {paper.get('title', atom['paper_id'])}")
        print(f"atom: {atom['atom_id']} | confidence: {atom['confidence']}")
        print(f"card: cards/{atom['paper_id']}.json | source: {paper.get('source_text', '')}")
        print(f"claim: {claim.get('statement', atom['claim_id'])}")
        print(f"evidence: {atom['statement']}")
        print(f"supports: {atom['supports']}")
        print(f"does not establish: {atom['does_not_establish']}")
        print(
            "locator: "
            f"{locator['page']}, {locator['section']}, "
            f"lines {locator.get('extracted_text_lines', 'not recorded')}"
        )


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language literature question")
    parser.add_argument("--top-k", type=positive_int, default=5)
    taxonomy = load_json(INDEX_DIR / "taxonomy.json")["dimensions"]
    for flag, dimension in {"organ": "organ_scope", "unit": "analysis_units", "stain": "stains",
                            "method": "methods", "validation": "validation", "focus": "research_focus"}.items():
        parser.add_argument("--" + flag, choices=taxonomy[dimension]["values"])
    parser.add_argument("--json", action="store_true", help="Emit a clean JSON retrieval bundle")
    parser.add_argument(
        "--include-drafts",
        action="store_true",
        help="Include draft, unresolved, and needs-update evidence for discovery only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from validate_reference import validate
    errors, _, _ = validate()
    if errors:
        print("Invalid literature index: " + "; ".join(errors), file=sys.stderr)
        return 1
    mapping = {"organ": "organ_scope", "unit": "analysis_units", "stain": "stains",
               "method": "methods", "validation": "validation", "focus": "research_focus"}
    filters = {dim: getattr(args, flag) for flag, dim in mapping.items() if getattr(args, flag)}
    result = search(args.query, args.include_drafts, args.top_k, filters)
    if args.json:
        topics, ranked, cards, papers, claims = result
        bundle = {"query": args.query, "filters": filters,
                  "topics": [t["topic_id"] for _, t in topics],
                  "linked_local_artifacts": sorted({p for _, t in topics for p in t.get("linked_local_artifacts", [])}),
                  "evidence": [{"score": score, **atom, "paper_title": cards[atom["paper_id"]]["title"],
                                "card_path": papers[atom["paper_id"]]["card_path"],
                                "source_text": cards[atom["paper_id"]]["source_text"]} for score, atom in ranked]}
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
    else:
        render(args.query, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
