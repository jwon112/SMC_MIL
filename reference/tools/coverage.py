"""Validated facets and coverage reports for the WSI evidence library."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "indexes" / name).read_text(encoding="utf-8"))


def validate_facets(paper_ids):
    errors = []
    taxonomy = load("taxonomy.json")["dimensions"]
    facets = load("paper_facets.json")["papers"]
    ids = [f.get("paper_id") for f in facets]
    if len(ids) != len(set(ids)):
        errors.append("duplicate paper facet ID")
    if set(ids) != set(paper_ids):
        errors.append("facet/index paper IDs differ")
    for facet in facets:
        for dimension, spec in taxonomy.items():
            values = facet.get(dimension)
            if not isinstance(values, list) or not values:
                errors.append(f"{facet.get('paper_id')}: missing facet {dimension}")
            elif set(values) - set(spec["values"]):
                errors.append(f"{facet.get('paper_id')}: invalid {dimension}: {values}")
        for key in ("review_scope", "source_version", "reviewed_on"):
            if not facet.get(key):
                errors.append(f"{facet.get('paper_id')}: missing {key}")
    queue = load("review_queue.json")["candidates"]
    queue_ids = [q.get("paper_id") for q in queue]
    if len(queue_ids) != len(set(queue_ids)):
        errors.append("duplicate review queue ID")
    for entry in queue:
        if entry.get("paper_id") in paper_ids:
            errors.append(f"already registered paper remains in queue: {entry['paper_id']}")
        for key in ("paper_id", "title", "source_url", "review_question", "next_action", "source_status"):
            if not entry.get(key):
                errors.append(f"queue {entry.get('paper_id')}: missing {key}")
        if entry.get("priority") not in (1, 2, 3):
            errors.append(f"queue {entry.get('paper_id')}: invalid priority")
        if entry.get("review_status") != "pending":
            errors.append(f"queue {entry.get('paper_id')}: only pending candidates belong here")
        if not entry.get("source_url", "").startswith("https://"):
            errors.append(f"queue {entry.get('paper_id')}: invalid URL")
    return errors


def build_coverage():
    papers = load("papers.json")["papers"]
    errors = validate_facets({p["paper_id"] for p in papers})
    if errors:
        raise ValueError("; ".join(errors))
    taxonomy = load("taxonomy.json")["dimensions"]
    facets = load("paper_facets.json")["papers"]
    queue = load("review_queue.json")["candidates"]
    atoms = [json.loads(line) for p in (ROOT / "evidence").glob("*.jsonl")
             for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    reviewed = [a for a in atoms if a.get("review_status") == "reviewed"
                and a.get("confidence") in {"direct", "qualified"}]
    lines = ["# WSI literature coverage", "", "Generated from indexes; rerun `python reference/tools/coverage.py`.", "",
             f"Registered papers: {len(papers)}; reviewed atoms: {len(reviewed)}; pending candidates: {len(queue)}.", "",
             "Counts describe this curated library, not completeness of the published literature.",
             "A reviewed atom supports its bounded statement; a relevant method paper is not direct heart-transplant clinical validation.", ""]
    for dimension, spec in taxonomy.items():
        lines += [f"## {spec['label']}", "", "| Facet | Papers | Reviewed atoms |", "|---|---:|---:|"]
        for value in spec["values"]:
            selected = {f["paper_id"] for f in facets if value in f[dimension]}
            count = sum(a["paper_id"] in selected for a in reviewed)
            lines.append(f"| {value} | {len(selected)} | {count} |")
        lines.append("")
    topics = load("topics.json")["topics"]
    lines += ["## Topic coverage", "", "| Topic | Routed papers | Reviewed routed atoms |", "|---|---:|---:|"]
    for topic in topics:
        count = sum(a["claim_id"] in topic["claim_ids"] for a in reviewed)
        lines.append(f"| {topic['topic_id']} | {len(topic['paper_ids'])} | {count} |")
    lines += ["", "## Registered paper comparison", "", "| Paper | Year | Organ | Units | Methods | Validation |", "|---|---:|---|---|---|---|"]
    facet_map = {f["paper_id"]: f for f in facets}
    for paper in sorted(papers, key=lambda p: (p.get("year", 0), p["paper_id"])):
        f = facet_map[paper["paper_id"]]
        values = [", ".join(f[k]) for k in ("organ_scope", "analysis_units", "methods", "validation")]
        lines.append(f"| [{paper['paper_id']}](../{paper['card_path']}) | {paper.get('year', '')} | " + " | ".join(values) + " |")
    lines += ["", "## Open research questions", ""]
    for gap in load("taxonomy.json")["research_gaps"]:
        lines.append(f"- **{gap['id']}**: {gap['question']} ({gap['status']})")
    lines += ["", "## Review queue", "", "| Priority | Candidate | Question / next action |", "|---|---|---|"]
    for entry in sorted(queue, key=lambda q: (q["priority"], q["paper_id"])):
        lines.append(f"| {entry['priority']} | [{entry['title']}]({entry['source_url']}) | {entry['review_question']} / {entry['next_action']} |")
    output = ROOT / "indexes" / "category_gaps.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"COVERAGE: {len(papers)} papers, {len(reviewed)} reviewed atoms, {len(queue)} candidates -> {output}")


if __name__ == "__main__":
    build_coverage()
