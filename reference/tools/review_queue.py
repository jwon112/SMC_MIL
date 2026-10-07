"""List pending candidates separately from reviewed evidence."""
import argparse
import json
import sys
from coverage import load

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--priority", choices=[1, 2, 3], type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [r for r in load("review_queue.json")["candidates"]
            if (args.priority is None or r["priority"] == args.priority)
            and (not args.query or args.query.lower() in json.dumps(r, ensure_ascii=False).lower())]
    rows.sort(key=lambda r: (r["priority"], r["paper_id"]))
    if args.json:
        print(json.dumps({"status": "pending_not_evidence", "candidates": rows}, ensure_ascii=False, indent=2))
    else:
        print(f"PENDING REVIEW: {len(rows)} candidates; not available as reviewed evidence")
        for r in rows:
            print(f"\nP{r['priority']} {r['paper_id']}: {r['title']}\n  source: {r['source_url']} ({r['source_status']})\n  question: {r['review_question']}\n  next: {r['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
