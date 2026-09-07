#!/usr/bin/env python3
"""Export a dependency-free, offline stain-review package for a web browser."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html as html_module
import json
from pathlib import Path
import re
import shutil

import pandas as pd
from PIL import Image


DECISION_COLUMNS = [
    "slide_id", "stain_group", "stain_raw", "stain_confidence",
    "stain_note", "reviewed_at",
]
STAIN_GROUPS = {"HE", "IHC", "special_other", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="slide_curation_manifest.csv or a stain-review queue CSV.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decisions", type=Path, default=None,
        help="Optional existing stain_slide_review.csv to preload.",
    )
    parser.add_argument(
        "--selection", choices=["unknown", "filename-unresolved", "all"], default="unknown",
        help=(
            "Export final unknown slides (default), slides not resolved by the "
            "original filename rules, or every manifest row."
        ),
    )
    parser.add_argument(
        "--only-signature", action="append", default=[], metavar="SIGNATURE",
        help="Export only an exact stain signature; repeat for multiple signatures.",
    )
    parser.add_argument(
        "--exclude-signature", action="append", default=[], metavar="SIGNATURE",
        help="Exclude an exact stain signature; repeat for multiple signatures.",
    )
    parser.add_argument(
        "--signature-contains", action="append", default=[], metavar="TEXT",
        help="Export signatures containing text (case-insensitive); repeat for OR matching.",
    )
    parser.add_argument(
        "--exclude-signature-contains", action="append", default=[], metavar="TEXT",
        help="Exclude signatures containing text (case-insensitive); repeat for OR matching.",
    )
    parser.add_argument("--title", default="WSI stain review")
    parser.add_argument("--thumbnail-max-px", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str).fillna("")


def safe_filename(rank: int, slide_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", slide_id).strip("._")
    if not safe_id:
        safe_id = "slide"
    return f"{rank:04d}__{safe_id[:140]}.jpg"


def resolve_thumbnail(raw_path: str, manifest_path: Path) -> Path:
    source = Path(raw_path)
    candidates = [source]
    if not source.is_absolute():
        candidates.extend([manifest_path.parent / source, Path.cwd() / source])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(raw_path)


def copy_thumbnail(source: Path, destination: Path, max_px: int, quality: int) -> None:
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((max_px, max_px))
            image.save(destination, format="JPEG", quality=quality, optimize=True)
    except Exception:
        if source.suffix.lower() in {".jpg", ".jpeg"}:
            shutil.copy2(source, destination)
        else:
            raise


def parse_reviewer_note(note: str) -> tuple[str, str]:
    match = re.match(r"^reviewer=([^;]*)(?:;\s*)?(.*)$", note, flags=re.DOTALL)
    if not match:
        return "", note
    return match.group(1).strip(), match.group(2).strip()


def prepare_items(args: argparse.Namespace) -> list[dict[str, str]]:
    manifest = read_csv(args.manifest)
    required = {"slide_id", "thumbnail_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest.slide_id.duplicated().any():
        raise ValueError("Manifest contains duplicate slide_id values")

    if args.selection == "unknown":
        if "stain_group" not in manifest:
            raise ValueError("Selection 'unknown' requires a stain_group column")
        manifest = manifest[manifest.stain_group.str.strip().isin(["", "unknown"])].copy()
    elif args.selection == "filename-unresolved":
        if "stain_group_auto" in manifest:
            manifest = manifest[
                manifest.stain_group_auto.str.strip().isin(["", "unknown"])
            ].copy()
        elif "stain_source_auto" in manifest:
            manifest = manifest[
                ~manifest.stain_source_auto.str.strip().eq("filename_rule")
            ].copy()
        else:
            raise ValueError(
                "Selection 'filename-unresolved' requires stain_group_auto or stain_source_auto"
            )
    else:
        manifest = manifest.copy()

    if (
        args.only_signature or args.exclude_signature
        or args.signature_contains or args.exclude_signature_contains
    ):
        if "stain_signature" not in manifest:
            raise ValueError("Signature filters require a stain_signature column")
        signatures = manifest.stain_signature.str.strip().str.upper()
        only = {value.strip().upper() for value in args.only_signature if value.strip()}
        excluded = {value.strip().upper() for value in args.exclude_signature if value.strip()}
        contains = {value.strip().upper() for value in args.signature_contains if value.strip()}
        excluded_contains = {
            value.strip().upper() for value in args.exclude_signature_contains if value.strip()
        }
        if only:
            manifest = manifest[signatures.isin(only)].copy()
            signatures = manifest.stain_signature.str.strip().str.upper()
        if contains:
            include_mask = pd.Series(False, index=manifest.index)
            for value in contains:
                include_mask |= signatures.str.contains(value, regex=False)
            manifest = manifest[include_mask].copy()
            signatures = manifest.stain_signature.str.strip().str.upper()
        if excluded:
            manifest = manifest[~signatures.isin(excluded)].copy()
            signatures = manifest.stain_signature.str.strip().str.upper()
        if excluded_contains:
            exclude_mask = pd.Series(False, index=manifest.index)
            for value in excluded_contains:
                exclude_mask |= signatures.str.contains(value, regex=False)
            manifest = manifest[~exclude_mask].copy()

    sort_columns = [
        column for column in ["stain_color_cluster", "stain_signature", "slide_id"]
        if column in manifest.columns
    ]
    manifest.sort_values(sort_columns, inplace=True, na_position="last")
    manifest.reset_index(drop=True, inplace=True)
    if manifest.empty:
        raise ValueError("No slides matched the requested selection")

    decisions: dict[str, dict[str, str]] = {}
    if args.decisions:
        existing = read_csv(args.decisions)
        required_decisions = {"slide_id", "stain_group"}
        missing = required_decisions.difference(existing.columns)
        if missing:
            raise ValueError(f"Decision CSV missing columns: {sorted(missing)}")
        if existing.slide_id.duplicated().any():
            raise ValueError("Decision CSV contains duplicate slide_id values")
        invalid = set(existing.stain_group).difference(STAIN_GROUPS | {""})
        if invalid:
            raise ValueError(f"Unsupported stain groups: {sorted(invalid)}")
        for column in DECISION_COLUMNS:
            if column not in existing:
                existing[column] = ""
        decisions = existing.set_index("slide_id")[DECISION_COLUMNS[1:]].to_dict("index")

    image_dir = args.output_dir / "thumbnails"
    image_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, str]] = []
    failures: list[str] = []
    for rank, row in enumerate(manifest.to_dict("records"), start=1):
        slide_id = str(row["slide_id"])
        filename = safe_filename(rank, slide_id)
        try:
            source = resolve_thumbnail(str(row["thumbnail_path"]), args.manifest)
            copy_thumbnail(
                source, image_dir / filename,
                max_px=args.thumbnail_max_px, quality=args.jpeg_quality,
            )
        except Exception as exc:
            failures.append(f"{slide_id}\t{type(exc).__name__}\t{row['thumbnail_path']}")
            continue

        decision = decisions.get(slide_id, {})
        reviewer, clean_note = parse_reviewer_note(decision.get("stain_note", ""))
        items.append(
            {
                "slide_id": slide_id,
                "image": f"thumbnails/{filename}",
                "stain_signature": str(row.get("stain_signature", "")),
                "stain_color_cluster": str(row.get("stain_color_cluster", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "slide_rel_path": str(row.get("slide_rel_path", "")),
                "stain_group": decision.get("stain_group", ""),
                "stain_raw": decision.get("stain_raw", ""),
                "stain_confidence": decision.get("stain_confidence", "medium") or "medium",
                "stain_note": clean_note,
                "reviewer": reviewer,
                "reviewed_at": decision.get("reviewed_at", ""),
            }
        )

    if failures:
        (args.output_dir / "thumbnail_failures.tsv").write_text(
            "slide_id\terror\tthumbnail_path\n" + "\n".join(failures) + "\n",
            encoding="utf-8",
        )
    if not items:
        raise RuntimeError("No readable thumbnails were exported")
    return items


def write_initial_csv(items: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        for item in items:
            note_parts = []
            if item["reviewer"]:
                note_parts.append(f"reviewer={item['reviewer']}")
            if item["stain_note"]:
                note_parts.append(item["stain_note"])
            writer.writerow(
                {
                    "slide_id": item["slide_id"],
                    "stain_group": item["stain_group"],
                    "stain_raw": item["stain_raw"],
                    "stain_confidence": item["stain_confidence"] if item["stain_group"] else "",
                    "stain_note": "; ".join(note_parts) if item["stain_group"] else "",
                    "reviewed_at": item["reviewed_at"] if item["stain_group"] else "",
                }
            )


HTML_TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f4f5f6; --panel: #ffffff; --line: #d8dcdf; --text: #202427;
      --muted: #687078; --accent: #176b87; --he: #b53f78; --ihc: #8a5a18;
      --special: #247c68; --unknown: #626a72;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Arial, "Malgun Gothic", sans-serif; letter-spacing: 0; }
    button, input, select, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    .app { height: 100vh; min-height: 700px; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; }
    .topbar { min-height: 64px; padding: 10px 18px; background: #252a2e; color: white; display: flex; align-items: center; gap: 16px; }
    .title { font-size: 19px; font-weight: 700; white-space: nowrap; }
    .progress-wrap { flex: 1; min-width: 180px; }
    .progress-line { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 5px; color: #e3e7e9; }
    .progress { height: 7px; overflow: hidden; background: #4b5258; }
    .progress > div { height: 100%; width: 0; background: #4fb2cd; transition: width .15s ease; }
    .top-actions { display: flex; gap: 8px; }
    .top-actions button, .file-label { border: 1px solid #737b81; background: #343a3f; color: white; padding: 8px 11px; }
    .file-label { cursor: pointer; display: inline-flex; align-items: center; }
    .file-label input { display: none; }
    main { min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 340px; }
    .viewer { min-height: 0; padding: 14px; display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 10px; }
    .meta { background: var(--panel); border: 1px solid var(--line); padding: 10px 12px; display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .slide-id { font-weight: 700; overflow-wrap: anywhere; }
    .meta-sub { margin-top: 5px; color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }
    .position { display: flex; align-items: center; gap: 6px; }
    .position input { width: 72px; padding: 6px; border: 1px solid var(--line); }
    .stage { position: relative; overflow: auto; display: flex; align-items: center; justify-content: center; background: #e7e9ea; border: 1px solid #cbd0d3; }
    .stage img { display: block; max-width: 100%; max-height: calc(100vh - 190px); object-fit: contain; transform-origin: center; transition: transform .1s ease; }
    .zoom-tools { position: absolute; right: 10px; top: 10px; z-index: 2; display: flex; gap: 5px; }
    .zoom-tools button { width: 36px; height: 34px; border: 1px solid #8a9196; background: rgba(255,255,255,.94); font-weight: 700; }
    aside { min-height: 0; overflow-y: auto; padding: 14px 14px 14px 0; }
    .form-section { background: var(--panel); border: 1px solid var(--line); padding: 13px; margin-bottom: 10px; }
    .section-title { font-size: 13px; font-weight: 700; color: var(--muted); margin-bottom: 10px; text-transform: uppercase; }
    label.field { display: block; margin-bottom: 11px; font-size: 13px; color: var(--muted); }
    .field input, .field select, .field textarea { width: 100%; margin-top: 5px; border: 1px solid #bfc5c9; background: white; padding: 8px; color: var(--text); }
    .field textarea { min-height: 82px; resize: vertical; }
    .decision-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .decision { min-height: 47px; border: 2px solid transparent; color: white; font-weight: 700; }
    .decision[data-group="HE"] { background: var(--he); }
    .decision[data-group="IHC"] { background: var(--ihc); }
    .decision[data-group="special_other"] { background: var(--special); }
    .decision[data-group="unknown"] { background: var(--unknown); }
    .decision.selected { outline: 3px solid #16191b; outline-offset: 2px; }
    .counts { display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; font-size: 13px; }
    .count-row { display: flex; justify-content: space-between; border-bottom: 1px solid #eceeef; padding: 5px 0; }
    .bottom { min-height: 54px; padding: 8px 18px; background: white; border-top: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .nav { display: flex; gap: 8px; }
    .nav button { min-width: 98px; padding: 8px 12px; border: 1px solid #aeb4b8; background: white; }
    .nav .primary { background: var(--accent); border-color: var(--accent); color: white; }
    .status { font-size: 13px; color: var(--muted); text-align: right; }
    @media (max-width: 900px) {
      .app { height: auto; min-height: 100vh; }
      main { grid-template-columns: 1fr; }
      .viewer { grid-template-rows: auto minmax(420px, 62vh); }
      aside { padding: 0 14px 14px; }
      .stage img { max-height: 62vh; }
      .topbar { flex-wrap: wrap; }
      .top-actions { width: 100%; }
    }
  </style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="title">__TITLE__</div>
    <div class="progress-wrap">
      <div class="progress-line"><span id="progressText"></span><span id="pendingText"></span></div>
      <div class="progress"><div id="progressBar"></div></div>
    </div>
    <div class="top-actions">
      <label class="file-label" title="기존 stain_review_results.csv를 불러옵니다">CSV 불러오기<input id="csvInput" type="file" accept=".csv,text/csv"></label>
      <button id="exportButton" title="현재 판정 결과를 CSV 파일로 저장합니다">CSV 저장</button>
    </div>
  </header>

  <main>
    <section class="viewer">
      <div class="meta">
        <div>
          <div class="slide-id" id="slideId"></div>
          <div class="meta-sub" id="metadata"></div>
        </div>
        <label class="position">슬라이드 <input id="positionInput" type="number" min="1"></label>
      </div>
      <div class="stage" id="stage">
        <div class="zoom-tools">
          <button id="zoomOut" title="축소">−</button>
          <button id="zoomReset" title="화면에 맞춤">↺</button>
          <button id="zoomIn" title="확대">+</button>
        </div>
        <img id="slideImage" alt="검토할 슬라이드 썸네일">
      </div>
    </section>

    <aside>
      <div class="form-section">
        <div class="section-title">판정</div>
        <div class="decision-grid">
          <button class="decision" data-group="HE" title="단축키 1">H&amp;E</button>
          <button class="decision" data-group="IHC" title="단축키 2">IHC</button>
          <button class="decision" data-group="special_other" title="단축키 3">Special stain</button>
          <button class="decision" data-group="unknown" title="단축키 4">보류</button>
        </div>
      </div>
      <div class="form-section">
        <div class="section-title">세부 기록</div>
        <label class="field">검토자<input id="reviewerInput" autocomplete="name"></label>
        <label class="field">세부 염색명<input id="rawInput" placeholder="예: Masson trichrome"></label>
        <label class="field">확신도<select id="confidenceInput"><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select></label>
        <label class="field">메모<textarea id="noteInput"></textarea></label>
      </div>
      <div class="form-section">
        <div class="section-title">분류 현황</div>
        <div class="counts" id="counts"></div>
      </div>
    </aside>
  </main>

  <footer class="bottom">
    <div class="nav">
      <button id="previousButton">← 이전</button>
      <button id="skipButton">다음 →</button>
      <button id="clearButton">판정 지우기</button>
      <button class="primary" id="pendingButton">다음 미분류</button>
    </div>
    <div class="status" id="status">브라우저에 자동 저장됨</div>
  </footer>
</div>

<script>
const packageId = __PACKAGE_ID__;
const items = __ITEMS_JSON__;
const validGroups = new Set(["HE", "IHC", "special_other", "unknown"]);
const storageKey = `stain-review-${packageId}`;
let index = 0;
let zoom = 1;

const el = id => document.getElementById(id);

function loadLocalState() {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (!saved || !saved.decisions) return;
    for (const item of items) {
      if (saved.decisions[item.slide_id]) Object.assign(item, saved.decisions[item.slide_id]);
    }
    index = Math.max(0, Math.min(items.length - 1, Number(saved.index) || 0));
  } catch (_) {}
}

function saveLocalState(message = "브라우저에 자동 저장됨") {
  const decisions = {};
  for (const item of items) {
    if (item.stain_group) {
      decisions[item.slide_id] = {
        stain_group: item.stain_group, stain_raw: item.stain_raw,
        stain_confidence: item.stain_confidence, stain_note: item.stain_note,
        reviewer: item.reviewer, reviewed_at: item.reviewed_at
      };
    }
  }
  localStorage.setItem(storageKey, JSON.stringify({index, decisions}));
  el("status").textContent = message;
}

function collectForm() {
  const item = items[index];
  item.reviewer = el("reviewerInput").value.trim();
  item.stain_raw = el("rawInput").value.trim();
  item.stain_confidence = el("confidenceInput").value;
  item.stain_note = el("noteInput").value.trim();
}

function setDecision(group) {
  collectForm();
  const item = items[index];
  item.stain_group = group;
  item.reviewed_at = new Date().toISOString();
  saveLocalState(`${item.slide_id}: ${group} 저장됨`);
  const next = findPending(index + 1);
  if (next !== -1) index = next;
  else if (index < items.length - 1) index += 1;
  render();
}

function findPending(start = 0) {
  for (let i = start; i < items.length; i++) if (!items[i].stain_group) return i;
  for (let i = 0; i < start; i++) if (!items[i].stain_group) return i;
  return -1;
}

function move(delta) {
  collectForm();
  saveLocalState();
  index = Math.max(0, Math.min(items.length - 1, index + delta));
  render();
}

function resetZoom() {
  zoom = 1;
  el("slideImage").style.transform = "scale(1)";
}

function renderCounts() {
  const counts = {HE: 0, IHC: 0, special_other: 0, unknown: 0};
  for (const item of items) if (counts[item.stain_group] !== undefined) counts[item.stain_group]++;
  const reviewed = Object.values(counts).reduce((a, b) => a + b, 0);
  el("progressText").textContent = `${reviewed} / ${items.length} 판정 완료`;
  el("pendingText").textContent = `미분류 ${items.length - reviewed}`;
  el("progressBar").style.width = `${items.length ? reviewed / items.length * 100 : 0}%`;
  const labels = [["H&E", counts.HE], ["IHC", counts.IHC], ["Special", counts.special_other], ["보류", counts.unknown]];
  el("counts").replaceChildren(...labels.map(([label, count]) => {
    const row = document.createElement("div"); row.className = "count-row";
    const name = document.createElement("span"); name.textContent = label;
    const value = document.createElement("strong"); value.textContent = count;
    row.append(name, value); return row;
  }));
}

function render() {
  const item = items[index];
  el("slideId").textContent = item.slide_id;
  const details = [
    item.stain_signature && `Signature: ${item.stain_signature}`,
    item.stain_color_cluster && `Cluster: ${item.stain_color_cluster}`,
    item.source_dataset && `Dataset: ${item.source_dataset}`,
    item.slide_rel_path && `Path: ${item.slide_rel_path}`
  ].filter(Boolean);
  el("metadata").textContent = details.join("  |  ");
  el("positionInput").max = items.length;
  el("positionInput").value = index + 1;
  el("slideImage").src = item.image;
  el("reviewerInput").value = item.reviewer || items.find(x => x.reviewer)?.reviewer || "";
  el("rawInput").value = item.stain_raw || "";
  el("confidenceInput").value = item.stain_confidence || "medium";
  el("noteInput").value = item.stain_note || "";
  document.querySelectorAll(".decision").forEach(button => {
    button.classList.toggle("selected", button.dataset.group === item.stain_group);
  });
  resetZoom();
  renderCounts();
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportCsv() {
  collectForm(); saveLocalState("CSV 저장 준비 완료");
  const header = ["slide_id", "stain_group", "stain_raw", "stain_confidence", "stain_note", "reviewed_at"];
  const rows = items.map(item => {
    const note = [item.reviewer && `reviewer=${item.reviewer}`, item.stain_note].filter(Boolean).join("; ");
    return [item.slide_id, item.stain_group, item.stain_raw,
      item.stain_group ? item.stain_confidence : "", item.stain_group ? note : "",
      item.stain_group ? item.reviewed_at : ""].map(csvEscape).join(",");
  });
  const blob = new Blob(["\ufeff" + [header.join(","), ...rows].join("\r\n")], {type: "text/csv;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "stain_review_results.csv";
  document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(link.href);
  el("status").textContent = "stain_review_results.csv 저장됨";
}

function parseCsv(text) {
  const rows = []; let row = []; let field = ""; let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"') quoted = true;
    else if (char === ",") { row.push(field); field = ""; }
    else if (char === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += char;
  }
  if (field || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows;
}

async function importCsv(file) {
  const rows = parseCsv((await file.text()).replace(/^\ufeff/, ""));
  if (!rows.length) return;
  const header = rows[0]; const indices = Object.fromEntries(header.map((name, i) => [name, i]));
  if (indices.slide_id === undefined || indices.stain_group === undefined) {
    alert("slide_id와 stain_group 열이 필요합니다."); return;
  }
  const byId = new Map(items.map(item => [item.slide_id, item])); let loaded = 0;
  for (const row of rows.slice(1)) {
    const item = byId.get(row[indices.slide_id]); if (!item) continue;
    const group = row[indices.stain_group] || ""; if (group && !validGroups.has(group)) continue;
    item.stain_group = group;
    item.stain_raw = row[indices.stain_raw] || "";
    item.stain_confidence = row[indices.stain_confidence] || "medium";
    let note = row[indices.stain_note] || "";
    const match = note.match(/^reviewer=([^;]*)(?:;\s*)?(.*)$/s);
    item.reviewer = match ? match[1].trim() : item.reviewer;
    item.stain_note = match ? match[2].trim() : note;
    item.reviewed_at = row[indices.reviewed_at] || "";
    if (group) loaded++;
  }
  const pending = findPending(0); if (pending !== -1) index = pending;
  saveLocalState(`${loaded}개 판정 불러옴`); render();
}

document.querySelectorAll(".decision").forEach(button => button.addEventListener("click", () => setDecision(button.dataset.group)));
el("previousButton").addEventListener("click", () => move(-1));
el("skipButton").addEventListener("click", () => move(1));
el("pendingButton").addEventListener("click", () => { collectForm(); const next = findPending(index + 1); if (next !== -1) index = next; saveLocalState(); render(); });
el("clearButton").addEventListener("click", () => { const item = items[index]; item.stain_group = ""; item.stain_raw = ""; item.stain_note = ""; item.reviewed_at = ""; saveLocalState("현재 판정 지움"); render(); });
el("positionInput").addEventListener("change", event => { collectForm(); index = Math.max(0, Math.min(items.length - 1, Number(event.target.value) - 1)); saveLocalState(); render(); });
el("exportButton").addEventListener("click", exportCsv);
el("csvInput").addEventListener("change", event => { if (event.target.files[0]) importCsv(event.target.files[0]); event.target.value = ""; });
el("zoomIn").addEventListener("click", () => { zoom = Math.min(4, zoom + .25); el("slideImage").style.transform = `scale(${zoom})`; });
el("zoomOut").addEventListener("click", () => { zoom = Math.max(.5, zoom - .25); el("slideImage").style.transform = `scale(${zoom})`; });
el("zoomReset").addEventListener("click", resetZoom);
document.addEventListener("keydown", event => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (event.key === "ArrowLeft") move(-1);
  else if (event.key === "ArrowRight") move(1);
  else if ({"1":"HE", "2":"IHC", "3":"special_other", "4":"unknown"}[event.key]) setDecision({"1":"HE", "2":"IHC", "3":"special_other", "4":"unknown"}[event.key]);
});
window.addEventListener("beforeunload", () => { collectForm(); saveLocalState(); });

loadLocalState();
const firstPending = findPending(0); if (firstPending !== -1) index = firstPending;
render();
</script>
</body>
</html>
'''


def write_html(items: list[dict[str, str]], title: str, output_path: Path) -> None:
    package_hash = hashlib.sha256(
        "\n".join(item["slide_id"] for item in items).encode("utf-8")
    ).hexdigest()[:16]
    item_json = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    safe_title = html_module.escape(title)
    html = (
        HTML_TEMPLATE
        .replace("__TITLE__", safe_title)
        .replace("__PACKAGE_ID__", json.dumps(package_hash))
        .replace("__ITEMS_JSON__", item_json)
    )
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. Use --overwrite to replace the package."
            )
        for filename in ["index.html", "stain_review_results.csv", "thumbnail_failures.tsv", "package_created_at.txt"]:
            path = args.output_dir / filename
            if path.is_file():
                path.unlink()
        thumbnail_dir = args.output_dir / "thumbnails"
        if thumbnail_dir.is_dir():
            shutil.rmtree(thumbnail_dir)
        if any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to remove unrelated files from output directory: {args.output_dir}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = prepare_items(args)
    write_initial_csv(items, args.output_dir / "stain_review_results.csv")
    write_html(items, args.title, args.output_dir / "index.html")
    (args.output_dir / "package_created_at.txt").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="ascii"
    )

    failures = args.output_dir / "thumbnail_failures.tsv"
    print(f"[OK] Portable review package: {args.output_dir}")
    print(f"Slides exported: {len(items)}")
    print(f"Open: {args.output_dir / 'index.html'}")
    print(f"Initial/results CSV: {args.output_dir / 'stain_review_results.csv'}")
    if failures.is_file():
        print(f"[WARN] Some thumbnails failed: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
