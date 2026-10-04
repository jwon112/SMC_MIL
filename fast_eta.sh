#!/usr/bin/env bash
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd /home/jupyter/image_team/projects/SMC_MIL
/home/jupyter/image_team/miniconda3/envs/clam_latest/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import os
import re
import time
import h5py
from run_wsi_pathomics import load_config, cohort_and_jobs

cfg = load_config("pathomics_all_wsi.json")
root = Path(cfg["output"]) / "full"
files = list(root.glob("parallel_execution_*.json"))
if not files:
    raise SystemExit("No parallel execution receipt found.")
run = json.loads(max(files, key=lambda p: p.stat().st_mtime).read_text())
started = datetime.fromisoformat(run["started_utc"]).timestamp()
logpath = Path("pathomics_all_wsi_fast.log")
text = logpath.read_text(errors="replace")
if "Parallel preflight:" not in text:
    raise SystemExit("Preflight missing; confirm log filename.")
text = text.split("Parallel preflight:")[-1]
snapshot = time.time()
elapsed = snapshot - started
header = re.search(r"eligible=(\d+), unavailable=(\d+), workers=(\d+)", text)
if not header or int(header[1]) != run["eligible"] or int(header[3]) != run["workers"] or logpath.stat().st_mtime < started:
    raise SystemExit("Log/receipt mismatch; no ETA calculated.")
if run.get("status") != "running":
    raise SystemExit("Execution is not marked running: " + str(run.get("status")))
try:
    os.kill(int(run["pid"]), 0)
except ProcessLookupError:
    raise SystemExit("Recorded parent is not running; no live ETA.")
except PermissionError:
    print("WARNING: cannot verify parent process permissions.")

cmdline = Path("/proc") / str(run["pid"]) / "cmdline"
if cmdline.exists():
    args = cmdline.read_bytes().replace(bytes([0]), b" ").decode(errors="replace")
    if "run_pathomics_fast.py" not in args or " run " not in args:
        raise SystemExit("PID is not the fast extraction runner; no live ETA.")
print("Log age: %.1f minutes" % ((snapshot-logpath.stat().st_mtime)/60), flush=True)
if elapsed < 3600:
    print("LOW CONFIDENCE: less than one hour since start.", flush=True)
if snapshot-logpath.stat().st_mtime > 600:
    print("WARNING: log has not updated for over 10 minutes; check job health.", flush=True)

progress = {}
completed = {}
reused = {}
failed = set()
for line in text.splitlines():
    m = re.match(r"\[([^\]]+)\] patches (\d+)/(\d+), QC (\d+), elapsed ([\d.]+)s", line)
    if m:
        progress[m[1]] = (int(m[2]), int(m[3]), float(m[5]))
    m = re.match(r"\[(completed|reused)\] ([^:]+): (\d+) patches", line)
    if m:
        (completed if m[1] == "completed" else reused)[m[2]] = int(m[3])
        failed.discard(m[2])
    m = re.match(r"\[FAILED\] ([^:]+):", line)
    if m:
        failed.add(m[1])
active = {sid: v for sid, v in progress.items() if sid not in completed and sid not in reused and sid not in failed}
processed = sum(completed.values()) + sum(v[0] for v in active.values())
if elapsed <= 0 or not processed:
    raise SystemExit("Not enough patch progress yet.")
print("Snapshot elapsed hours: %.2f" % (elapsed/3600), flush=True)
print("Completed / reused / failed slides:", len(completed), len(reused), len(failed), flush=True)
print("In-progress slides with patch logs:", len(active), flush=True)
print("New completed patches:", sum(completed.values()), flush=True)
print("In-progress patches already processed:", sum(v[0] for v in active.values()), flush=True)
rate = processed/elapsed
print("Aggregate throughput including in-progress: %.2f patches/s (%.0f patches/hour)" % (rate, rate*3600), flush=True)

_, jobs, exclusions = cohort_and_jobs(cfg)
if len(jobs) != run["eligible"]:
    raise SystemExit("Current eligible population differs from running job; no full ETA.")
from extract_features_dicom import read_coordinates
remaining = 0
uncounted = []
queued = 0
skipped_failed = 0
print("Counting remaining source coordinates using H5 shapes where possible; no images read...", flush=True)
for i, job in enumerate(jobs, 1):
    sid = job["slide_id"]
    if sid in failed:
        skipped_failed += 1
        continue
    if sid in completed or sid in reused:
        continue
    if sid in active:
        n, total, _ = active[sid]
        remaining += max(0, total-n)
        continue
    queued += 1
    try:
        path = Path(job["spec"]["coords_path"])
        with h5py.File(path, "r") as h:
            d = h.get("coords")
            count = d.shape[0] if isinstance(d, h5py.Dataset) and d.ndim == 2 and d.shape[1] == 2 else None
        if count is None:
            count = len(read_coordinates(path).coords_level)
        remaining += count
    except Exception as exc:
        uncounted.append((sid, str(exc)))
    if i % 500 == 0:
        print("Inventory checked:", i, "/", len(jobs), flush=True)

print("Remaining patch count estimate:", remaining)
print("Queued slides using raw source coordinate count:", queued)
print("Failed slides excluded from ETA:", skipped_failed)
print("Uncounted source files:", len(uncounted))
if uncounted:
    print("Uncounted examples:", uncounted[:10])
    raise SystemExit("Cannot give a whole-run ETA with uncounted sources. Counts above are partial.")
hours = remaining/rate/3600
kst = timezone(timedelta(hours=9))
print("Estimated remaining at snapshot: %.1f hours (%.2f days)" % (hours, hours/24))
if hours < 24*365:
    finish = datetime.fromtimestamp(snapshot, kst) + timedelta(hours=hours)
    print("Provisional finish:", finish.strftime("%Y-%m-%d %H:%M"), "KST")
print("WARNING: raw queued counts precede nonoverlap removal and future failures; current throughput may not represent later stains, MPP, or patch complexity.")
print("Progress logs update every 100 patches, so in-progress work is approximate. No extra multiplication by worker count is applied.")
print("Excludes failed-slide repairs, final reporting and manual QA. Not a confidence interval or promised deadline.")
print("Read-only diagnostic; no outputs changed and no jobs stopped.")
PY
