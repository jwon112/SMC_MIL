#!/usr/bin/env bash
# Read-only checks. No installs, writes, signals, or extraction launches.
echo '--- HOST ---'
hostname
echo '--- CURRENT DIRECTORY ---'
pwd
echo '--- PROJECT ---'
cd /home/jupyter/image_team/projects/SMC_MIL || exit 1
pwd
echo '--- CONFIGURED OUTPUT ---'
/home/jupyter/image_team/miniconda3/envs/clam_latest/bin/python - <<'PY'
from pathlib import Path
import json
from datetime import datetime, timezone

cfg_path = Path('pathomics_all_wsi.json')
print('config:', cfg_path.resolve())
if not cfg_path.exists():
    raise SystemExit('Config missing in this project.')
cfg = json.loads(cfg_path.read_text())
print('output in config:', repr(cfg.get('output')))
paths = [Path(cfg['output']) / 'full', Path('/home/jupyter/data/image_team/pathomics_all_wsi_v1/full')]
seen = set()
for root in paths:
    if str(root) in seen:
        continue
    seen.add(str(root))
    print('checking:', root, 'exists:', root.exists(), 'resolved:', root.resolve())
    if root.is_dir():
        receipts = sorted(root.glob('parallel_execution_*.json'), key=lambda p: p.stat().st_mtime)
        print('parallel receipt count:', len(receipts))
        for p in receipts[-5:]:
            print('receipt:', p.name, 'updated UTC:', datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat())
        print('run_config exists:', (root/'run_config.json').exists())
        print('slide completion receipt count:', sum(1 for _ in root.glob('slides/*/complete.json')))
print('No extraction started or stopped; no files changed.')
PY
echo '--- EXTRACTION PROCESSES ---'
ps -eww -o pid,ppid,stat,args | grep -E '[r]un_pathomics_parallel.py|[r]un_wsi_pathomics.py'
echo '--- EXPECTED 30-WORKER LOG ---'
ls -l pathomics_all_wsi_30workers.log
tail -n 8 pathomics_all_wsi_30workers.log
