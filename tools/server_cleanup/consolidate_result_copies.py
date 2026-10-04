"""Consolidate reviewed result copies without changing canonical experiment files.

The plan is explicit: CSV equality includes column order and row multiplicity.
Apply moves redundant copies to an ignored local backup, with a restore journal.
No files or experiment directories are deleted.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

PLAN = Path(__file__).with_name('result_copy_plan.json')
STATE = '.server_result_sync/consolidation_20261004'


def safe_path(root, name):
    relative = Path(name)
    path = root / relative
    if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or path.resolve() != path or not path.resolve().is_relative_to(root)):
        raise ValueError('Unsafe path: ' + name)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_table(path):
    rows = list(csv.reader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
    if not rows:
        raise ValueError('Empty CSV: ' + str(path))
    return rows[0], Counter(tuple(row) for row in rows[1:])


def validate(root, entries):
    ready, missing = [], []
    for item in entries:
        src = safe_path(root, item['source'])
        dst = safe_path(root, item['target'])
        if not src.exists():
            missing.append(item['source'])
            continue
        if not src.is_file() or sha(src) != item['source_sha256']:
            raise ValueError('Source changed: ' + item['source'])
        if item['action'] == 'deduplicate_csv':
            if not dst.is_file() or csv_table(src) != csv_table(dst):
                raise ValueError('Canonical CSV differs: ' + item['target'])
            backup = safe_path(root, STATE + '/copies/' + item['source'])
            if backup.exists():
                raise ValueError('Backup already exists: ' + str(backup))
        elif item['action'] == 'rename':
            if dst.exists():
                raise ValueError('Rename destination exists: ' + item['target'])
        else:
            raise ValueError('Unknown action')
        ready.append(item)
    return ready, missing


def run(root, action, entries=None):
    root = root.resolve()
    entries = entries if entries is not None else json.loads(PLAN.read_text(encoding='utf-8'))['files']
    ready, missing = validate(root, entries)
    print(f'READY: {len(ready)}; absent/already consolidated: {len(missing)}')
    if action == 'plan':
        for item in ready:
            print(item['action'], item['source'], '->', item['target'])
        return
    # Validate the full plan before moving anything. Journal each move for recovery.
    state = safe_path(root, STATE)
    state.mkdir(parents=True, exist_ok=True)
    journal = state / 'moves.jsonl'
    for item in ready:
        src = safe_path(root, item['source'])
        destination = (STATE + '/copies/' + item['source']
                       if item['action'] == 'deduplicate_csv' else item['target'])
        dst = safe_path(root, destination)
        if sha(src) != item['source_sha256']:
            raise ValueError('Source changed during apply: ' + item['source'])
        dst.parent.mkdir(parents=True, exist_ok=True)
        with journal.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'source': item['source'], 'destination': destination,
                                     'sha256': item['source_sha256']}) + '\n')
            handle.flush()
        shutil.move(str(src), str(dst))
        if sha(dst) != item['source_sha256']:
            raise ValueError('Moved file hash changed: ' + destination)
    print(f'APPLIED: {len(ready)}; original bytes preserved; journal: {journal}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('plan', 'apply'), default='plan')
    args = parser.parse_args()
    run(Path.cwd(), args.action)
