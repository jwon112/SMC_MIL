"""Inventory shareable results, then stage only an unchanged, size-checked plan.

Run plan first. Stage never commits or pushes; use normal git commit / git push.
Limits are local review defaults, not hosting service limits.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

from publish_server_sources import git, sensitive, write_json

STATE = '.server_result_sync'
ROOTS = ('results', 'results_old', 'results_old2', 'archive/experiments')
EXTENSIONS = {'.csv', '.tsv', '.json', '.yaml', '.yml', '.txt', '.md', '.log',
              '.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html'}
TEXT = EXTENSIONS - {'.png', '.jpg', '.jpeg', '.pdf'}
MIB = 1024 * 1024


def checkout(root):
    actual = Path(git(root, 'rev-parse', '--show-toplevel').stdout.decode().strip()).resolve()
    if actual != root:
        raise ValueError('Run from the repository root')
    state = root / STATE
    if state.is_symlink() or state.resolve() != state:
        raise ValueError('State directory must not be a symlink')
    return git(root, 'rev-parse', 'HEAD').stdout.decode().strip()


def inspect(root, name, max_file_bytes):
    p = root / name
    if not any(name.startswith(base + '/') for base in ROOTS):
        raise ValueError('Outside result roots: ' + name)
    if p.is_symlink() or not p.resolve().is_relative_to(root) or p.resolve() != p:
        raise ValueError('Symlink or noncanonical path: ' + name)
    if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
        raise ValueError('Unsupported result file: ' + name)
    before = p.stat()
    if before.st_size > max_file_bytes:
        raise ValueError('Over per-file limit')
    raw = p.read_bytes()
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during inspection')
    if p.suffix.lower() in TEXT:
        reason = sensitive(raw.decode('utf-8', errors='replace'))
        if reason:
            raise ValueError('Potential ' + reason)
    # Git filters/line-ending conversion determine the actual staged blob.
    result = subprocess.run(['git', '-C', str(root), 'hash-object', '--path=' + name, '--stdin'],
                            input=raw, capture_output=True, check=True)
    return {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
            'blob': result.stdout.decode().strip()}


def plan(root, max_file_mib=50, max_total_mib=500):
    head = checkout(root)
    selected, held = [], []
    totals = defaultdict(lambda: {'files': 0, 'bytes': 0})
    for base in ROOTS:
        folder = root / base
        if not folder.exists():
            continue
        if folder.resolve() != folder or folder.is_symlink():
            raise ValueError('Result root is a symlink: ' + base)
        for p in sorted(folder.rglob('*')):
            if not p.is_file():
                continue
            name = p.relative_to(root).as_posix()
            key = p.suffix.lower() or '(no extension)'
            totals[key]['files'] += 1
            totals[key]['bytes'] += p.stat().st_size
            try:
                if git(root, 'check-ignore', '--no-index', '-q', '--', name, check=False).returncode == 0:
                    raise ValueError('Git ignored (binary, archive or other excluded output)')
                selected.append(inspect(root, name, int(max_file_mib * MIB)))
            except ValueError as exc:
                held.append({'path': name, 'bytes': p.stat().st_size, 'reason': str(exc)})
    size = sum(x['bytes'] for x in selected)
    payload = {'version': 1, 'head': head, 'max_file_bytes': int(max_file_mib * MIB),
               'max_total_bytes': int(max_total_mib * MIB), 'selected': selected, 'held': held,
               'extensions': dict(totals), 'selected_bytes': size}
    write_json(root / STATE / 'plan.json', payload)
    print(f'SELECTED: {len(selected)} files, {size / MIB:.2f} MiB')
    print(f'HELD: {len(held)} files (preserved on server)')
    print('Extension totals, including held files:')
    for ext, item in sorted(totals.items(), key=lambda x: -x[1]['bytes']):
        print(f"  {ext}: {item['files']} files, {item['bytes'] / MIB:.2f} MiB")
    print('Largest selected files:')
    for item in sorted(selected, key=lambda x: -x['bytes'])[:15]:
        print(f"  {item['bytes'] / MIB:.2f} MiB  {item['path']}")
    print('Plan:', root / STATE / 'plan.json')
    if size > payload['max_total_bytes']:
        print('BLOCKED: selected total exceeds review limit; inspect before choosing a higher --max-total-mib.')
    print('No result file or Git index changed.')
    return payload


def stage(root):
    head = checkout(root)
    payload = json.loads((root / STATE / 'plan.json').read_text(encoding='utf-8'))
    if payload.get('version') != 1 or payload['head'] != head:
        raise ValueError('HEAD changed; run plan again')
    if git(root, 'diff', '--cached', '--name-only').stdout:
        raise ValueError('Existing staged changes; finish or unstage them first')
    selected = payload['selected']
    names = [x['path'] for x in selected]
    if not names or len(names) != len(set(names)):
        raise ValueError('Empty or duplicate selection')
    if sum(x['bytes'] for x in selected) > payload['max_total_bytes']:
        raise ValueError('Selected total exceeds review limit; inspect and run plan again')
    for item in selected:
        if inspect(root, item['path'], payload['max_file_bytes']) != item:
            raise ValueError('Result changed since plan; run plan again: ' + item['path'])
    pathspec = b''.join(name.encode('utf-8') + b'\0' for name in names)
    subprocess.run(['git', '-C', str(root), '--literal-pathspecs', 'add',
                    '--pathspec-from-file=-', '--pathspec-file-nul'], input=pathspec, check=True)
    indexed = {}
    for entry in git(root, 'ls-files', '--stage', '-z', '--', *ROOTS).stdout.decode().split('\0'):
        if entry:
            metadata, name = entry.split('\t', 1)
            indexed[name] = metadata.split()[1]
    for item in selected:
        if indexed.get(item['path']) != item['blob']:
            raise ValueError('Staged result changed during staging; do not commit. Unstage and re-plan: ' + item['path'])
    changed = set(filter(None, git(root, 'diff', '--cached', '--name-only', '-z').stdout.decode().split('\0')))
    if not changed.issubset(names):
        raise ValueError('Unexpected staged paths; do not commit')
    print(f'STAGED: {len(changed)} changed result files; no commit or push performed.')
    print('Review: git diff --cached --stat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('plan', 'stage'), default='plan')
    parser.add_argument('--max-file-mib', type=float, default=50)
    parser.add_argument('--max-total-mib', type=float, default=500)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if args.max_file_mib <= 0 or args.max_total_mib <= 0:
        parser.error('Size limits must be positive')
    if args.action == 'plan':
        plan(root, args.max_file_mib, args.max_total_mib)
    else:
        stage(root)
