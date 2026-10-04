"""Resolve archived SMC results through an explicit project-local registry."""
from pathlib import Path
import json
import sys

PROJECT = Path(__file__).resolve().parents[1]
REGISTRY = '.smc_result_archive.json'

def mappings():
    location = PROJECT / REGISTRY
    if not location.exists():
        return {}
    data = json.loads(location.read_text(encoding='utf-8'))
    if data.get('version') != 1:
        raise ValueError('Unsupported result archive registry')
    result = {}
    for source, target in data['moves'].items():
        old, new = PROJECT/source, PROJECT/target
        if (Path(source).is_absolute() or Path(target).is_absolute()
                or '..' in Path(source).parts or '..' in Path(target).parts
                or not source.startswith('results/')
                or not target.startswith('archive/experiments/')):
            raise ValueError('Unsafe result archive registry path')
        if not old.resolve().is_relative_to(PROJECT) or not new.resolve().is_relative_to(PROJECT):
            raise ValueError('Archive registry escaped project')
        if old.exists() and new.exists():
            raise RuntimeError('Both original and archived experiment exist: ' + source)
        if not old.exists() and not new.exists():
            raise FileNotFoundError('Both experiment locations missing: ' + source)
        result[old] = new
    return result

def resolve_result_path(path):
    """Original path when unmoved; archive path after move, including children."""
    original = Path(path)
    absolute = original.absolute()
    for old,new in mappings().items():
        if absolute == old or old in absolute.parents:
            return new / absolute.relative_to(old) if new.exists() else original
    return original

def iter_result_dirs(root):
    """Preserve the original top-level experiment set, including archived dirs."""
    root = Path(root)
    found = {p.name:p for p in root.iterdir()}
    for old,new in mappings().items():
        if old.parent == root.absolute() and new.exists():
            if old.name in found:
                raise RuntimeError('Duplicate experiment: ' + old.name)
            found[old.name] = new
    return [found[n] for n in sorted(found)]

def rglob_result_files(root,pattern):
    """Include nested external summaries stored inside moved experiments."""
    root=Path(root)
    found=set(root.rglob(pattern))
    for old,new in mappings().items():
        if old.is_relative_to(root.absolute()) and new.exists():
            found.update(new.rglob(pattern))
    return sorted(found)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tools/smc_result_paths.py OLD_PATH')
    print(resolve_result_path(sys.argv[1]))
