"""Archive selected historical results after receiving reader changes through Git.

Default plan. apply moves results and creates a registry; verify checks integrity;
rollback restores result locations. Git manages all code versions independently.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

HERE = Path(__file__).resolve().parent
STATE_NAME = '.server_cleanup_phase2_20261004'
REGISTRY = '.smc_result_archive.json'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def code_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)

def safe(root,rel):
    p=root/rel
    if Path(rel).is_absolute() or '..' in Path(rel).parts or '\\' in rel:
        raise ValueError('Unsafe relative path: '+rel)
    if p.resolve()!=p or not p.resolve().is_relative_to(root):
        raise ValueError('Symlink or path outside project: '+str(p))
    return p

def inventory(folder):
    result={}
    for p in sorted(folder.rglob('*')):
        rel=p.relative_to(folder).as_posix()
        st=p.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise ValueError('Symlink inside result; inspect manually: '+str(p))
        if stat.S_ISDIR(st.st_mode):
            result[rel]={'type':'directory'}
        elif stat.S_ISREG(st.st_mode):
            result[rel]={'type':'file','size':st.st_size,'sha256':digest(p)}
        else:
            raise ValueError('Special file inside result: '+str(p))
    return result

def read_package():
    plan=json.loads((HERE/'move_plan.json').read_text(encoding='utf-8'))['moves']
    hashes=json.loads((HERE/'code_manifest.json').read_text(encoding='utf-8'))
    patches={n:{'before':h,'after':h} for n,h in hashes.items()}
    sources=set()
    for item in plan:
        source=item['source']
        name=Path(source).name
        group='weak_linkage' if 'weakunique' in name else 'early_cv'
        if (not source.startswith('results/smc_') or len(Path(source).parts)!=2
                or 'cv3val' not in name or source in sources
                or item['target']!=f'archive/experiments/{group}/{name}'):
            raise ValueError('Invalid move plan')
        sources.add(source)
    return plan,patches

def check_processes(root,plan,patches):
    if not Path('/proc').is_dir():
        raise ValueError('Apply/rollback requires a Linux server with /proc')
    paths=[root/item[k] for item in plan for k in ('source','target')]
    scripts={Path(s).name for s in patches}
    # Legacy launchers may start a child after a momentary gap in training.
    scripts.update(('main.py','smc_he_aug.py','run_smc_stain_queue.sh','run_smc_repeated_cv_grid.sh'))
    absolute_scripts={str(root/name) for name in patches}
    absolute_scripts.update(str(root/name) for name in ('main.py','smc_he_aug.py','tools/run_smc_stain_queue.sh','tools/run_smc_repeated_cv_grid.sh'))
    busy=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():
            continue
        try:
            args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
            cwd=Path(os.readlink(proc/'cwd'))
            if any(a in absolute_scripts for a in args) or ((root==cwd or root in cwd.parents) and any(Path(a).name in scripts for a in args if a)):
                busy.append(proc.name)
                continue
            if any(str(p) in a or (root==cwd and p.name in a) for a in args for p in paths):
                busy.append(proc.name)
                continue
            for link in [proc/'cwd']+list((proc/'fd').iterdir()):
                try:
                    target=Path(os.readlink(link))
                except FileNotFoundError:
                    continue
                if any(target==p or p in target.parents for p in paths):
                    busy.append(proc.name)
                    break
        except (FileNotFoundError,ProcessLookupError):
            continue
        except PermissionError:
            raise ValueError('Cannot inspect process '+proc.name)
    if busy:
        raise ValueError('Historical training/reader or selected result in use; PIDs: '+','.join(sorted(set(busy))))

def inspect(root,plan,patches):
    selected=[]
    missing=[]
    for item in plan:
        source,target=safe(root,item['source']),safe(root,item['target'])
        if target.exists():
            raise ValueError('Archive destination exists: '+str(target))
        if not source.exists():
            missing.append(item['source'])
            continue
        if not source.is_dir():
            raise ValueError('Expected result directory: '+str(source))
        selected.append(item)
    for name,h in patches.items():
        target=safe(root,name)
        if h['before'] is None:
            if target.exists():
                raise ValueError('New helper already exists: '+name)
        elif not target.is_file() or code_digest(target)!=h['before']:
            raise ValueError('Git-managed reader differs from expected version: '+name)
    if (root/REGISTRY).exists() or (root/REGISTRY).is_symlink():
        raise ValueError('An archive registry already exists; inspect transaction before proceeding')
    return selected,missing

def load_helper(root):
    spec=importlib.util.spec_from_file_location('_smc_archive_check',root/'tools/smc_result_paths.py')
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper

def verify(root,state):
    expected={x['source']:x['target'] for x in state['moves']}
    actual=json.loads(safe(root,REGISTRY).read_text(encoding='utf-8'))
    if actual!={'version':1,'moves':expected}:
        raise ValueError('Registry mismatch')
    for name,h in state['patches'].items():
        if code_digest(safe(root,name))!=h['after']:
            raise ValueError('Installed reader changed: '+name)
    helper=load_helper(root)
    after=sorted(p.name for p in helper.iter_result_dirs(root/'results') if p.is_dir())
    if after!=state['original_result_dirs']:
        raise ValueError('Experiment discovery set changed')
    files=0
    for item in state['moves']:
        src,dst=safe(root,item['source']),safe(root,item['target'])
        if src.exists() or not dst.is_dir():
            raise ValueError('Unexpected move state: '+item['source'])
        current=inventory(dst)
        if current!=state['inventories'][item['source']]:
            raise ValueError('Archived content differs: '+item['source'])
        if helper.resolve_result_path(src)!=dst:
            raise ValueError('Original path does not resolve: '+item['source'])
        files+=sum(x['type']=='file' for x in current.values())
    report={'verified':True,'moved_experiments':len(state['moves']),'unchanged_files':files,
            'experiment_discovery_unchanged':True,'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(root/STATE_NAME/'verification.json',report)
    print(json.dumps(report,indent=2))
    return report

def restore(root,state):
    # Preflight ALL entries before making the first rollback mutation.
    for item in state['moves']:
        src,dst=safe(root,item['source']),safe(root,item['target'])
        if src.exists()==dst.exists():
            raise ValueError('Rollback collision or missing result: '+item['source'])
        if inventory(dst if dst.exists() else src)!=state['inventories'][item['source']]:
            raise ValueError('Result changed since migration: '+item['source'])
    registry=safe(root,REGISTRY)
    expected={'version':1,'moves':{x['source']:x['target'] for x in state['moves']}}
    if registry.exists() and json.loads(registry.read_text(encoding='utf-8'))!=expected:
        raise ValueError('Registry changed; refusing rollback')
    for item in reversed(state['moves']):
        src,dst=safe(root,item['source']),safe(root,item['target'])
        if dst.exists():
            dst.rename(src)
    if registry.exists(): registry.unlink()
    state['status']='rolled_back'
    write_json(root/STATE_NAME/'transaction.json',state)
    print('RESTORED original result locations; code remains managed by Git')

def apply(root,plan,patches):
    selected,missing=inspect(root,plan,patches)
    if not selected:
        raise ValueError('No listed historical results found')
    check_processes(root,selected,patches)
    state={'version':1,'git_managed':True,'status':'preparing','moves':selected,'missing':missing,'patches':patches,
           'original_result_dirs':sorted(p.name for p in (root/'results').iterdir() if p.is_dir()),'inventories':{}}
    for item in selected:
        src=safe(root,item['source'])
        state['inventories'][item['source']]=inventory(src)
        dst=safe(root,item['target'])
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.stat().st_dev!=dst.parent.stat().st_dev:
            raise ValueError('Archive must be on same filesystem: '+item['source'])
    check_processes(root,selected,patches)
    inspect(root,plan,patches)
    state['status']='prepared'
    write_json(root/STATE_NAME/'transaction.json',state)
    try:
        write_json(root/REGISTRY,{'version':1,'moves':{x['source']:x['target'] for x in selected}})
        for item in selected:
            src,dst=safe(root,item['source']),safe(root,item['target'])
            if inventory(src)!=state['inventories'][item['source']]:
                raise ValueError('Result changed before move: '+item['source'])
            src.rename(dst)
            print('MOVED',item['source'],'->',item['target'],flush=True)
        verify(root,state)
        state['status']='committed'
        write_json(root/STATE_NAME/'transaction.json',state)
    except Exception:
        print('Migration failed; restoring the prepared transaction',flush=True)
        restore(root,state)
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--action',choices=('plan','apply','verify','rollback'),default='plan')
    args=parser.parse_args()
    root=args.project.resolve()
    if not (root/'train_smc_event_stain_mil.py').is_file() or not (root/'results').is_dir():
        parser.error('Run from SMC_MIL project root')
    safe(root,STATE_NAME)
    plan,patches=read_package()
    state_dir=root/STATE_NAME
    transaction=state_dir/'transaction.json'
    if args.action=='plan':
        if transaction.exists():
            print('Existing transaction:',json.loads(transaction.read_text(encoding='utf-8'))['status'])
            print('Use --action verify or --action rollback')
            return
        selected,missing=inspect(root,plan,patches)
        check_processes(root,selected,patches)
        report={'selected':selected,'missing':missing,'patch_files':list(patches)}
        write_json(state_dir/'preview.json',report)
        print('READY:',len(selected),'historical result directories;',dict(Counter(x['group'] for x in selected)))
        print('Verified Git-managed reader/launcher files:',len(patches))
        print('Not found:',len(missing))
        print('Exact move list:',state_dir/'preview.json')
        print('No project code or result directory changed. Next: --action apply')
        return
    state_dir.mkdir(parents=True,exist_ok=True)
    lock=state_dir/'operation.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        if transaction.exists():
            state=json.loads(transaction.read_text(encoding='utf-8'))
            if not state.get('git_managed'):
                raise ValueError('ZIP-managed transaction detected; use its original tool for rollback first')
            if state['moves']!=[x for x in plan if x['source'] in {i['source'] for i in state['moves']}] or state['patches']!=patches:
                raise ValueError('Transaction differs from packaged plan')
            if args.action=='rollback':
                if state['status']=='rolled_back':
                    print('Already rolled back')
                    return
                check_processes(root,state['moves'],patches)
                restore(root,state)
            elif args.action=='verify' or (args.action=='apply' and state['status']=='committed'):
                verify(root,state)
            elif args.action=='apply' and state['status']=='rolled_back':
                apply(root,plan,patches)
            else:
                raise ValueError('Existing transaction; use rollback before a new migration')
        elif args.action=='apply':
            apply(root,plan,patches)
        else:
            raise ValueError('No transaction to verify/rollback')
    finally:
        lock.unlink()

if __name__=='__main__':
    main()
