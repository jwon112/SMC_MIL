"""Review and publish server source additions/edits without ingesting datasets.

plan writes .server_source_sync/plan.json. publish revalidates that exact plan,
backs up notebooks before removing outputs, stages only its selected paths,
commits, and pushes main. Unknown configuration/data files remain review items.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

STATE='.server_source_sync'
CODE={'.py','.sh','.bash','.mjs','.js','.ts','.tsx','.jsx','.r','.jl','.c','.cpp','.h','.hpp','.m','.ipynb'}
DOC={'.md','.rst'}
CONFIG={'.yaml','.yml','.toml','.ini','.cfg'}
MAX_BYTES=2_000_000
PATHOMICS_CONFIGS={'pathomics_all_wsi.json','pathomics_extract_exploratory.json'}
SECRETS=[
 ('private key',re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
 ('GitHub token',re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b')),
 ('HuggingFace token',re.compile(r'\bhf_[A-Za-z0-9]{25,}\b')),
 ('AWS access key',re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
 ('credential assignment',re.compile(r'''(?i)(?:api_key|access_token|secret_key|password)\s*[:=]\s*["']([^"'\n]{8,})["']''')),
]

def git(root,*args,check=True):
    result=subprocess.run(['git','-C',str(root),*args],capture_output=True,check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace').strip() or 'Git command failed')
    return result

def digest(data): return hashlib.sha256(data).hexdigest()
def normalized(data): return data.replace(b'\r\n',b'\n')

def notebook_clean(raw):
    data=json.loads(raw.decode('utf-8-sig'))
    if data.get('nbformat')!=4: raise ValueError('Only notebook format 4 is supported')
    data['metadata']={k:v for k,v in data.get('metadata',{}).items() if k in ('kernelspec','language_info')}
    for cell in data.get('cells',[]):
        cell['metadata']={}
        cell.pop('attachments',None)
        if cell.get('cell_type')=='code':
            cell['outputs']=[]
            cell['execution_count']=None
    return (json.dumps(data,ensure_ascii=False,indent=1)+'\n').encode('utf-8')

def classify(name):
    p=Path(name);suffix=p.suffix.lower()
    # This is intentionally stricter than .gitignore, including tracked datasets.
    blocked={'data','_data','datasets','dataset_csv','splits','results','results_old','results_old2','archive','backups','features','runs','ckpts','checkpoints','.git','.server_source_sync'}
    if any(part in blocked or part.startswith('.server_cleanup') for part in p.parts): return None
    if any(part.endswith('_20261004') and ('control' in part or 'cleanup' in part) for part in p.parts[:-1]): return None
    if suffix in CODE: return 'notebook' if suffix=='.ipynb' else 'code'
    if suffix in DOC: return 'document'
    if suffix in CONFIG: return 'configuration'
    if p.name in ('Makefile','Dockerfile','LICENSE','.gitignore','.gitattributes') or p.name.startswith('LICENSE.'):
        return 'project metadata'
    if suffix=='.txt' and (p.name.lower().startswith(('readme','requirements','license')) or p.parts[0]=='docs'):
        return 'document'
    if suffix=='.json' and (p.name in ('package.json','package-lock.json','tsconfig.json') or 'config' in p.stem.lower()):
        return 'configuration'
    return None

def sensitive(text):
    for reason,pattern in SECRETS:
        for match in pattern.finditer(text):
            value=match.group(1) if match.lastindex else match.group(0)
            if reason=='credential assignment' and any(s in value.lower() for s in ('example','placeholder','your_','changeme','os.environ','getenv')):
                continue
            return reason
    return None

def inspect_file(root,name):
    if Path(name).is_absolute() or '..' in Path(name).parts: return None,'invalid relative path'
    p=root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root): return None,'symlink or path outside checkout'
    if not p.is_file(): return None,'deletion/directory needs explicit review'
    kind=classify(name)
    if p.stat().st_size>MAX_BYTES: return None,'over 2 MB; review separately'
    if kind is None and name not in PATHOMICS_CONFIGS and name!='eta.txt':
        return None,'dataset/output or unclassified file'
    raw=p.read_bytes()
    if name in PATHOMICS_CONFIGS:
        try: config=json.loads(raw.decode('utf-8-sig'))
        except (UnicodeError,ValueError): return None,'invalid pathomics configuration JSON'
        if (not isinstance(config,dict) or not all(isinstance(config.get(k),str) for k in ('cohort_csv','stain_csv','baseline_dir','output'))
                or not isinstance(config.get('sources'),dict)
                or any(k in config for k in ('rows','records','patients','slide_ids','patient_ids'))):
            return None,'pathomics configuration schema not recognized; review separately'
        kind='configuration'
    if name=='eta.txt':
        first=raw.splitlines()[0] if raw else b''
        if first.startswith(b'#!') and any(x in first for x in (b'/bash',b'/sh',b'env bash',b'env sh',b'python')):
            kind='code'
        else:
            return None,'referenced by smc.sh; preserve and inspect as possible shell script'
    try:
        payload=notebook_clean(raw) if kind=='notebook' else raw
        text=payload.decode('utf-8-sig')
    except (UnicodeError,ValueError): return None,'not a supported text file/notebook'
    if '\x00' in text: return None,'binary content'
    reason=sensitive(text)
    if reason: return None,'potential '+reason
    return {'path':name,'kind':kind,'bytes':len(raw),'sha256':digest(raw),'publish_sha256':digest(normalized(payload))},None

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def check_checkout(root):
    actual=Path(git(root,'rev-parse','--show-toplevel').stdout.decode().strip()).resolve()
    if actual!=root: raise ValueError('Run from the repository root')
    if git(root,'branch','--show-current').stdout.decode().strip()!='main':
        raise ValueError('Use the synchronized main checkout for this one-time consolidation')
    if git(root,'diff','--cached','--name-only').stdout:
        raise ValueError('Existing staged changes: finish or unstage them before consolidation')
    state=root/STATE
    if state.is_symlink() or state.resolve()!=state: raise ValueError('State path is a symlink')
    return git(root,'rev-parse','HEAD').stdout.decode().strip()

def make_plan(root):
    head=check_checkout(root)
    untracked=git(root,'ls-files','--others','--exclude-standard','-z').stdout.decode().split('\0')
    changed=git(root,'diff','HEAD','--name-only','-z').stdout.decode().split('\0')
    selected=[];held=[]
    for name in sorted(set(filter(None,untracked+changed))):
        entry,reason=inspect_file(root,name)
        if entry: selected.append(entry)
        else: held.append({'path':name,'reason':reason})
    plan={'version':1,'head':head,'selected':selected,'held':held,'created_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(root/STATE/'plan.json',plan)
    print('READY:',len(selected),'files;',dict(Counter(x['kind'] for x in selected)))
    for item in selected: print('  ADD/UPDATE',item['path'])
    print('HELD:',len(held))
    for item in held[:30]: print('  HOLD',item['path'],'-',item['reason'])
    print('Full selection:',root/STATE/'plan.json')
    print('Next: python tools/publish_server_sources.py --action publish')
    return plan

def publish(root):
    head=check_checkout(root)
    git(root,'var','GIT_AUTHOR_IDENT')
    git(root,'var','GIT_COMMITTER_IDENT')
    path=root/STATE/'plan.json'
    plan=json.loads(path.read_text(encoding='utf-8'))
    if plan.get('version')!=1 or plan['head']!=head: raise ValueError('HEAD changed; run plan again')
    if not plan['selected']: raise ValueError('No source files selected')
    names=[x['path'] for x in plan['selected']]
    if len(names)!=len(set(names)): raise ValueError('Duplicate plan paths')
    for item in plan['selected']:
        current,reason=inspect_file(root,item['path'])
        if current!=item: raise ValueError('File changed or no longer eligible: '+item['path'])
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for item in plan['selected']:
        p=root/item['path']
        if item['kind']=='notebook':
            backup=root/STATE/'notebook_backups'/timestamp/item['path']
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,backup)
            p.write_bytes(notebook_clean(p.read_bytes()))
    # Exact paths only; git add . / -A is never used.
    git(root,'add','--',*names)
    staged=set(filter(None,git(root,'diff','--cached','--name-only','-z').stdout.decode().split('\0')))
    if not staged.issubset(set(names)): raise ValueError('Unexpected staged paths; commit stopped')
    for item in plan['selected']:
        staged_bytes=git(root,'show',':'+item['path']).stdout
        if digest(normalized(staged_bytes))!=item['publish_sha256']:
            raise ValueError('Staged file differs from reviewed plan: '+item['path'])
    if not staged: raise ValueError('No publishable diff after normalization')
    result=git(root,'commit','-m','Consolidate server research source and configuration')
    print(result.stdout.decode(errors='replace'),flush=True)
    commit=git(root,'rev-parse','HEAD').stdout.decode().strip()
    write_json(root/STATE/'last_commit.json',{'commit':commit,'files':sorted(staged),'notebook_backup':timestamp})
    result=git(root,'push','origin','main',check=False)
    if result.returncode:
        print(result.stderr.decode(errors='replace'),flush=True)
        raise RuntimeError('Commit preserved locally: '+commit+'. Resolve push error, then retry git push origin main.')
    print('PUSHED:',commit)
    return commit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--action',choices=('plan','publish'),default='plan')
    args=p.parse_args();root=args.project.resolve()
    if args.action=='plan': make_plan(root)
    else:
        (root/STATE).mkdir(exist_ok=True)
        lock=root/STATE/'publish.lock'
        with lock.open('x'): pass
        try: publish(root)
        finally: lock.unlink()
