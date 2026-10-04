"""Adopt reviewed ZIP-installed edits before a fast-forward Git update.

Run a copy outside the checkout after git fetch origin. Preview by default.
Only incoming paths are considered. Matching/reviewed local edits are preserved
in a named Git stash. Unreviewed differences stop before any change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

def git(root,*args,check=True):
    return subprocess.run(['git','-C',str(root),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check)

def normalized(data):
    return hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()

def synchronize(root,ref,apply=False):
    root=root.resolve()
    actual=Path(git(root,'rev-parse','--show-toplevel').stdout.decode().strip()).resolve()
    if actual!=root:
        raise ValueError('Run from the repository root')
    target=git(root,'rev-parse',ref+'^{commit}').stdout.decode().strip()
    current=git(root,'rev-parse','HEAD').stdout.decode().strip()
    if current==target:
        print('Already at',target)
        return
    if git(root,'merge-base','--is-ancestor',current,target,check=False).returncode:
        raise ValueError('Server has divergent commits; merge review is required')
    baselines=json.loads(git(root,'show',target+':tools/server_cleanup/adoption_baselines.json').stdout)
    incoming=git(root,'diff','--no-renames','--name-only','-z',current,target).stdout.decode().split('\0')
    preserve=[]
    blocked=[]
    for name in filter(None,incoming):
        path=root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            blocked.append(name+' (symlink)')
            continue
        tracked=git(root,'ls-files','--error-unmatch','--',name,check=False).returncode==0
        changed=bool(git(root,'diff','HEAD','--',name).stdout) if tracked else path.exists()
        if not changed:
            continue
        if not path.is_file():
            blocked.append(name+' (local deletion/directory)')
            continue
        if not tracked and git(root,'check-ignore','-q','--',name,check=False).returncode==0:
            blocked.append(name+' (ignored incoming file)')
            continue
        incoming_file=git(root,'show',target+':'+name,check=False)
        if incoming_file.returncode:
            blocked.append(name+' (incoming deletion)')
            continue
        local=normalized(path.read_bytes())
        if local!=normalized(incoming_file.stdout) and local!=baselines.get(name):
            blocked.append(name+' (unreviewed local content)')
            continue
        preserve.append(name)
    if blocked:
        raise ValueError('No changes made. Review these server paths:\n'+'\n'.join(blocked))
    print('Fast-forward:',current[:12],'->',target[:12])
    print('Reviewed local paths to preserve:',len(preserve))
    for name in preserve: print('  '+name)
    if not apply:
        print('Preview only. Run with --apply to preserve edits and update.')
        return
    # Recheck the inspected checkout immediately before altering its worktree.
    if git(root,'rev-parse','HEAD').stdout.decode().strip()!=current:
        raise ValueError('HEAD changed during inspection')
    stash=None
    if preserve:
        label='smc-before-git-adoption-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        git(root,'stash','push','--include-untracked','-m',label,'--',*preserve)
        stash=git(root,'rev-parse','refs/stash').stdout.decode().strip()
        print('Preserved server edits in stash:',stash,flush=True)
        print('Keep this stash as a backup; do not pop it onto the updated files.',flush=True)
    result=git(root,'merge','--ff-only',target,check=False)
    if result.returncode:
        print(result.stderr.decode(errors='replace'))
        raise RuntimeError('Fast-forward failed; preserved edits remain in stash '+str(stash))
    print(result.stdout.decode(errors='replace'))
    print('UPDATED:',target)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--ref',default='origin/main')
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    synchronize(args.project,args.ref,args.apply)
