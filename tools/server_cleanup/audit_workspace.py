"""Collect one bounded server audit for root clutter and label version review.

No project/label files are moved or edited. Output contains source code and
metadata; label row values, images, features, checkpoints and full logs are omitted.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

TEXT_LIMIT=3_000_000
LABEL_LIMIT=64_000_000
SKIP={'.git','__pycache__','.ipynb_checkpoints','data','_data','runs','results','results_old','results_old2','archive','backups','node_modules','checkpoints','ckpts'}
SOURCE_ROOTS=('tools','models','utils','dataset_modules','wsi_core','vis_utils','docs','mini')
SOURCE_EXT={'.py','.sh','.md','.txt','.json','.yaml','.yml','.ipynb'}
LABEL_ROLES={
 'wsi_curation_v2_provisional_20260930':'current: provisional stain curation',
 'event_stain_mil_gold_provisional_20260930':'current: event labels and patient splits',
 'wsi_curation_v2_final':'provenance: source of provisional curation',
 'event_stain_mil_gold_v1':'baseline: original gold event cohort',
 'wsi_stain_provisional_20260930_upload':'required input: pending_review_25.csv and application provenance',
 'stain_multiseed':'review evidence: stain ensemble and review queue',
 'wsi_curation_v1':'historical candidate: verify version/content/references',
 'wsi_curation_v2':'historical candidate: verify version/content/references',
 'weak_wsi_linkage_v1':'historical linkage evidence',
 'weak_wsi_linkage_v2':'historical linkage evidence',
 'weak_unique_train_v1':'historical weak-label training inputs',
 'stain_cv_weakunique3_v1':'historical weak-label stain cohorts',
 'future_significant_gold_v1':'future prediction research inputs',
 'future_significant_pair_gold_v1':'future prediction research inputs',
 'slide_quality_exclusions.csv':'required input: WSI quality exclusions',
}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def meta(path):
    st=path.lstat()
    result={'name':path.name,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,
            'kind':'symlink' if path.is_symlink() else 'directory' if path.is_dir() else 'file'}
    if path.is_symlink():
        result.update(link=os.readlink(path),resolved=str(path.resolve()),target_exists=path.exists())
    return result

def csv_info(path):
    """Only schema and aggregate counts; never copy row IDs or row content."""
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        columns=reader.fieldnames or []
        counts={k:Counter() for k in ('stain_group','label','split','review_status') if k in columns}
        ids={k:set() for k in ('slide_id','event_id','case_id') if k in columns}
        nonempty=Counter()
        rows=0
        for row in reader:
            rows+=1
            for k,values in ids.items():
                value=row.get(k)
                if value:
                    nonempty[k]+=1
                    values.add(value)
            for k,counter in counts.items():
                value=row.get(k) or ''
                # Bound free-text status fields; no arbitrary label text in output.
                if k=='review_status': value='nonempty' if value else 'empty'
                if value in counter or len(counter)<24: counter[value]+=1
                else: counter['<other_values>']+=1
    return {'rows':rows,'columns':columns,'unique_nonempty_ids':{k:len(v) for k,v in ids.items()},
            'repeated_nonempty_ids':{k:nonempty[k]-len(v) for k,v in ids.items()},
            'counts':{k:dict(v) for k,v in counts.items()}}

def walk_sources(root):
    for p in root.iterdir():
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in SOURCE_EXT: yield p
    for name in SOURCE_ROOTS:
        base=root/name
        if not base.is_dir() or base.is_symlink(): continue
        for parent,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in SKIP and not (Path(parent)/d).is_symlink())
            for name in sorted(files):
                p=Path(parent)/name
                if p.suffix.lower() in SOURCE_EXT and not p.is_symlink(): yield p

def git_files(root):
    run=subprocess.run(['git','-C',str(root),'ls-files','-z'],capture_output=True,check=True)
    return set(run.stdout.decode('utf-8').split('\0'))

def walk_configs(root):
    for name in ('results','results_old','results_old2','archive/experiments'):
        base=root/name
        if not base.is_dir() or base.is_symlink(): continue
        for parent,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in {'.git','__pycache__','.ipynb_checkpoints'} and not (Path(parent)/d).is_symlink())
            for name in sorted(files):
                p=Path(parent)/name
                if p.is_symlink() or p.suffix.lower() not in {'.json','.txt','.yaml','.yml'}: continue
                if 'he_aug_inputs' in p.parts or any(word in name.lower() for word in ('config','args','setting','experiment','audit')):
                    yield p

def process_refs(project,labels):
    references=[]
    unreadable=[]
    proc=Path('/proc')
    if not proc.is_dir(): return [],['/proc unavailable']
    for p in proc.iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid(): continue
        try:
            cwd=Path(os.readlink(p/'cwd'))
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            selected=set()
            for token in args:
                if not token or len(token)>4096 or '\n' in token or token.startswith('-'): continue
                candidate=Path(token)
                if not candidate.is_absolute(): candidate=cwd/candidate
                if candidate.is_relative_to(project) or candidate.is_relative_to(labels):
                    try:
                        if candidate.exists(): selected.add(str(candidate))
                    except OSError: continue
            for fd in (p/'fd').iterdir():
                try: candidate=Path(os.readlink(fd))
                except FileNotFoundError: continue
                if candidate.is_relative_to(project) or candidate.is_relative_to(labels): selected.add(str(candidate))
            if selected:
                references.append({'pid':int(p.name),'paths':sorted(selected)})
        except (FileNotFoundError,ProcessLookupError): continue
        except PermissionError: unreadable.append(p.name)
    return references,unreadable

def audit(project,labels,out):
    project=project.resolve();labels=labels.resolve()
    if not (project/'train_smc_event_stain_mil.py').is_file() or not labels.is_dir():
        raise ValueError('Expected SMC_MIL project and existing labels root')
    tracked=git_files(project)
    root_items=[]
    for p in sorted(project.iterdir()):
        if p.name.startswith('.') or p.name=='archive': continue
        item=meta(p)
        item['tracked_by_git']=p.name in tracked or any(n.startswith(p.name+'/') for n in tracked)
        if item['kind']=='file' and p.stat().st_size<=TEXT_LIMIT:
            item['sha256']=sha(p)
        root_items.append(item)
    label_files=[]
    label_dirs=[]
    for p in sorted((labels/'derived').iterdir()):
        if p.name.startswith('.'): continue
        label_dirs.append({**meta(p),'role':LABEL_ROLES.get(p.name,'review material or role unconfirmed')})
    for parent,dirs,files in os.walk(labels,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in SKIP and not (Path(parent)/d).is_symlink())
        for name in sorted(files):
            p=Path(parent)/name
            if p.suffix.lower() not in ('.csv','.json','.xlsx','.xls','.txt','.py','.sh','.yaml','.yml'): continue
            item={**meta(p),'path':p.relative_to(labels).as_posix()}
            if not p.is_symlink() and p.is_file() and p.stat().st_size<=LABEL_LIMIT:
                try:
                    item['sha256']=sha(p)
                    if p.suffix.lower()=='.csv': item['table']=csv_info(p)
                    item['stable_during_read']=p.stat().st_mtime_ns==item['mtime_ns'] and p.stat().st_size==item['bytes']
                except (OSError,UnicodeError,csv.Error) as exc: item['error']=str(exc)
            else: item['content_check']='skipped: size or symlink'
            label_files.append(item)
    targets={x['name'] for x in root_items if x['kind']=='file'}|{x['name'] for x in label_dirs}
    refs=defaultdict(list);totals=Counter();source_index=[];config_index=[];source_payload={};skipped=[]
    pattern=re.compile(r'(?<![\w.-])('+ '|'.join(re.escape(x) for x in sorted(targets,key=len,reverse=True))+r')(?![\w.-])')
    source_paths=set(walk_sources(project))
    for p in sorted(source_paths|set(walk_configs(project))):
        rel=p.relative_to(project).as_posix()
        if p.stat().st_size>TEXT_LIMIT:
            skipped.append(rel);continue
        try:
            raw=p.read_bytes();text=raw.decode('utf-8-sig')
            if p.suffix=='.ipynb':
                notebook=json.loads(text)
                text='\n'.join(''.join(c.get('source',[])) for c in notebook.get('cells',[]) if c.get('cell_type')=='code')
                raw=text.encode('utf-8')
                export=rel+'.code.txt'
            else: export=rel
            # Source/config collection is bounded and does not traverse result/data trees.
            entry={'path':rel,'tracked':rel in tracked,'sha256':sha(p),'bytes':p.stat().st_size}
            if p in source_paths:
                source_index.append(entry)
                source_payload[export]=raw
            else:
                config_index.append(entry)
            for line_no,line in enumerate(text.splitlines(),1):
                for name in set(pattern.findall(line)):
                    if rel==name: continue
                    totals[name]+=1
                    if len(refs[name])<12: refs[name].append({'source':rel,'line':line_no,'kind':'literal'})
            if p.suffix in {'.py','.ipynb'}:
                try: tree=ast.parse(text)
                except SyntaxError: continue
                for node in ast.walk(tree):
                    modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                    for module in modules:
                        name=module.split('.')[0]+'.py'
                        if name in targets and rel!=name:
                            totals[name]+=1
                            if len(refs[name])<12: refs[name].append({'source':rel,'line':node.lineno,'kind':'import'})
        except (OSError,UnicodeError,ValueError) as exc: skipped.append(rel+': '+str(exc))
    processes,unreadable=process_refs(project,labels)
    active={x for p in processes for x in p['paths']}
    for item in root_items:
        name=item['name'];path=project/name
        item['reference_count']=totals[name];item['references']=refs[name]
        item['process_reference']=str(path) in active
        if item['kind']=='directory': item['decision']='keep pending inventory/dependency review'
        elif item['tracked_by_git']: item['decision']='Git-managed: reorganize via code change, not ad-hoc move'
        elif item['process_reference']: item['decision']='keep: process reference'
        elif name in ('=','ion','psed','rror:','rs'):
            item['decision']='inspect: possible shell paste artifact'
            if item['kind']=='file' and item['bytes']<=2048:
                item['short_content']=path.read_bytes().decode('utf-8',errors='replace')
        elif path.suffix in ('.py','.sh','.ipynb'): item['decision']='preserve server-only research code before relocation'
        elif path.suffix=='.log' or name=='eta.txt': item['decision']='archive candidate: review references and process coverage'
        else: item['decision']='review purpose before relocation'
    for item in label_dirs:
        item['reference_count']=totals[item['name']];item['references']=refs[item['name']]
    groups=defaultdict(list)
    for item in label_files:
        if item.get('sha256') and item.get('stable_during_read'): groups[item['sha256']].append(item['path'])
    report={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'project':str(project),'labels':str(labels),
            'git_head':subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip(),
            'root_items':root_items,'label_directories':label_dirs,'label_files':label_files,
            'identical_label_files':[v for v in groups.values() if len(v)>1],
            'sources':source_index,'configuration_files_scanned':config_index,'skipped_sources':skipped,'process_references':processes,'unreadable_processes':unreadable,
            'limitations':['No reference match does not prove non-use; dynamic paths, launch environments and prior result configs may refer indirectly.',
                          'Directory timestamps are not last-use evidence. Byte-identical files do not prove that a whole label version is redundant.',
                          'Process snapshot cannot rule out future subprocess use. No files have been moved.']}
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    payload=json.dumps(report,ensure_ascii=False,indent=2)
    (out/'audit.json').write_text(payload,encoding='utf-8')
    rows=''.join('<tr><td>'+html.escape(x['name'])+'</td><td>'+html.escape(x['decision'])+'</td><td>'+str(x['reference_count'])+'</td><td>'+str(x['process_reference'])+'</td></tr>' for x in root_items)
    page='<html lang="ko"><meta charset="utf-8"><title>SMC workspace audit</title><style>body{font:15px/1.6 system-ui;margin:30px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>SMC 루트·라벨 점검</h1><p>이 보고서는 분류 근거를 수집합니다. 파일 이동은 수행하지 않았습니다. 참조 미검출은 미사용 판정이 아닙니다.</p><table><tr><th>항목</th><th>검토 방향</th><th>참조 수</th><th>프로세스 참조</th></tr>'+rows+'</table><p>라벨 행 수·스키마·동일 파일·참조 위치는 audit.json에 있습니다.</p></html>'
    (out/'audit.html').write_text(page,encoding='utf-8')
    with zipfile.ZipFile(out/'workspace_audit.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('audit.json',payload)
        z.writestr('audit.html',page)
        for rel,raw in source_payload.items(): z.writestr('source/'+rel,raw)
    summary={'root_items':len(root_items),'server_only_source_files':sum(not x['tracked'] for x in source_index),
             'label_metadata_files':len(label_files),'identical_label_file_groups':len(report['identical_label_files']),
             'unreadable_processes':len(unreadable),'bundle':str(out/'workspace_audit.zip')}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--labels',type=Path,default=Path('/home/jupyter/data/image_team/labels'))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    destination=args.output or args.project/'.server_cleanup_phase3'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit(args.project,args.labels,destination)
