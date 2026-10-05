"""Audit, run and summarize the paired 2048/4096 training patch-cap control."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from summarize_smc_event_stain_mil import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['audit', 'run', 'summarize'], default='audit')
    p.add_argument('--gpu', default='1')
    p.add_argument('--manifest-root', type=Path, default=Path('/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930'))
    p.add_argument('--feature-dir', type=Path, default=Path('/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2/l0_0p25mpp_40x'))
    p.add_argument('--results-root', type=Path, default=PROJECT/'results/smc_event_patch_control_20261004_exclude25')
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 11, 21, 31, 41])
    p.add_argument('--conditions', nargs='+', choices=['cap2048', 'cap4096'], default=['cap2048', 'cap4096'])
    p.add_argument('--threshold', type=float, default=.5, help='Fixed threshold for summary metrics; choose independently of held-out OOF labels')
    args = p.parse_args()
    task = args.manifest_root.resolve()/'acr_high'
    features = args.feature_dir.resolve()
    output = args.results_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(task/'events.csv', dtype={'event_id':str,'case_id':str})
    slides = pd.read_csv(task/'event_slides.csv', dtype=str)
    expected = set(events.event_id)
    if events.event_id.duplicated().any() or slides.slide_id.duplicated().any():
        raise ValueError('Duplicate event or slide IDs')
    if args.action in ['audit', 'run']:
        inputs=[task/'events.csv',task/'event_slides.csv']
        inputs += [task/'splits'/f'seed{seed}'/f'splits_{fold}.csv' for seed in args.seeds for fold in range(5)]
        provenance={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        record=output/'input_sha256.json'
        if record.exists() and json.loads(record.read_text())!=provenance:
            raise ValueError('Input files or requested seed set changed; use a new results directory.')
        record.write_text(json.dumps(provenance,indent=2))
        records=[]
        for slide_id in slides.slide_id:
            x = torch.load(features/'pt_files'/f'{slide_id}.pt', map_location='cpu', weights_only=True)
            if x.ndim!=2 or len(x)==0 or x.shape[1]!=1536:
                raise ValueError(f'Invalid feature bag: {slide_id} {x.shape}')
            records.append({'slide_id':slide_id,'patches':len(x),'train_cap2048':min(len(x),2048),'train_cap4096':min(len(x),4096)})
            del x
        audit=slides.merge(pd.DataFrame(records),on='slide_id',validate='one_to_one').merge(events[['event_id','case_id','label']],on='event_id',validate='many_to_one')
        audit.to_csv(output/'patch_counts.csv',index=False)
        affected=audit.loc[audit.patches>2048]
        report={'events':len(events),'positive_events':int(events.label.sum()),'slides':len(audit),
                'affected_slides':len(affected),'affected_events':affected.event_id.nunique(),
                'affected_positive_events':affected.loc[affected.label==1,'event_id'].nunique(),
                'positive_patients':events.loc[events.label==1,'case_id'].nunique(),
                'validation_cap':2048,'stain_mode':'aware_nomask',
                'scope':'Training patch cap only; not new patient diversity or pixel augmentation.'}
        (output/'patch_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        if args.action=='audit':return
        if not len(affected):raise ValueError('No slides exceed 2048 patches; both conditions would be identical.')
    if args.action=='run':
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONUNBUFFERED='1')
        for condition in args.conditions:
            cap=int(condition.removeprefix('cap'))
            for seed in args.seeds:
                dest=output/f'{condition}_seed{seed}'
                config=dest/'run_config.json'
                if config.exists():
                    previous=json.loads(config.read_text())
                    for k,v in {'max_patches_per_slide':cap,'eval_max_patches_per_slide':2048,'seed':seed,'no_presence_mask':True,'stain_mode':'aware','paired_fold_seeding':True}.items():
                        if previous.get(k)!=v:raise ValueError(f'Incompatible existing configuration: {dest} {k}')
                    for k,v in {'event_csv':task/'events.csv','event_slides_csv':task/'event_slides.csv','feature_dir':features,'split_dir':task/'splits'/f'seed{seed}'}.items():
                        if Path(previous[k]).resolve()!=v:raise ValueError(f'Incompatible source path: {dest} {k}')
                if (dest/'oof_predictions.csv').exists():
                    existing=pd.read_csv(dest/'oof_predictions.csv',dtype={'event_id':str})
                    if not config.exists() or existing.event_id.duplicated().any() or set(existing.event_id)!=expected:
                        raise ValueError(f'Invalid completed OOF: {dest}')
                    print('[SKIP]',dest,flush=True);continue
                command=[sys.executable,str(PROJECT/'train_smc_event_patch_control.py'),
                         '--event-csv',str(task/'events.csv'),'--event-slides-csv',str(task/'event_slides.csv'),
                         '--split-dir',str(task/'splits'/f'seed{seed}'),'--feature-dir',str(features),
                         '--results-dir',str(dest),'--folds','5','--seed',str(seed),'--stain-mode','aware',
                         '--no-presence-mask','--paired-fold-seeding','--max-patches-per-slide',str(cap),
                         '--eval-max-patches-per-slide','2048','--device','cuda']
                print('[RUN]',condition,'seed',seed,flush=True)
                subprocess.run(command,cwd=PROJECT,env=env,check=True)
        return
    # Summarize only complete paired runs on exactly the same event/label set.
    per_seed=[];ensembles=[];initializations={}
    for condition in args.conditions:
        frames=[]
        for seed in args.seeds:
            dest=output/f'{condition}_seed{seed}'
            f=pd.read_csv(dest/'oof_predictions.csv',dtype={'event_id':str,'case_id':str})
            if f.event_id.duplicated().any() or set(f.event_id)!=expected:raise ValueError(f'Invalid event coverage: {dest}')
            if not f.set_index('event_id').label.sort_index().equals(events.set_index('event_id').label.sort_index()):raise ValueError('Label mismatch')
            if not f.set_index('event_id').case_id.sort_index().equals(events.set_index('event_id').case_id.sort_index()):raise ValueError('Patient mapping mismatch')
            if not f.probability.between(0,1).all():raise ValueError('Invalid probabilities')
            hashes=json.loads((dest/'initialization_hashes.json').read_text())
            if len(hashes)!=5:raise ValueError('Incomplete initialization hashes')
            if seed in initializations and initializations[seed]!=hashes:raise ValueError('Paired initializations differ')
            initializations[seed]=hashes
            per_seed.append({'condition':condition,'seed':seed,**metrics(f,args.threshold)})
            f['seed']=seed;frames.append(f)
        merged=pd.concat(frames)
        ensemble=merged.groupby('event_id',as_index=False).agg(case_id=('case_id','first'),label=('label','first'),probability=('probability','mean'))
        ensemble.to_csv(output/f'{condition}_ensemble_oof.csv',index=False)
        ensembles.append({'condition':condition,**metrics(ensemble,args.threshold)})
    pd.DataFrame(per_seed).to_csv(output/'per_seed_metrics.csv',index=False)
    pd.DataFrame(ensembles).to_csv(output/'ensemble_summary.csv',index=False)
    print(pd.DataFrame(ensembles).to_string(index=False))
    print(pd.DataFrame(per_seed).groupby('condition')[['auroc','pr_auc','precision','f1','mcc','sensitivity','specificity','balanced_accuracy']].agg(['mean','std']).to_string())


if __name__=='__main__':main()
