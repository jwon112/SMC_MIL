"""Run one frozen attention baseline after research stain recovery.

Uses saved independent fit/stop/test roles and compares on the common 575 events.
Prepared tables and generated results contain identifiers; keep them local.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from prepare_smc_research_stains import digest
from utils.threshold_control import assert_disjoint, decision_metrics, validate_events


def read(path):
    return pd.read_csv(path,dtype={'event_id':str,'case_id':str,'slide_id':str},float_precision='round_trip')


def same_mapping(actual,expected):
    validate_events(actual)
    cols=['event_id','case_id','label']
    pd.testing.assert_frame_equal(actual[cols].sort_values('event_id').reset_index(drop=True),expected[cols].sort_values('event_id').reset_index(drop=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=['audit','run','summarize'],default='audit')
    p.add_argument('--manifest-root',type=Path,default=PROJECT/'results/smc_stain_research_20261009')
    p.add_argument('--reference-root',type=Path)
    p.add_argument('--feature-dir',type=Path)
    p.add_argument('--results-root',type=Path,default=PROJECT/'results/smc_event_research_stains_20261009')
    p.add_argument('--cohorts',nargs='+',choices=['restored7','restored3'],default=['restored7'])
    p.add_argument('--gpu',default='1');p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    m=json.loads((a.manifest_root/'manifest.json').read_text(encoding='utf-8'))
    reference=a.reference_root or Path(m['reference_results_root'])
    old_protocol=json.loads((reference/'protocol.json').read_text(encoding='utf-8'))
    a.feature_dir=(a.feature_dir or Path(old_protocol['feature_dir'])).resolve()
    for rel,expected in m['output_sha256_lf'].items():
        if digest(a.manifest_root/rel)!=expected:raise ValueError(f'Prepared input changed: {rel}')
    sources=[a.manifest_root/'manifest.json',reference/'protocol.json',Path(__file__),
             PROJECT/'tools/prepare_smc_research_stains.py',PROJECT/'tools/run_smc_imbalance_control.py',
             PROJECT/'utils/event_mil.py',PROJECT/'utils/imbalance_control.py',PROJECT/'utils/threshold_control.py',
             PROJECT/'models/stain_aware_event_mil.py',PROJECT/'train_smc_event_stain_mil.py']
    sources += [reference/f'seed{seed}'/'balanced_sampler_oof_predictions.csv' for seed in m['seeds']]
    sources += [reference/f'seed{seed}'/f'fold_{fold}'/name for seed in m['seeds'] for fold in range(5)
                for name in ('partitions.csv','balanced_sampler/fit_config.json')]
    source_hashes={str(x.resolve()):digest(x) for x in sources}
    old_events=read(a.manifest_root/'baseline575/acr_high/events.csv')
    frames={};partitions={};inventory={};missing=[]
    for cohort in a.cohorts:
        folder=a.manifest_root/cohort/'acr_high'
        events=read(folder/'events.csv');slides=read(folder/'event_slides.csv');validate_events(events)
        if len(events)!=578 or int(events.label.sum())!=17 or slides.slide_id.duplicated().any() or set(slides.event_id)!=set(events.event_id):
            raise ValueError('Invalid recovered cohort')
        if not set(slides.stain_group)<={'HE','IHC','other'}:raise ValueError('Unknown stain group')
        frames[cohort]=(events,slides)
        for seed in m['seeds']:
            tested=[]
            for fold in range(5):
                part=read(a.manifest_root/'partitions'/f'seed{seed}'/f'fold_{fold}'/'partitions.csv')
                same_mapping(part,events)
                if set(part.role)!={'fit','stop','test'}:raise ValueError('Invalid partition roles')
                roles={role:part[part.role.eq(role)].sort_values('event_id').reset_index(drop=True) for role in ['fit','stop','test']}
                assert_disjoint(*roles.values())
                previous=read(reference/f'seed{seed}'/f'fold_{fold}'/'partitions.csv')
                pd.testing.assert_frame_equal(part[part.event_id.isin(old_events.event_id)].sort_values('event_id').reset_index(drop=True),previous.sort_values('event_id').reset_index(drop=True))
                partitions[seed,fold]=roles;tested.extend(roles['test'].event_id)
            if sorted(tested)!=sorted(events.event_id):raise ValueError('Invalid outer OOF membership')
        for sid in sorted(set(slides.slide_id)):
            file=a.feature_dir/'pt_files'/f'{sid}.pt'
            if not file.is_file():missing.append(sid);continue
            stat=file.stat();inventory[sid]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            if sid in old_protocol['feature_inventory'] and stat.st_size!=old_protocol['feature_inventory'][sid]['bytes']:
                raise ValueError(f'Baseline feature size changed: {sid}')
    audit=dict(cohorts=a.cohorts,events=578,positive_events=17,patients=m['patients'],positive_patients=m['positive_patients'],
        reference_roles_preserved=True,seeds=m['seeds'],model_fits=25*len(a.cohorts),condition='balanced_sampler',
        feature_dir=str(a.feature_dir),feature_files_found=len(inventory),missing_feature_files=len(set(missing)),
        ready_to_train=not missing,primary_evaluation='common575 paired with saved baseline; recovered578 separately',threshold=.5)
    a.results_root.mkdir(parents=True,exist_ok=True)
    (a.results_root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    if a.action=='audit':print(json.dumps(audit,indent=2));return
    if a.action=='run' and missing:raise FileNotFoundError(f'Missing {len(set(missing))} feature bags; run on server with --feature-dir pointing to the complete 40x UNI-v2 bags')
    protocol=dict(**audit,source_sha256_lf=source_hashes,prepared_outputs_sha256_lf=m['output_sha256_lf'],
        max_epochs=old_protocol['max_epochs'],min_epochs=old_protocol['min_epochs'],patience=old_protocol['patience'],
        device=a.device,stopping='unweighted CE on saved independent stop patients',feature_inventory=inventory)
    record=a.results_root/'protocol.json'
    if a.action=='run':
        if record.exists() and json.loads(record.read_text(encoding='utf-8'))!=protocol:raise ValueError('Run inputs/settings changed; use a new results root')
        record.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
        # Inspect all seven restored bags before committing GPU time.
        import torch
        decisions=read(a.manifest_root/'stain_decisions_gold7.csv')
        for sid in decisions.slide_id:
            if sid not in inventory:continue
            bag=torch.load(a.feature_dir/'pt_files'/f'{sid}.pt',map_location='cpu',weights_only=True)
            if not isinstance(bag,torch.Tensor) or bag.ndim!=2 or bag.shape[1]!=1536 or not len(bag) or not torch.isfinite(bag).all():
                raise ValueError(f'Invalid restored UNI-v2 feature bag: {sid}')
    else:
        if not record.exists():raise FileNotFoundError('Training protocol missing')
        saved=json.loads(record.read_text(encoding='utf-8'))
        if saved['source_sha256_lf']!=source_hashes or saved['prepared_outputs_sha256_lf']!=m['output_sha256_lf'] or saved['cohorts']!=a.cohorts:
            raise ValueError('Summarization inputs differ from training')
    from run_smc_imbalance_control import fit_and_evaluate
    a.max_epochs=old_protocol['max_epochs'];a.min_epochs=old_protocol['min_epochs'];a.patience=old_protocol['patience']
    metrics=[];paired=[];recovered_predictions=[]
    for cohort,(events,slides) in frames.items():
        for seed in m['seeds']:
            predicted=[]
            for fold in range(5):
                roles=partitions[seed,fold];dest=a.results_root/cohort/f'seed{seed}'/f'fold_{fold}'
                dest.mkdir(parents=True,exist_ok=True)
                prediction=dest/'outer_predictions.csv'
                if not prediction.exists():
                    if a.action=='summarize':raise FileNotFoundError(f'Incomplete run: {prediction}')
                    print(f'[RUN] {cohort} seed={seed} fold={fold}',flush=True)
                    fit_and_evaluate(roles['fit'],roles['stop'],roles['test'],slides,a,seed,fold,'balanced_sampler',dest)
                f=read(prediction);same_mapping(f,roles['test'])
                if not f.fold.eq(fold).all() or not np.isfinite(f.probability).all() or not f.probability.between(0,1).all():raise ValueError('Invalid saved outer predictions')
                config=json.loads((dest/'fit_config.json').read_text())
                previous_config=json.loads((reference/f'seed{seed}'/f'fold_{fold}/balanced_sampler/fit_config.json').read_text())
                if config['initialization_sha256']!=previous_config['initialization_sha256']:raise ValueError('Model initialization differs from baseline')
                predicted.append(f)
            f=pd.concat(predicted,ignore_index=True);same_mapping(f,events)
            f.to_csv(a.results_root/cohort/f'seed{seed}'/'oof_predictions.csv',index=False)
            common=f[f.event_id.isin(old_events.event_id)].copy();same_mapping(common,old_events)
            baseline=read(reference/f'seed{seed}'/'balanced_sampler_oof_predictions.csv');same_mapping(baseline,old_events)
            common_metrics=decision_metrics(common,common.probability>=.5);baseline_metrics=decision_metrics(baseline,baseline.probability>=.5)
            for scope,table in [('recovered578',f),('common575',common)]:
                metrics.append(dict(cohort=cohort,seed=seed,scope=scope,**decision_metrics(table,table.probability>=.5)))
            paired.append(dict(cohort=cohort,seed=seed,**{k:common_metrics[k]-baseline_metrics[k] for k in ['auroc','pr_auc','tp','fp','fn','precision','sensitivity','specificity','f1','f2','mcc']}))
            restored=f[~f.event_id.isin(old_events.event_id)].copy();restored['seed']=seed;restored['cohort']=cohort
            recovered_predictions.append(restored)
    pd.DataFrame(metrics).to_csv(a.results_root/'per_seed_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(a.results_root/'paired_common575_deltas.csv',index=False)
    pd.concat(recovered_predictions,ignore_index=True).to_csv(a.results_root/'recovered3_predictions.csv',index=False)
    columns=['auroc','pr_auc','tp','fp','fn','precision','sensitivity','specificity','f1','f2','mcc']
    summary=pd.DataFrame(metrics).groupby(['cohort','scope'])[columns].agg(['mean','std'])
    summary.to_csv(a.results_root/'seed_summary.csv');print(summary.to_string())
    print('[COMPLETE] common575 paired evaluation and recovered578 evaluation are reported separately')


if __name__=='__main__':main()
