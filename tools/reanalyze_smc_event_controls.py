"""Recompute event-control metrics, paired comparisons and historical context.

Probability scores and patient-cluster intervals are descriptive OOF diagnostics.
Bootstrap fixes fitted OOF predictions: it does not repeat training or model selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from summarize_smc_event_stain_mil import metrics

SEEDS = [1, 11, 21, 31, 41]
SCORES = ['auroc','pr_auc','f1','precision','mcc','sensitivity','specificity',
          'balanced_accuracy','f2','npv','brier','brier_positive','brier_negative',
          'balanced_brier','log_loss','sensitivity_at_specificity_90','sensitivity_at_specificity_95']
PAIRS = [('presence_aware','presence_zero'), ('patch_cap2048','patch_cap4096'),
         ('previous_aware','previous_nomask'),('previous_agnostic','previous_nomask')]


def evaluate(frame):
    row = metrics(frame, .5)
    y, p = frame.label.to_numpy(int), frame.probability.to_numpy(float)
    row.update(patients=frame.case_id.nunique(), positive_patients=frame.loc[frame.label.eq(1),'case_id'].nunique())
    row['f2'] = 5*row['tp']/(5*row['tp']+4*row['fn']+row['fp'])
    row['npv'] = row['tn']/(row['tn']+row['fn']) if row['tn']+row['fn'] else np.nan
    squared = (p-y)**2
    row['brier'] = brier_score_loss(y,p)
    row['brier_positive'] = float(squared[y==1].mean())
    row['brier_negative'] = float(squared[y==0].mean())
    row['balanced_brier'] = (row['brier_positive']+row['brier_negative'])/2
    row['log_loss'] = log_loss(y,p,labels=[0,1])
    return row


def load(path, manifest):
    raw = path.read_bytes()
    manifest[str(path)] = hashlib.sha256(raw).hexdigest()
    frame = pd.read_csv(path,dtype={'event_id':str,'case_id':str}).rename(columns={'prob_positive':'probability'})
    if not {'event_id','case_id','label','probability'}.issubset(frame.columns):
        raise ValueError('Missing prediction columns: '+str(path))
    if frame.event_id.duplicated().any() or frame[['event_id','case_id','label','probability']].isna().any().any():
        raise ValueError('Invalid coverage: '+str(path))
    if not frame.label.isin([0,1]).all() or not frame.probability.between(0,1).all():
        raise ValueError('Invalid predictions: '+str(path))
    if 'fold' in frame and frame.groupby('case_id').fold.nunique().max()!=1:
        raise ValueError('Patient appears in multiple held-out folds: '+str(path))
    return frame.sort_values('event_id').set_index('event_id',drop=False)


def check_same(left,right,fold=True):
    for col in ['event_id','case_id','label']+(['fold'] if fold and 'fold' in left and 'fold' in right else []):
        if not left[col].equals(right[col]):
            raise ValueError('Comparison alignment differs: '+col)


def quick_scores(y,p):
    positive = p>=.5
    tp,fp = int(((y==1)&positive).sum()), int(((y==0)&positive).sum())
    fn,tn = int(((y==1)&~positive).sum()), int(((y==0)&~positive).sum())
    denominator = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return {'auroc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),
            'f1':2*tp/(2*tp+fp+fn),'mcc':(tp*tn-fp*fn)/denominator if denominator else 0.,
            'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp)}


def cluster_deltas(left,right,repeats,seed):
    check_same(left,right,fold=False)
    indices = [np.flatnonzero(left.case_id.to_numpy()==case) for case in sorted(left.case_id.unique())]
    rng = np.random.default_rng(seed)
    y = left.label.to_numpy(int)
    lp,rp = left.probability.to_numpy(float),right.probability.to_numpy(float)
    draws=[]
    for _ in range(repeats):
        sampled=np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
        ys=y[sampled]
        if len(np.unique(ys))<2: continue
        a,b=quick_scores(ys,lp[sampled]),quick_scores(ys,rp[sampled])
        draws.append({key:b[key]-a[key] for key in a})
    distribution=pd.DataFrame(draws)
    actual_a,actual_b=quick_scores(y,lp),quick_scores(y,rp)
    return [{'metric':col,'delta_right_minus_left':actual_b[col]-actual_a[col],
             'ci_low':distribution[col].quantile(.025),'ci_high':distribution[col].quantile(.975),
             'valid_resamples':len(draws)} for col in distribution]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('results/smc_metric_review_20261005'))
    parser.add_argument('--bootstrap',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstrap<1: parser.error('--bootstrap must be positive')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    provenance={};groups={};configs={};frames={};records=[]
    roots={
        'previous':'smc_event_stain_provisional_20260930_exclude25',
        'presence':'smc_event_presence_control_20261004_exclude25',
        'patch':'smc_event_patch_control_20261004_exclude25',
        'historical_gold':'smc_event_stain_gold5x5',
    }
    reference=load(Path('results')/roots['patch']/'cap2048_seed1/oof_predictions.csv',provenance)
    for family,base in roots.items():
        for path in sorted((Path('results')/base).glob('*/oof_predictions.csv')):
            config_path=path.with_name('run_config.json')
            config=json.loads(config_path.read_text())
            provenance[str(config_path)]=hashlib.sha256(config_path.read_bytes()).hexdigest()
            seed=int(config['seed'])
            task=Path(config['event_csv']).parent.name
            scale=Path(config['feature_dir']).name.split('_')[-1]
            if scale not in ('40x','20x','10x','5x'):
                if 'l0_0p25mpp' in path.parent.name: scale='40x'
                else: raise ValueError('Unknown magnification: '+str(path))
            mode=('agnostic' if config.get('stain_mode','aware')=='agnostic' else 'nomask' if config.get('no_presence_mask',False)
                  else 'zero' if config.get('presence_mask_values')=='zero' else 'aware')
            if family=='presence' and 'zero_mask' in path.parent.name: mode='zero'
            condition=(f'cap{config["max_patches_per_slide"]}' if family=='patch' else mode)
            group=f'{family}_{condition}' if family in ('previous','presence','patch') else f'{family}_{task}_{scale}_{condition}'
            f=load(path,provenance)
            if family!='historical_gold': check_same(reference,f,fold=False)
            frames[group,seed]=f;configs[group,seed]=config
            groups.setdefault(group,[]).append(seed)
            records.append({'family':family,'group':group,'task':task,'scale':scale,'seed':seed,**evaluate(f)})
            if family=='historical_gold' and task=='acr_high':
                cropped=f.loc[reference.index].copy()
                check_same(reference,cropped,fold=False)
                cg=group.replace('historical_gold','historical_gold_current_eval')
                frames[cg,seed]=cropped;groups.setdefault(cg,[]).append(seed)
                records.append({'family':'historical_gold_current_eval','group':cg,'task':task,'scale':scale,'seed':seed,**evaluate(cropped)})
    # Prior slide-CLAM predictions aggregated to pathology events (same unit).
    for path in sorted(Path('results/smc_slide_to_event_summary').glob('*_seed*_event_predictions.csv')):
        if 'ensemble' in path.name: continue
        prefix,seed_text=path.stem.removesuffix('_event_predictions').rsplit('_seed',1)
        task,scale=prefix.rsplit('_',1);seed=int(seed_text)
        group=f'historical_slide_{task}_{scale}'
        f=load(path,provenance)
        frames[group,seed]=f;groups.setdefault(group,[]).append(seed)
        records.append({'family':'historical_slide','group':group,'task':task,'scale':scale,'seed':seed,**evaluate(f)})
        if task=='acr_high':
            f=f.loc[reference.index].copy();check_same(reference,f,fold=False)
            group=f'historical_slide_current_eval_{task}_{scale}'
            frames[group,seed]=f;groups.setdefault(group,[]).append(seed)
            records.append({'family':'historical_slide_current_eval','group':group,'task':task,'scale':scale,'seed':seed,**evaluate(f)})
    seed_metrics=pd.DataFrame(records)
    summaries=[];ensembles={};diversity=[]
    for group,seeds in groups.items():
        if sorted(seeds)!=SEEDS: raise ValueError('Incomplete seed set: '+group)
        first=frames[group,seeds[0]]
        for seed in seeds[1:]: check_same(first,frames[group,seed],fold=False)
        matrix=np.column_stack([frames[group,seed].probability.to_numpy() for seed in SEEDS])
        ensemble=first.copy();ensemble['probability']=matrix.mean(axis=1)
        ensembles[group]=ensemble
        subset=seed_metrics.loc[seed_metrics.group.eq(group)]
        row={**subset.iloc[0][['family','group','task','scale']].to_dict(),'seeds':len(seeds),**evaluate(ensemble)}
        for score in SCORES:
            row[f'seed_mean_{score}']=subset[score].mean()
            row[f'seed_std_{score}']=subset[score].std(ddof=1)
        summaries.append(row)
        if group.startswith(('previous_','presence_','patch_')):
            ensemble[['event_id','case_id','label','probability']].to_csv(out/f'{group}_ensemble_oof.csv',index=False)
            for mask,label in [(first.label.to_numpy()==1,'positive'),(first.label.to_numpy()==0,'negative')]:
                corr=np.corrcoef(matrix[mask],rowvar=False)
                diversity.append({'group':group,'class':label,'events':int(mask.sum()),
                                  'mean_pairwise_probability_correlation':corr[np.triu_indices(5,1)].mean(),
                                  'mean_probability_std_between_seeds':matrix[mask].std(axis=1,ddof=1).mean()})
    comparisons=[];intervals=[];checks=[];positive_rows=[]
    for left,right in PAIRS:
        for seed in SEEDS:
            check_same(frames[left,seed],frames[right,seed])
            a,b=configs[left,seed],configs[right,seed]
            changes={k:[a.get(k),b.get(k)] for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
            if left=='patch_cap2048':
                base=Path('results')/roots['patch']
                paths=[base/f'{condition}_seed{seed}/initialization_hashes.json' for condition in ('cap2048','cap4096')]
                hashes=[json.loads(path.read_text()) for path in paths]
                for path in paths: provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                if len(hashes[0])!=5 or hashes[0]!=hashes[1]: raise ValueError('Patch initializations differ')
            checks.append({'left':left,'right':right,'seed':seed,'labels_patients_folds_aligned':True,
                           'configuration_differences':changes,'initialization_hashes_equal':True if left=='patch_cap2048' else None})
            a=evaluate(frames[left,seed]);b=evaluate(frames[right,seed])
            comparisons.append({'left':left,'right':right,'seed':seed,**{f'delta_{score}':b[score]-a[score] for score in SCORES}})
        for row in cluster_deltas(ensembles[left],ensembles[right],args.bootstrap,20261005):
            intervals.append({'left':left,'right':right,**row})
        for event in reference.loc[reference.label.eq(1)].index:
            row={'left':left,'right':right,'event_id':event,'case_id':reference.loc[event,'case_id']}
            for side,group in [('left',left),('right',right)]:
                row[f'{side}_ensemble_probability']=ensembles[group].loc[event,'probability']
                row[f'{side}_seed_detections']=sum(frames[group,seed].loc[event,'probability']>=.5 for seed in SEEDS)
            positive_rows.append(row)
    for name,table in [('per_seed_metrics',seed_metrics),('ensemble_metrics',pd.DataFrame(summaries)),
                       ('paired_seed_deltas',pd.DataFrame(comparisons)),('paired_patient_bootstrap',pd.DataFrame(intervals)),
                       ('seed_prediction_diversity',pd.DataFrame(diversity)),('positive_case_changes',pd.DataFrame(positive_rows))]:
        table.to_csv(out/f'{name}.csv',index=False)
    # Earlier augmentation exploration was a different task/cohort and selected
    # on validation loss. Recompute descriptively without ranking against ACR>=2R.
    early=[]
    base=Path('results/he_manual_acr_aug_explore_v2_best')
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        fs=[load(p,provenance) for p in sorted(folder.glob('fold_*/val_event_predictions.csv'))]
        if fs:
            pooled=pd.concat(fs)
            if pooled.event_id.duplicated().any(): raise ValueError('Exploratory events duplicated')
            early.append({'condition':folder.name,'folds':len(fs),'scope':'selected-validation exploratory HE cohort',**evaluate(pooled)})
    pd.DataFrame(early).to_csv(out/'historical_he_exploration.csv',index=False)
    manifest={'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
              'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
              'threshold':.5,'bootstrap_patient_repeats':args.bootstrap,'bootstrap_rng_seed':20261005,
              'bootstrap_scope':'Conditional on fitted OOF ensemble predictions; patient clusters retained; no retraining or selection uncertainty.',
              'input_sha256':provenance,'comparison_checks':checks}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    current=pd.DataFrame(summaries).loc[lambda d:d.family.isin(['previous','presence','patch'])]
    print(current[['group','events','positive_events','patients','positive_patients','auroc','pr_auc','f1','precision','mcc','tp','fp','fn','brier','balanced_brier']].to_string(index=False))
    print('Historical groups:',len(groups),'per-seed evaluations:',len(records),'input files:',len(provenance))
    print('Outputs:',out)


if __name__=='__main__': main()
