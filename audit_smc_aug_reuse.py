#!/usr/bin/env python3
"""Read-only coverage audit for existing SMC 40x gold splits and HE feature views.

Does not train, regenerate splits, modify labels, copy features or load tensors.
Run from the server SMC_MIL root. Coverage is not complete numerical feature QA.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

TASKS = {
    'acr_high': ('smc_acr_binary_0r1r_vs_2r3r.csv', 'smc_cv_acr_0r1r_vs_2r3r'),
    'amr': ('smc_amr_binary_pamr0_vs_positive.csv', 'smc_cv_amr_pamr0_vs_positive'),
    'significant': ('smc_significant_rejection_binary.csv', 'smc_cv_significant_rejection'),
}


def records(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def eligible_he(stains, label_source):
    """Use the same label rule as augment_uni2_wsi.select_specs."""
    selected = set()
    for row in stains:
        source = row.get('reference_source', '').strip()
        label = row.get('manual_label' if label_source == 'manual_only' else 'final_label', '').strip()
        allowed = {'manual_review'} if label_source == 'manual_only' else {'manual_review', 'filename_rule'}
        if label == 'HE' and source in allowed:
            selected.add(row['slide_id'])
    return selected


def scan_bank(bank_root, policy, total_views):
    """Search across old fold roots; fixed image views are not fold-owned labels."""
    available = {}
    problems = []
    signatures = {}
    for path in sorted(bank_root.rglob('*.json')):
        if path.parent.name != 'metadata' or 'smoke' in path.parts:
            continue
        if path.parent.parent.parent.name != policy:
            continue
        try:
            meta = json.loads(path.read_text())
            view = meta['view']
            sid = meta['slide_id']
            if not isinstance(view, int) or not 1 <= view < total_views:
                continue
            if meta.get('smoke_only') is not False:
                continue
            if (meta.get('stain') != 'HE' or meta.get('policy') != policy
                    or meta.get('feature_dim') != 1536 or meta.get('encoder') != 'uni_v2'
                    or meta.get('pyramid_level') != 0 or meta.get('patch_count', 0) <= 0
                    or path.stem != sid or path.parent.parent.name != f'view_{view}'):
                raise ValueError('Unexpected metadata')
            pt = path.parent.parent / 'pt_files' / f'{sid}.pt'
            if not pt.is_file() or pt.stat().st_size == 0:
                raise ValueError('Missing or empty PT file')
            sig = tuple(json.dumps(meta.get(k), sort_keys=True) for k in
                        ['bank_seed', 'coordinate_sha256', 'patch_count', 'patch_size', 'source_dataset',
                         'he_scale_range', 'he_implementation', 'he_scale_sampling'])
            key = (sid, view)
            if key in signatures and signatures[key] != sig:
                raise ValueError('Incompatible duplicate view metadata across bank roots')
            signatures[key] = sig
            available.setdefault(sid, {})[view] = str(pt)
        except (ValueError, KeyError, TypeError) as exc:
            problems.append({'path': str(path), 'error': str(exc)})
    complete = {sid for sid, views in available.items() if all(v in views for v in range(1, total_views))}
    return complete, available, problems


def audit(args):
    project = args.project.resolve()
    stain_path = project / args.stain_csv
    stains = records(stain_path)
    if not stains or not {'slide_id', 'final_label', 'reference_source'} <= set(stains[0]):
        raise ValueError('stain CSV needs slide_id, final_label, reference_source')
    stain_map = {r['slide_id']: r for r in stains}
    if len(stain_map) != len(stains):
        raise ValueError('Duplicate stain slide IDs')
    if args.label_source == 'manual_only' and 'manual_label' not in stains[0]:
        raise ValueError('manual_only requires manual_label column')
    selected_he = eligible_he(stains, args.label_source)
    baseline = args.feature_root / 'l0_0p25mpp_40x' / 'pt_files'
    if not baseline.is_dir():raise FileNotFoundError(baseline)
    complete, available, bank_problems = scan_bank(args.bank_root, args.policy, args.total_views)
    errors = []
    detail = []
    summaries = []
    seeds = [int(v) for v in args.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or any(s<0 for s in seeds):raise ValueError('Invalid split seeds')
    for worker in args.workers.split(','):
        csv_name, split_prefix = TASKS[worker]
        task_path = project / 'dataset_csv' / csv_name
        if not task_path.is_file():
            errors.append(f'Missing task CSV: {task_path}')
            continue
        rows = records(task_path)
        task = {r['slide_id']: r for r in rows}
        if len(task)!=len(rows):raise ValueError(f'Duplicate task slide IDs: {task_path}')
        train_union = set()
        he_union = set()
        fold_stats = []
        for seed in seeds:
            split_name = f'{split_prefix}_standard{args.folds}' + (f'_seed{seed}' if seed!=1 else '')
            for fold in range(args.folds):
                path = project / 'splits' / split_name / f'splits_{fold}.csv'
                if not path.is_file():
                    errors.append(f'Missing split: {path}')
                    continue
                split = records(path)
                partitions = {k:[r.get(k,'') for r in split if r.get(k,'')] for k in ['train','val','test']}
                for k, ids in partitions.items():
                    if len(ids)!=len(set(ids)):errors.append(f'{path}: duplicate IDs in {k}')
                    if set(ids)-task.keys():errors.append(f'{path}: {k} contains IDs absent from task CSV')
                tr, va, te = (set(partitions[k]) for k in ['train','val','test'])
                if tr&va or tr&te or va&te:errors.append(f'{path}: overlapping slide partitions')
                if te:errors.append(f'{path}: nonempty test split; verify CV mode before integration')
                if not tr or not va:errors.append(f'{path}: empty train/validation partition')
                if any(not task.get(s, {}).get('case_id', '').strip() for s in tr|va|te):
                    errors.append(f'{path}: missing patient case_id')
                tr_pat = {task.get(s,{}).get('case_id','') for s in tr}-{''}
                va_pat = {task.get(s,{}).get('case_id','') for s in va|te}-{''}
                if tr_pat&va_pat:errors.append(f'{path}: known patient overlap')
                original_missing = {s for s in tr|va|te if not (baseline/f'{s}.pt').is_file()}
                if original_missing:errors.append(f'{path}: {len(original_missing)} split slides lack baseline features')
                eligible = (tr & selected_he)-original_missing
                missing = eligible-complete
                train_union |= tr
                he_union |= eligible
                counts = {}
                for sid in tr:
                    family = stain_map.get(sid,{}).get('final_label','unmapped')
                    counts[family] = counts.get(family,0)+1
                fold_stats.append({'worker':worker,'seed':seed,'fold':fold,'train':len(tr),'val':len(va),
                                   'train_stains':counts,'eligible_he_train':len(eligible),
                                   'he_with_all_augmented_views':len(eligible&complete),
                                   'he_missing_augmented_views':len(missing),
                                   'baseline_missing':len(original_missing),
                                   'split_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        detail.extend(fold_stats)
        summaries.append({'worker':worker,'checked_folds':len(fold_stats),'expected_folds':len(seeds)*args.folds,
                          'unique_train_slides':len(train_union),'unique_eligible_he':len(he_union),
                          'he_with_all_views':len(he_union&complete),'he_needing_more_views':len(he_union-complete),
                          'missing_he_examples':sorted(he_union-complete)[:8],
                          'eligible_he_per_fold_range': [min([r['eligible_he_train'] for r in fold_stats],default=0),max([r['eligible_he_train'] for r in fold_stats],default=0)]})
    files = ['tools/run_smc_cv_grid.sh','main.py','utils/core_utils.py','utils/utils.py','dataset_modules/dataset_generic.py','feature_view_bank.py','augment_uni2_wsi.py']
    source_hashes = {f:hashlib.sha256((project/f).read_bytes()).hexdigest() if (project/f).is_file() else None for f in files}
    report = {'audit_only':True,'project':str(project),'policy':args.policy,'total_views':args.total_views,
              'label_source':args.label_source,'bank_root':str(args.bank_root),'bank_slides_with_all_views':len(complete),
              'tasks':summaries,'input_errors':errors,'bank_metadata_problems':bank_problems,
              'server_code_sha256':source_hashes,
              'note':'Coverage and basic metadata only. Does not prove tensor/coordinate/encoder parity. No training or files modified. Old fold_N is not matched to new fold_N; reuse requires fixed, label-independent generation and train-only selection.'}
    if args.details:report['fold_details']=detail
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--feature-root',type=Path,default=Path('/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2'))
    p.add_argument('--bank-root',type=Path,default=Path('/home/jupyter/data/image_team/uni_v2_aug/l0_0p25mpp_40x'))
    p.add_argument('--stain-csv',type=Path,default=Path('dataset_csv/stain_aug_labels.csv'))
    p.add_argument('--label-source',choices=['manual_only','manual_or_filename'],default='manual_only')
    p.add_argument('--policy',choices=['geometry','geometry_he_scale_light'],default='geometry')
    p.add_argument('--total-views',type=int,choices=[2,3,5],default=2)
    p.add_argument('--workers',default='acr_high,amr,significant')
    p.add_argument('--seeds',default='1,11,21,31,41')
    p.add_argument('--folds',type=int,choices=[3,5],default=5)
    p.add_argument('--details',action='store_true')
    args=p.parse_args()
    if set(args.workers.split(','))-TASKS.keys():p.error('Unknown worker')
    print(json.dumps(audit(args),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
