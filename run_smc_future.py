#!/usr/bin/env python3
"""Server-only CLAM future-label runner, physical GPU 0, sequential CV.

Uses the installed main.py training loop with --csv_path. Its task name is a
binary loader alias only; the new CSV exclusively supplies future labels.
Never changes main.py, original classification splits or feature tensors.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_smc_future import sha

TASK_ALIAS = 'task_smc_acr_binary_0r1r_vs_2r3r'


def metrics(frame):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    two = frame.label.nunique() == 2
    return dict(events=len(frame), positive_events=int(frame.label.sum()),
                event_average_precision=float(average_precision_score(frame.label, frame.probability)) if two else None,
                event_roc_auc=float(roc_auc_score(frame.label, frame.probability)) if two else None,
                event_brier=float(brier_score_loss(frame.label, frame.probability)))


def save_event_results(results, provenance, expected, folder, fold):
    if not isinstance(results, dict) or set(map(str, results)) != set(expected):
        raise ValueError('Returned predictions do not exactly match held-out validation slides')
    source = provenance.set_index('slide_id', verify_integrity=True)
    rows = []
    for sid, result in results.items():
        r = source.loc[str(sid)]
        probability = np.asarray(result['prob']).reshape(-1)
        if probability.shape != (2,) or not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
            raise ValueError('Invalid binary probability')
        if int(result['label']) != int(r.label):
            raise ValueError('Prediction label differs from future provenance')
        rows.append(dict(slide_id=str(sid), case_id=r.case_id, anchor_date=r.anchor_date,
                         label=int(r.label), probability=float(probability[1]), fold=int(fold)))
    slides = pd.DataFrame(rows)
    keys = ['case_id', 'anchor_date']
    if (slides.groupby(keys).label.nunique() != 1).any():
        raise ValueError('Different future labels within one anchor event')
    events = slides.groupby(keys, as_index=False).agg(label=('label', 'first'),
                       probability=('probability', 'mean'), slides=('slide_id', 'count'))
    events['fold'] = int(fold)
    slides.to_csv(folder / f'future_slide_predictions_{fold}.csv', index=False)
    events.to_csv(folder / f'future_event_predictions_{fold}.csv', index=False)
    report = dict(fold=int(fold), **metrics(events),
                  aggregation='mean probability across all included stains/slides of same patient/anchor date',
                  exploratory_cv_not_independent_final_test=True)
    (folder / f'future_metrics_{fold}.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Future event metrics:', json.dumps(report), flush=True)


def checked_partitions(cohort, path):
    split = pd.read_csv(path, dtype=str).fillna('')
    parts = {k: [v for v in split[k] if v] for k in ('train', 'val', 'test')}
    by_id = cohort.set_index('slide_id', verify_integrity=True)
    sets = {k: set(v) for k, v in parts.items()}
    if parts['test'] or not parts['train'] or not parts['val']:
        raise ValueError('Expected exploratory train/val CV with empty test')
    if sets['train'] & sets['val'] or sets['train'] | sets['val'] != set(by_id.index):
        raise ValueError('Split coverage/overlap error')
    for k in ('train', 'val'):
        if len(parts[k]) != len(sets[k]) or by_id.loc[parts[k]].label.nunique() != 2:
            raise ValueError('Duplicate slide or single-class split')
    if set(by_id.loc[parts['train']].case_id) & set(by_id.loc[parts['val']].case_id):
        raise ValueError('Patient leakage in future split')
    return parts


def entry():
    """Same-process hook to validate labels and save event-level CV predictions."""
    from smc_he_aug import check_frozen_inputs, wrap_train
    config = json.loads(Path(os.environ['SMC_FUTURE_CONFIG']).read_text())
    check_frozen_inputs(config)
    import utils.core_utils as core
    underlying = wrap_train(core.train, config) if config['condition'] == 'he_geometry' else core.train
    def train(datasets, cur, args, *pos, **kw):
        check_frozen_inputs(config)
        expected = config['folds'][f'{args.task}:{args.seed}:{int(cur)}']
        provenance = pd.read_csv(config['provenance'], dtype={'case_id': str, 'slide_id': str})
        by_id = provenance.set_index('slide_id', verify_integrity=True)
        for dataset, part in zip(datasets, ('train', 'val', 'test')):
            actual = [] if dataset is None else dataset.slide_data.slide_id.astype(str).tolist()
            if len(actual) != len(set(actual)) or set(actual) != set(expected[part]):
                raise ValueError('Installed loader changed the prepared future cohort')
            if dataset is not None:
                for row in dataset.slide_data.to_dict('records'):
                    truth = by_id.loc[str(row['slide_id'])]
                    if int(row['label']) != int(truth.label) or str(row['case_id']) != truth.case_id:
                        raise ValueError('Installed loader changed future labels/patient IDs')
        if not args.cv_validation:
            raise ValueError('This runner reports exploratory validation CV only')
        result = underlying(datasets, cur, args, *pos, **kw)
        save_event_results(result[0], provenance, expected['val'], Path(args.results_dir), int(cur))
        return result
    original = core.train
    core.train = train
    try:
        sys.argv = ['main.py', *sys.argv[2:]]
        runpy.run_path('main.py', run_name='__main__')
    finally:
        core.train = original


def frozen_check(root, manifest):
    for relative, expected in manifest['generated_sha256'].items():
        path = root / relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Prepared CSV changed: {path}; build a new cohort version')


def configuration(args, manifest, task, seed, condition):
    from smc_he_aug import check_frozen_inputs
    folder = args.cohort_root / task
    cohort = pd.read_csv(folder / 'cohort.csv', dtype={'slide_id': str, 'case_id': str})
    baseline = Path(manifest['baseline_dir'])
    missing = [sid for sid in cohort.slide_id if not (baseline / 'pt_files' / f'{sid}.pt').is_file()
               or (baseline / 'pt_files' / f'{sid}.pt').stat().st_size == 0]
    if missing:
        raise ValueError(f'{len(missing)} future slides lack original features')
    hashes = {str(args.cohort_root / 'manifest.json'): sha(args.cohort_root / 'manifest.json')}
    for relative in manifest['generated_sha256']:
        p = args.cohort_root / relative
        hashes[str(p)] = sha(p)
    for name in ('main.py', 'utils/core_utils.py', 'utils/utils.py', 'dataset_modules/dataset_generic.py',
                 'run_smc_future.py', 'prepare_smc_future.py'):
        hashes[str(args.project / name)] = sha(args.project / name)
    config = dict(condition=condition, input_sha256=hashes, folds={},
                  provenance=str(folder / 'provenance.csv'), baseline_dir=str(baseline))
    if condition == 'he_geometry':
        from audit_smc_aug_reuse import eligible_he, records
        from feature_view_bank import validate_feature_bank
        bank = json.loads(args.aug_config.read_text())
        check_frozen_inputs(bank)
        if bank['policy'] != 'geometry' or bank['total_views'] != 2:
            raise ValueError('This initial comparison requires geometry 2-view bank')
        if Path(bank['baseline_dir']).resolve() != baseline.resolve():
            raise ValueError('Augmentation and future cohort baseline roots differ')
        eligible = set(cohort.slide_id) & eligible_he(records(args.stain_csv), 'manual_only')
        missing = eligible - bank['bank_index'].keys()
        if missing:
            raise ValueError(f'{len(missing)} confirmed HE slides need new full feature banks; no silent fallback')
        config.update({key: copy.deepcopy(bank[key]) for key in
                       ('policy', 'total_views', 'view_seed', 'bank_seed', 'bank_index')})
        config['eligible_he'] = sorted(eligible)
        config['bank_index'] = {sid: config['bank_index'][sid] for sid in eligible}
        for name in ('smc_he_aug.py', 'feature_view_bank.py', 'audit_smc_aug_reuse.py'):
            hashes[str(args.project / name)] = sha(args.project / name)
        hashes[str(args.aug_config)] = sha(args.aug_config)
        hashes[str(args.stain_csv)] = sha(args.stain_csv)
        indexed = cohort.set_index('slide_id')
        for sid in sorted(eligible):
            entry = config['bank_index'][sid]
            if indexed.loc[sid].source_dataset != entry['source_dataset']:
                raise ValueError('Feature source differs from future cohort')
            for p, digest in entry['metadata_sha256'].items():
                if sha(p) != digest:
                    raise ValueError('Bank metadata changed')
            validate_feature_bank([dict(slide_id=sid, stain_group='HE', source_dataset=entry['source_dataset'])],
                                  baseline_dir=baseline, bank_root=Path(entry['root']), policy='geometry', total_views=2)
    val_union = set()
    for fold in range(manifest['folds']):
        path = folder / f'seed_{seed}' / f'splits_{fold}.csv'
        parts = checked_partitions(cohort, path)
        if val_union & set(parts['val']):
            raise ValueError('Repeated held-out slides within one CV seed')
        val_union.update(parts['val'])
        config['folds'][f'{TASK_ALIAS}:{seed}:{fold}'] = dict(**parts, sha256=sha(path))
    if val_union != set(cohort.slide_id):
        raise ValueError('Incomplete OOF coverage')
    return config


def training_command(args, manifest, task, seed, condition):
    experiment = f'{task}_{condition}'
    return [sys.executable, '-u', str(args.project / 'run_smc_future.py'), '--entry',
            '--task', TASK_ALIAS, '--csv_path', str(args.cohort_root / task / 'cohort.csv'),
            '--data_root_dir', manifest['baseline_dir'],
            '--split_dir', str(args.cohort_root / task / f'seed_{seed}'),
            '--results_dir', str(args.results_root), '--exp_code', experiment,
            '--k', str(manifest['folds']), '--seed', str(seed), '--model_type', 'clam_sb',
            '--model_size', 'small', '--embed_dim', '1536', '--drop_out', '0.25',
            '--lr', str(args.lr), '--reg', '1e-5', '--bag_loss', 'ce', '--inst_loss', 'svm',
            '--bag_weight', '0.7', '--B', '8', '--weighted_sample', '--early_stopping', '--cv-validation',
            '--max_epochs', str(args.max_epochs), '--early-stop-patience', str(args.patience),
            '--early-stop-min-epoch', str(args.min_epoch)]


def aggregate(folder, folds):
    reports = [json.loads((folder / f'future_metrics_{i}.json').read_text()) for i in range(folds)]
    pd.DataFrame(reports).to_csv(folder / 'future_fold_metrics.csv', index=False)
    events = pd.concat([pd.read_csv(folder / f'future_event_predictions_{i}.csv') for i in range(folds)])
    if events.duplicated(['case_id', 'anchor_date']).any():
        raise ValueError('Repeated event across held-out folds')
    events.to_csv(folder / 'future_oof_event_predictions.csv', index=False)
    result = dict(**metrics(events), exploratory_cv_not_independent_final_test=True,
                  note='Pooled OOF ranking can differ from fold-wise ranking; use paired fold comparisons too.')
    (folder / 'future_pooled_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'train'])
    p.add_argument('--project', type=Path, default=Path('.'))
    p.add_argument('--cohort-root', type=Path, default=Path('cohorts/smc_future_v1'))
    p.add_argument('--results-root', type=Path, default=Path('results/smc_future_v1'))
    p.add_argument('--aug-config', type=Path, default=Path('results/he_aug_inputs/heaug_geometry_2view_manual_only_v1.json'))
    p.add_argument('--stain-csv', type=Path, default=Path('dataset_csv/stain_aug_labels.csv'))
    p.add_argument('--endpoints', default='next_biopsy')
    p.add_argument('--outcomes', default='acr_high,amr,significant')
    p.add_argument('--conditions', default='original,he_geometry')
    p.add_argument('--seeds', default=None, help='Subset of prepared seeds, default all')
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min-epoch', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--restart-incomplete', action='store_true', help='Back up incomplete result then restart that task/seed')
    args = p.parse_args()
    args.project = args.project.resolve()
    for key in ('cohort_root', 'results_root', 'aug_config', 'stain_csv'):
        setattr(args, key, (args.project / getattr(args, key)).resolve())
    if args.max_epochs <= 0 or args.patience < 1 or not 0 <= args.min_epoch < args.max_epochs or args.lr <= 0:
        raise ValueError('Invalid training parameters')
    conditions = args.conditions.split(',')
    if not set(conditions) <= {'original', 'he_geometry'} or len(conditions) != len(set(conditions)):
        raise ValueError('conditions must be original,he_geometry or a single condition')
    manifest = json.loads((args.cohort_root / 'manifest.json').read_text())
    if manifest['audit_only']:
        raise ValueError('Workbook audit is not a feature-verified training cohort')
    frozen_check(args.cohort_root, manifest)
    seeds = manifest['seeds'] if args.seeds is None else [int(s) for s in args.seeds.split(',')]
    if not set(seeds) <= set(manifest['seeds']) or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be a unique subset of prepared seeds')
    main_source = (args.project / 'main.py').read_text()
    for flag in ('--csv_path', '--cv-validation', '--early-stop-patience', '--early-stop-min-epoch'):
        if flag not in main_source:
            raise ValueError(f'Installed main.py lacks {flag}; do not silently change training protocol')
    core_source = (args.project / 'utils/core_utils.py').read_text()
    for required in ('early_stop_patience', 'early_stop_min_epoch'):
        if required not in core_source:
            raise ValueError(f'Installed core_utils.py does not use {required}; inspect server version first')
    jobs = []
    for endpoint in args.endpoints.split(','):
        for outcome in args.outcomes.split(','):
            task = f'{endpoint}_{outcome}'
            if task not in manifest['tasks']:
                raise ValueError(f'Task not prepared: {task}')
            if not manifest['tasks'][task]['ready']:
                raise ValueError(f'Task blocked: {task}: {manifest["tasks"][task].get("blocked_reason")}')
            for condition in conditions:
                for seed in seeds:
                    config = configuration(args, manifest, task, seed, condition)
                    command = training_command(args, manifest, task, seed, condition)
                    print(f'Preflight OK: {task} {condition} seed={seed}; GPU=0; HE={len(config.get("eligible_he", []))}', flush=True)
                    jobs.append((task, condition, seed, config, command))
    if args.command == 'audit':
        print(f'Audit only: {len(jobs)} task/condition/seed runs; no training started.')
        return
    # One GPU0 job per result root; does not interrupt unrelated existing jobs.
    import fcntl
    args.results_root.mkdir(parents=True, exist_ok=True)
    with (args.results_root / '.runner.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = []
        for task, condition, seed, config, command in jobs:
            frozen_check(args.cohort_root, manifest)
            folder = args.results_root / f'{task}_{condition}_s{seed}'
            receipt = {'config': config, 'command': command}
            if folder.exists() and any(folder.iterdir()):
                old = folder / 'future_run_config.json'
                if not old.is_file() or json.loads(old.read_text()) != receipt:
                    raise ValueError(f'Result configuration changed; use a new --results-root: {folder}')
                if (folder / 'summary.csv').is_file() and all((folder / f'future_metrics_{i}.json').is_file() for i in range(manifest['folds'])):
                    summary.append(dict(task=task, condition=condition, seed=seed, **aggregate(folder, manifest['folds'])))
                    print(f'Already complete: {folder}', flush=True)
                    continue
                if not args.restart_incomplete:
                    raise ValueError(f'Incomplete run: {folder}; --restart-incomplete backs it up before restarting')
                backup = folder.with_name(folder.name + '.incomplete_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                folder.rename(backup)
                print(f'Preserved incomplete run: {backup}', flush=True)
            folder.mkdir()
            (folder / 'future_run_config.json').write_text(json.dumps(receipt, indent=2) + '\n')
            config_path = folder / 'future_input_config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', SMC_FUTURE_CONFIG=str(config_path))
            print(f'Running {task}/{condition}/seed={seed}; log: {folder / "training.log"}', flush=True)
            with (folder / 'training.log').open('x') as log:
                subprocess.run(command, cwd=args.project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary.append(dict(task=task, condition=condition, seed=seed, **aggregate(folder, manifest['folds'])))
            pd.DataFrame(summary).to_csv(args.results_root / 'future_summary.csv', index=False)
    pd.DataFrame(summary).to_csv(args.results_root / 'future_summary.csv', index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--entry':
        entry()
    else:
        main()
