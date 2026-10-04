#!/usr/bin/env python3
"""Preserve installed SMC gold CV; augment confirmed HE in training only.

audit: read-only coverage/space check. prepare: fill missing feature banks and
freeze an index. train: run the installed grid with only its entry point and
experiment name changed. Defaults are the user's 40x, 5-fold x 5-seed protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from audit_smc_aug_reuse import TASKS, audit, eligible_he, records
from smc_he_aug import digest
from feature_view_bank import feature_path, torch_feature_loader, validate_feature_bank

SCALE = 'l0_0p25mpp_40x'
TASK_NAMES = {key: 'task_' + value[0][:-4] for key, value in TASKS.items()}
SHORT_NAMES = {'acr_high': 'acr_high_grade', 'amr': 'amr_positive', 'significant': 'significant_rejection'}
SOURCES = {
    'exp3': ('dicom', '/home/jupyter/data/image_team/exp3_inbox', '_clam/dicom_feature_manifest.csv'),
    'mrxs13': ('mrxs', '/home/jupyter/data/image_team/mrxs13_inbox', '_clam/mrxs_feature_manifest_l0.csv'),
}


def split_dir(worker, seed, folds):
    return TASKS[worker][1] + f'_standard{folds}' + (f'_seed{seed}' if seed != 1 else '')


def tag(args):
    return f'heaug_{args.policy}_{args.total_views}view_{args.label_source}_v1'


def input_path(args):
    return args.project / 'results' / 'he_aug_inputs' / f'{tag(args)}.json'


@contextmanager
def lock_file(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another process is using {path}; do not prepare/train concurrently here')
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_inputs(args):
    report = audit(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report['input_errors'] or report['bank_metadata_problems']:
        raise RuntimeError('Audit failed; resolve the reported input/metadata errors first')
    stains = records(args.project / args.stain_csv)
    he = eligible_he(stains, args.label_source)
    folds = {}
    slides = {}
    hashes = {str((args.project / args.stain_csv).resolve()): digest(args.project / args.stain_csv)}
    for worker in args.workers.split(','):
        task_csv = args.project / 'dataset_csv' / TASKS[worker][0]
        hashes[str(task_csv)] = digest(task_csv)
        task_rows = {r['slide_id']: r for r in records(task_csv)}
        for seed in map(int, args.seeds.split(',')):
            for fold in range(args.folds):
                path = args.project / 'splits' / split_dir(worker, seed, args.folds) / f'splits_{fold}.csv'
                rows = records(path)
                partitions = {key: [r.get(key, '') for r in rows if r.get(key, '')]
                              for key in ('train', 'val', 'test')}
                partitions.update(worker=worker, seed=seed, fold=fold, path=str(path),
                                  task_csv=str(task_csv), sha256=digest(path))
                folds[f'{TASK_NAMES[worker]}:{seed}:{fold}'] = partitions
                hashes[str(path)] = digest(path)
                for sid in set(partitions['train']) & he:
                    row = task_rows[sid]
                    source = row.get('source_dataset', '').strip()
                    if source not in SOURCES:
                        raise ValueError(f'{sid}: unsupported/missing source_dataset={source!r}')
                    if sid in slides and slides[sid]['source_dataset'] != source:
                        raise ValueError(f'Conflicting source_dataset across tasks: {sid}')
                    slides[sid] = {'slide_id': sid, 'source_dataset': source, 'stain_group': 'HE'}
    if not slides:
        raise ValueError('No confirmed HE training slides under this label rule')
    return folds, slides, hashes


def find_banks(args, slides, deep=False):
    """Find a complete bank per slide; never equate old and new fold numbers."""
    roots = {}
    for path in args.bank_root.rglob(f'{args.policy}/view_1/metadata/*.json'):
        if 'smoke' not in path.parts and path.stem in slides:
            roots.setdefault(path.stem, set()).add(path.parents[3].resolve())
    index = {}
    for sid, row in slides.items():
        for root in sorted(roots.get(sid, [])):
            metas = [root / args.policy / f'view_{v}' / 'metadata' / f'{sid}.json'
                     for v in range(1, args.total_views)]
            if not all(p.is_file() for p in metas):
                continue
            values = [json.loads(p.read_text()) for p in metas]
            if any(m.get('bank_seed') != args.bank_seed for m in values):
                raise ValueError(f'{sid}: bank seed differs; do not mix augmentation banks')
            if args.policy == 'geometry_he_scale_light':
                for m in values:
                    if (m.get('he_scale_range') != [0.95, 1.05]
                            or m.get('he_scale_sampling') != 'independent_uniform_per_patch_view'
                            or m.get('he_implementation') !=
                            'scikit-image rgb2hed/hed2rgb; apply varied-minus-unvaried RGB delta'):
                        raise ValueError(f'{sid}: incompatible H/E implementation')
            validate_feature_bank([row], baseline_dir=args.feature_root / SCALE,
                                  bank_root=root, policy=args.policy, total_views=args.total_views)
            count = values[0]['patch_count']
            if deep:
                for view in range(args.total_views):
                    pt = feature_path(args.feature_root / SCALE, root, args.policy, sid, view)
                    tensor = torch_feature_loader(pt)
                    if tuple(tensor.shape) != (count, 1536):
                        raise ValueError(f'Original/augmented bag shape differs: {pt}')
            index[sid] = {'root': str(root), 'patch_count': count,
                          'source_dataset': row['source_dataset'],
                          'metadata_sha256': {str(p): digest(p) for p in metas}}
            break
    return index


def nearest_existing(path):
    while not path.exists():
        path = path.parent
    return path


def execute(command, args, extra_env=None):
    print('Running:', ' '.join(map(str, command)), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    env.update(extra_env or {})
    subprocess.run(list(map(str, command)), cwd=args.project, env=env, check=True)


def prepare(args, folds, slides, index, hashes):
    shared = args.bank_root / 'gold_shared_v1'
    # Each extraction call uses a real, existing training split. No dummy task
    # labels, regenerated splits, or additional rows are used for training.
    for fold in folds.values():
        missing = (set(fold['train']) & slides.keys()) - index.keys()
        for source, (kind, dataset_root, manifest_rel) in SOURCES.items():
            ids = sorted(sid for sid in missing if slides[sid]['source_dataset'] == source)
            if not ids:
                continue
            queue = args.project / 'results' / 'he_aug_inputs' / 'queues'
            queue.mkdir(parents=True, exist_ok=True)
            list_path = queue / f'{tag(args)}_{fold["worker"]}_{fold["seed"]}_{fold["fold"]}_{source}.txt'
            list_path.write_text('\n'.join(ids) + '\n')
            command = [sys.executable, '-u', 'augment_uni2_wsi.py',
                       '--source', kind, '--source-dataset', source,
                       '--dataset-root', dataset_root, '--feature-manifest', str(Path(dataset_root) / manifest_rel),
                       '--baseline-dir', args.feature_root / SCALE,
                       '--stain-csv', args.project / args.stain_csv,
                       '--task-csv', fold['task_csv'], '--split-csv', fold['path'],
                       '--output-root', shared, '--stain', 'HE', '--label-source', args.label_source,
                       '--policy', args.policy, '--num-views', args.total_views - 1,
                       '--bank-seed', args.bank_seed, '--batch-size', args.batch_size,
                       '--check-original-patches', 16, '--device', 'cuda', '--amp',
                       '--include-slide-ids', list_path]
            if source == 'mrxs13':
                command.append('--allow-mrxs-baseline-h5-coord-mismatch')
            execute(command, args)
            found = find_banks(args, {sid: slides[sid] for sid in ids})
            if set(ids) - found.keys():
                raise RuntimeError(f'Extractor did not produce all requested slides: {set(ids)-found.keys()}')
            index.update(found)
    print('Checking original and augmented tensors (shape, finite values, coordinates)...', flush=True)
    index = find_banks(args, slides, deep=True)
    if slides.keys() - index.keys():
        raise RuntimeError('Incomplete bank after extraction')
    for name in ('main.py', 'tools/run_smc_cv_grid.sh', 'utils/core_utils.py', 'utils/utils.py',
                 'dataset_modules/dataset_generic.py', 'smc_he_aug.py', 'run_smc_gold_he_aug.py',
                 'audit_smc_aug_reuse.py', 'feature_view_bank.py', 'augment_uni2_wsi.py'):
        hashes[str(args.project / name)] = digest(args.project / name)
    config = {'schema': 1, 'policy': args.policy, 'total_views': args.total_views,
              'label_source': args.label_source, 'bank_seed': args.bank_seed, 'view_seed': args.view_seed,
              'baseline_dir': str(args.feature_root / SCALE), 'eligible_he': sorted(slides),
              'bank_index': index, 'folds': folds, 'input_sha256': hashes}
    destination = input_path(args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and json.loads(destination.read_text()) != config:
        raise RuntimeError(f'Frozen index differs: {destination}. Preserve previous results/index before changing inputs.')
    if not destination.exists():
        with destination.open('x') as handle:
            json.dump(config, handle, indent=2)
    print(f'Ready: {len(index)} HE slides, {len(folds)} folds. Index: {destination}', flush=True)


def adapted_grid(source):
    """Only two surgical text substitutions; fail on an unrecognized script."""
    entry = 'python main.py'
    experiment = 'exp_code="smc_${short_name}_${scale}_uni2_clamsb${exp_mode}"'
    if source.count(entry) != 1 or source.count(experiment) != 1:
        raise ValueError('Server grid script differs from supported layout; inspect before adapting')
    return source.replace(entry, 'python smc_he_aug.py').replace(
        experiment, 'exp_code="smc_${short_name}_${scale}_uni2_clamsb${exp_mode}_${SMC_HE_AUG_TAG:?}"')


def train(args, folds):
    from smc_he_aug import check_frozen_inputs
    destination = input_path(args)
    if not destination.is_file():
        raise FileNotFoundError(f'Run prepare first: {destination}')
    config = json.loads(destination.read_text())
    check_frozen_inputs(config)
    if config['bank_seed'] != args.bank_seed or config['view_seed'] != args.view_seed:
        raise ValueError('Requested bank/view seeds differ from prepared configuration')
    if Path(config['baseline_dir']).resolve() != (args.feature_root / SCALE).resolve():
        raise ValueError('Requested baseline root differs from prepared configuration')
    if not folds.keys() <= config['folds'].keys():
        raise ValueError('Requested folds are not all in the prepared index')
    grid = args.project / 'tools' / f'run_smc_cv_grid_{tag(args)}.sh'
    content = adapted_grid((args.project / 'tools/run_smc_cv_grid.sh').read_text())
    with lock_file(args.project / 'results/he_aug_inputs' / f'{tag(args)}.grid.lock'):
        if grid.exists() and grid.read_text() != content:
            raise RuntimeError(f'Existing adapter differs: {grid}')
        if not grid.exists():
            with grid.open('x') as handle:
                handle.write(content)
    for seed in map(int, args.seeds.split(',')):
        for worker in args.workers.split(','):
            exp = f'smc_{SHORT_NAMES[worker]}_{SCALE}_uni2_clamsb_cv{args.folds}val_{tag(args)}_s{seed}'
            result = args.project / 'results' / exp
            receipt = result / 'he_run_config.json'
            run_config = {'prepared_index_sha256': digest(destination), 'seed': seed, 'worker': worker,
                          'max_epochs': args.max_epochs, 'early_stop_patience': args.early_stop_patience,
                          'early_stop_min_epoch': args.early_stop_min_epoch,
                          'adapted_grid_sha256': digest(grid)}
            with lock_file(args.project / 'results' / 'he_aug_inputs' / f'{exp}.lock'):
                if result.exists() and any(result.iterdir()):
                    if not receipt.is_file() or json.loads(receipt.read_text()) != run_config:
                        raise RuntimeError(f'Result configuration differs: {result}')
                    if (result / 'summary.csv').is_file():
                        print(f'Already complete: {result}', flush=True)
                        continue
                    # Do not silently overwrite partial checkpoints or claim exact resume.
                    if not args.restart_incomplete:
                        raise RuntimeError(f'Incomplete run: {result}. Use --restart-incomplete to back up and restart this task/seed.')
                    import datetime
                    backup = result.with_name(result.name + '.incomplete_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
                    result.rename(backup)
                    print(f'Preserved incomplete run at {backup}; restarting this task/seed', flush=True)
                result.mkdir(parents=True, exist_ok=True)
                with receipt.open('x') as handle:
                    json.dump(run_config, handle, indent=2)
                log = args.project / 'results/logs' / f'{exp}.log'
                if log.exists():
                    import datetime
                    log.rename(log.with_name(log.name + '.saved_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
                execute(['bash', grid, '--gpu', args.gpu, '--worker', worker, '--scale-worker', '40x',
                         '--folds', args.folds, '--seed', seed, '--feature-root', args.feature_root,
                         '--max-epochs', args.max_epochs, '--early-stop-patience', args.early_stop_patience,
                         '--early-stop-min-epoch', args.early_stop_min_epoch], args,
                        {'SMC_HE_AUG_CONFIG': str(destination), 'SMC_HE_AUG_TAG': tag(args)})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['audit', 'prepare', 'train'])
    p.add_argument('--project', type=Path, default=Path('.'))
    p.add_argument('--feature-root', type=Path, default=Path('data/features/uni_v2'))
    p.add_argument('--bank-root', type=Path, default=Path('/home/jupyter/data/image_team/uni_v2_aug/l0_0p25mpp_40x'))
    p.add_argument('--stain-csv', type=Path, default=Path('dataset_csv/stain_aug_labels.csv'))
    p.add_argument('--label-source', choices=['manual_only', 'manual_or_filename'], default='manual_only')
    p.add_argument('--policy', choices=['geometry', 'geometry_he_scale_light'], default='geometry')
    p.add_argument('--total-views', type=int, choices=[2, 3, 5], default=2)
    p.add_argument('--workers', default='acr_high,amr,significant')
    p.add_argument('--seeds', default='1,11,21,31,41')
    p.add_argument('--folds', type=int, choices=[3, 5], default=5)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--bank-seed', type=int, default=20260917)
    p.add_argument('--view-seed', type=int, default=20260918)
    p.add_argument('--max-epochs', type=int, default=50)
    p.add_argument('--early-stop-patience', type=int, default=10)
    p.add_argument('--early-stop-min-epoch', type=int, default=10)
    p.add_argument('--restart-incomplete', action='store_true')
    p.add_argument('--details', action='store_true')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if set(args.workers.split(',')) - TASKS.keys():
        p.error('workers must be a comma-separated subset of acr_high,amr,significant')
    if args.batch_size < 1 or args.max_epochs < 1 or args.early_stop_patience < 1 or args.early_stop_min_epoch < 0:
        p.error('Invalid batch/epoch/patience settings')
    args.project = args.project.resolve()
    args.feature_root = (args.project / args.feature_root).resolve()
    args.bank_root = args.bank_root.resolve()
    folds, slides, hashes = build_inputs(args)
    index = find_banks(args, slides)
    missing = slides.keys() - index.keys()
    estimate = sum((args.feature_root / SCALE / 'pt_files' / f'{sid}.pt').stat().st_size
                   for sid in missing) * (args.total_views - 1)
    free = shutil.disk_usage(nearest_existing(args.bank_root)).free
    print(f'Confirmed HE: {len(slides)}; reusable full banks: {len(index)}; need generation: {len(missing)}')
    print(f'Approx additional feature storage: {estimate/2**30:.2f} GiB; free: {free/2**30:.2f} GiB', flush=True)
    if args.command == 'audit':
        return
    if args.command == 'prepare' and free < estimate * 1.2 + 2 * 2**30:
        raise RuntimeError('Insufficient feature storage with safety margin')
    if args.command == 'prepare':
        with lock_file(args.bank_root / 'gold_shared_v1.prepare.lock'):
            prepare(args, folds, slides, index, hashes)
    else:
        train(args, folds)


if __name__ == '__main__':
    main()
