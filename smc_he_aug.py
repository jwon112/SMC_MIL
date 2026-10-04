"""Train-only HE view adapter; leaves the installed SMC training loop untouched.

Used as main.py's entry point by an isolated copy of run_smc_cv_grid.sh.
No original source file, split, label, baseline feature or checkpoint is edited.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import runpy
from collections import Counter
from pathlib import Path

from feature_view_bank import ViewSelector, feature_path, torch_feature_loader, validate_feature_bank


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen_inputs(config):
    for path, expected in config['input_sha256'].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise RuntimeError(f'Input changed after preflight: {path}; rerun audit/prepare')


class HETrainViewSplit:
    """Keep every original training row and its sampling weight/class label.

    Only eligible HE rows consume the private view RNG. Original draws and all
    other stains use the installed dataset's __getitem__, unchanged.
    """
    def __init__(self, base, config, seed, log_path):
        if getattr(base, 'use_h5', False) or isinstance(base.data_dir, dict):
            raise ValueError('This adapter supports single-root PT bags only (no MAQW/H5)')
        if Path(base.data_dir).resolve() != Path(config['baseline_dir']).resolve():
            raise ValueError('Training feature root differs from the audited baseline')
        self.base = base
        self.config = config
        self.slide_data = base.slide_data
        self.slide_cls_ids = base.slide_cls_ids
        self.selector = ViewSelector(config['total_views'], seed, 0.5)
        self.eligible = set(config['eligible_he']) & set(self.slide_data.slide_id.astype(str))
        missing = self.eligible - config['bank_index'].keys()
        if missing:
            raise RuntimeError(f'Eligible HE views missing: {sorted(missing)[:10]}; run prepare first')
        self.counts = Counter()
        self.draws = 0
        self.handle = Path(log_path).open('x', newline='', encoding='utf-8')
        self.writer = csv.writer(self.handle)
        self.writer.writerow(['draw', 'slide_id', 'label', 'eligible_he', 'view', 'feature_path'])

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def getlabel(self, index):
        return self.base.getlabel(index)

    def __getitem__(self, index):
        import torch
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError('Use num_workers=0; private draw logging is single-process')
        row = self.slide_data.iloc[index]
        sid = str(row['slide_id'])
        is_he = sid in self.eligible
        view = self.selector.draw() if is_he else 0
        bank = Path(self.config['bank_index'][sid]['root']) if is_he else None
        path = feature_path(Path(self.config['baseline_dir']), bank, self.config['policy'], sid, view)
        if view == 0:
            result = self.base[index]
        else:
            tensor = torch_feature_loader(path)
            expected = self.config['bank_index'][sid]['patch_count']
            if tensor.shape != (expected, 1536):
                raise ValueError(f'Augmented bag shape differs from preflight: {path}')
            result = tensor, self.base.getlabel(index)
        self.writer.writerow([self.draws, sid, row['label'], int(is_he), view, str(path)])
        self.handle.flush()
        self.counts[f'{"HE" if is_he else "unchanged"}_view_{view}'] += 1
        self.draws += 1
        return result

    def close(self):
        self.handle.close()


def wrap_train(original_train, config):
    def train(datasets, cur, args, *positional, **kwargs):
        # main.py uses np.arange for folds. Normalize before JSON logging and
        # seed arithmetic; np.int64 is not serializable by json.dumps.
        cur = int(cur)
        training_seed = int(args.seed)
        if positional or kwargs.get('world_size', 1) != 1 or getattr(args, 'distributed', False):
            raise ValueError('This adapter supports the single-GPU grid only')
        if getattr(args, 'use_maqw', False) or getattr(args, 'use_ddpm_denoise', False):
            raise ValueError('MAQW/DDPM are outside this fixed baseline comparison')
        check_frozen_inputs(config)
        fold_key = f'{args.task}:{args.seed}:{cur}'
        expected = config['folds'][fold_key]
        base, val, test = datasets
        for split, key in ((base, 'train'), (val, 'val'), (test, 'test')):
            actual = [] if split is None else split.slide_data.slide_id.astype(str).tolist()
            if len(actual) != len(set(actual)) or set(actual) != set(expected[key]):
                raise ValueError(f'Actual {key} cohort differs from audited gold split: {fold_key}')
        for sid in set(base.slide_data.slide_id.astype(str)) & set(config['eligible_he']):
            entry = config['bank_index'][sid]
            for path, sha in entry['metadata_sha256'].items():
                if digest(path) != sha:
                    raise ValueError(f'Bank metadata changed: {path}')
            validate_feature_bank([{'slide_id': sid, 'stain_group': 'HE',
                                    'source_dataset': entry['source_dataset']}],
                                  baseline_dir=Path(config['baseline_dir']),
                                  bank_root=Path(entry['root']), policy=config['policy'],
                                  total_views=config['total_views'])
        result_dir = Path(args.results_dir)
        # Distinct from model/sampler RNG; same draws across geometry/H-E policy runs.
        seed = int(config['view_seed']) + training_seed * 1009 + cur * 100003
        wrapped = HETrainViewSplit(base, config, seed, result_dir / f'he_view_draws_{cur}.csv')
        info = {'fold': cur, 'seed': training_seed, 'train_slides': len(base),
                'eligible_he_train': len(wrapped.eligible),
                'unchanged_train': len(base) - len(wrapped.eligible),
                'policy': config['policy'], 'total_views': config['total_views'],
                'he_original_probability': 0.5, 'view_seed': seed,
                'val_test_original_only': True, 'split_sha256': expected['sha256'],
                'validation_used_for_checkpoint_selection': True,
                'independent_final_test': False}
        try:
            print('HE augmentation:', json.dumps(info), flush=True)
            result = original_train((wrapped, val, test), cur, args, **kwargs)
            info['completed'] = True
            return result
        finally:
            wrapped.close()
            info['draw_counts'] = dict(wrapped.counts)
            info['draws'] = wrapped.draws
            (result_dir / f'he_augmentation_{cur}.json').write_text(json.dumps(info, indent=2) + '\n')
    return train


def main():
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        runpy.run_path('main.py', run_name='__main__')
        return
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('DDP is not supported by this comparison adapter')
    config_path = os.environ.get('SMC_HE_AUG_CONFIG')
    if not config_path:
        raise RuntimeError('Run via run_smc_gold_he_aug.py train, not directly')
    config = json.loads(Path(config_path).read_text())
    check_frozen_inputs(config)
    import utils.core_utils as core
    original = core.train
    core.train = wrap_train(original, config)
    try:
        # main.py imports the adapter in this process only; all CLI options and
        # the installed core training/checkpoint/summary implementation remain.
        runpy.run_path('main.py', run_name='__main__')
    finally:
        core.train = original


if __name__ == '__main__':
    main()
