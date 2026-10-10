"""Compare agnostic attention with completed stain-aware recovery on 578 events."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from prepare_smc_research_stains import digest
from run_smc_research_stains import read, same_mapping
from utils.threshold_control import assert_disjoint, decision_metrics

METRICS = ['auroc', 'pr_auc', 'tp', 'fp', 'fn', 'precision', 'sensitivity',
           'specificity', 'f1', 'f2', 'mcc']


def validated_prediction(path, expected, fold=None):
    frame = read(path)
    same_mapping(frame, expected)
    if not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
        raise ValueError(f'Invalid probabilities: {path}')
    if fold is not None and not frame.fold.eq(fold).all():
        raise ValueError(f'Incorrect fold: {path}')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['audit', 'run', 'summarize'], default='audit')
    parser.add_argument('--manifest-root', type=Path, default=PROJECT/'results/smc_stain_research_20261009')
    parser.add_argument('--aware-root', type=Path, default=PROJECT/'results/smc_event_research_stains_20261009')
    parser.add_argument('--results-root', type=Path, default=PROJECT/'results/smc_event_stain_control_20261010')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.results_root.resolve() in {args.aware_root.resolve(), args.manifest_root.resolve()}:
        raise ValueError('Use a separate output directory')
    prepared = json.loads((args.manifest_root/'manifest.json').read_text(encoding='utf-8'))
    aware = json.loads((args.aware_root/'protocol.json').read_text(encoding='utf-8'))
    if aware['cohorts'] != ['restored7'] or aware['condition'] != 'balanced_sampler' or aware['threshold'] != .5:
        raise ValueError('Expected completed restored7 balanced-sampler aware reference')
    if aware['prepared_outputs_sha256_lf'] != prepared['output_sha256_lf'] or aware['seeds'] != prepared['seeds']:
        raise ValueError('Prepared inputs differ from completed aware reference')
    if prepared['seeds'] != [1, 11, 21, 31, 41]:
        raise ValueError('Expected the five prespecified recovery seeds')
    # Refuse comparisons after a frozen dependency or saved input was changed.
    for path, expected in aware['source_sha256_lf'].items():
        if digest(Path(path)) != expected:
            raise ValueError(f'Aware reference dependency changed: {path}')
    for relative, expected in prepared['output_sha256_lf'].items():
        if digest(args.manifest_root/relative) != expected:
            raise ValueError(f'Prepared input changed: {relative}')
    events = read(args.manifest_root/'restored7/acr_high/events.csv')
    slides = read(args.manifest_root/'restored7/acr_high/event_slides.csv')
    old_events = read(args.manifest_root/'baseline575/acr_high/events.csv')
    same_mapping(events[events.event_id.isin(old_events.event_id)], old_events)
    if len(events) != 578 or events.label.sum() != 17 or len(slides) != 1269:
        raise ValueError('Unexpected recovery cohort')
    if slides.slide_id.duplicated().any() or set(slides.event_id) != set(events.event_id):
        raise ValueError('Invalid slide membership')
    if not set(slides.stain_group) <= {'HE', 'IHC', 'other'}:
        raise ValueError('Unknown stain would drop slides')
    sources = [Path(__file__), PROJECT/'tools/recovered_stain_control.py',
               args.aware_root/'protocol.json', args.manifest_root/'manifest.json']
    sources += [Path(p) for p in aware['source_sha256_lf']]
    partitions = {}; aware_oof = {}
    for seed in prepared['seeds']:
        predictions = []
        for fold in range(5):
            part_file = args.manifest_root/'partitions'/f'seed{seed}'/f'fold_{fold}'/'partitions.csv'
            part = read(part_file); same_mapping(part, events)
            if set(part.role) != {'fit', 'stop', 'test'}:
                raise ValueError('Invalid partition roles')
            roles = {role: part[part.role.eq(role)].sort_values('event_id').reset_index(drop=True)
                     for role in ('fit', 'stop', 'test')}
            assert_disjoint(*roles.values())
            if any(roles[r].label.nunique() != 2 for r in ('fit', 'stop')):
                raise ValueError('Fit or stop lacks a class')
            partitions[seed, fold] = roles
            reference = args.aware_root/'restored7'/f'seed{seed}'/f'fold_{fold}'
            config = json.loads((reference/'fit_config.json').read_text())
            if (config['seed'], config['fold'], config['condition'], config['mask'], config['patch_cap']) != (seed, fold, 'balanced_sampler', 'observed', 2048):
                raise ValueError('Unexpected aware fit settings')
            if config['fit_events'] != len(roles['fit']) or config['fit_positive'] != int(roles['fit'].label.sum()):
                raise ValueError('Aware fit population differs')
            predictions.append(validated_prediction(reference/'outer_predictions.csv', roles['test'], fold))
            sources += [part_file, reference/'fit_config.json', reference/'outer_predictions.csv',
                        reference/'selection.json', reference/'training_draws.csv']
        combined = pd.concat(predictions, ignore_index=True); same_mapping(combined, events)
        saved_oof = args.aware_root/'restored7'/f'seed{seed}'/'oof_predictions.csv'
        saved = validated_prediction(saved_oof, events)
        pd.testing.assert_frame_equal(combined.sort_values('event_id').reset_index(drop=True),
                                      saved.sort_values('event_id').reset_index(drop=True))
        sources.append(saved_oof); aware_oof[seed] = combined
    args.feature_dir = Path(aware['feature_dir']).resolve()
    inventory = {}; missing = []; changed = []
    if set(slides.slide_id) != set(aware['feature_inventory']):
        raise ValueError('Aware feature inventory has different slide membership')
    for sid in sorted(set(slides.slide_id)):
        file = args.feature_dir/'pt_files'/f'{sid}.pt'
        if not file.is_file():
            missing.append(sid); continue
        stat = file.stat(); current = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        inventory[sid] = current
        if current != aware['feature_inventory'][sid]:
            changed.append(sid)
    audit = dict(events=len(events), positive_events=int(events.label.sum()), slides=len(slides),
                 seeds=prepared['seeds'], new_model_fits=25, reused_aware_fits=25,
                 roles='exact saved fit/stop/test', agnostic_presence_input='constant 1',
                 feature_files_found=len(inventory), missing_feature_files=len(missing),
                 changed_feature_files=len(changed), ready_to_train=not missing and not changed,
                 comparison='agnostic minus aware on identical evaluation events')
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    if args.action == 'audit':
        print(json.dumps(audit, indent=2)); return
    if args.action == 'run' and not audit['ready_to_train']:
        raise ValueError('Feature files missing or changed since aware training; inspect audit')
    args.max_epochs = aware['max_epochs']; args.min_epochs = aware['min_epochs']; args.patience = aware['patience']
    protocol = dict(version=1, seeds=prepared['seeds'], threshold=.5, mode='agnostic',
                    device=args.device, feature_dir=str(args.feature_dir),
                    max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience,
                    prepared_outputs_sha256_lf=prepared['output_sha256_lf'],
                    source_sha256_lf={str(p.resolve()): digest(p) for p in sources},
                    aware_feature_inventory=aware['feature_inventory'],
                    sampling='identical event draws in common epochs; patch draws can differ',
                    architecture='one branch and constant presence input; parameter count differs')
    record = args.results_root/'protocol.json'
    if record.exists():
        if json.loads(record.read_text()) != protocol:
            raise ValueError('Control inputs/settings changed; use a new results root')
    elif args.action == 'summarize':
        raise FileNotFoundError('Control training protocol missing')
    else:
        record.write_text(json.dumps(protocol, indent=2)+'\n')
    from recovered_stain_control import fit_agnostic
    rows = []; deltas = []; recovered = []
    for seed in prepared['seeds']:
        predictions = []
        for fold in range(5):
            roles = partitions[seed, fold]
            dest = args.results_root/'agnostic'/f'seed{seed}'/f'fold_{fold}'
            reference = args.aware_root/'restored7'/f'seed{seed}'/f'fold_{fold}'
            dest.mkdir(parents=True, exist_ok=True)
            prediction = dest/'outer_predictions.csv'
            if not prediction.exists():
                if args.action == 'summarize':
                    raise FileNotFoundError(f'Incomplete control: {prediction}')
                print(f'[RUN] agnostic seed={seed} fold={fold}', flush=True)
                fit_agnostic(roles['fit'], roles['stop'], roles['test'], slides, args, seed, fold, reference, dest)
            config = json.loads((dest/'fit_config.json').read_text())
            if (config['mode'], config['seed'], config['fold'], config['use_stain_branches']) != ('agnostic', seed, fold, False):
                raise ValueError('Invalid completed control fit')
            predictions.append(validated_prediction(prediction, roles['test'], fold))
        current = pd.concat(predictions, ignore_index=True); same_mapping(current, events)
        current.to_csv(args.results_root/'agnostic'/f'seed{seed}'/'oof_predictions.csv', index=False)
        for scope, ids in [('recovered578', set(events.event_id)), ('common575', set(old_events.event_id))]:
            metrics = {}
            for mode, frame in [('aware', aware_oof[seed]), ('agnostic', current)]:
                subset = frame[frame.event_id.isin(ids)]
                metrics[mode] = decision_metrics(subset, subset.probability >= .5)
                rows.append(dict(seed=seed, scope=scope, mode=mode, **metrics[mode]))
            deltas.append(dict(seed=seed, scope=scope,
                               **{key: metrics['agnostic'][key]-metrics['aware'][key] for key in METRICS}))
        for mode, frame in [('aware', aware_oof[seed]), ('agnostic', current)]:
            subset = frame[~frame.event_id.isin(old_events.event_id)].copy()
            subset['seed'] = seed; subset['mode'] = mode; recovered.append(subset)
    pd.DataFrame(rows).to_csv(args.results_root/'per_seed_metrics.csv', index=False)
    pd.DataFrame(deltas).to_csv(args.results_root/'paired_mode_deltas.csv', index=False)
    pd.concat(recovered, ignore_index=True).to_csv(args.results_root/'recovered3_predictions.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['scope', 'mode'])[METRICS].agg(['mean', 'std'])
    summary.to_csv(args.results_root/'seed_summary.csv'); print(summary.to_string())
    print('[COMPLETE] 25 agnostic fits; deltas are agnostic minus aware; identical 578/575 evaluation scopes')


if __name__ == '__main__':
    main()
