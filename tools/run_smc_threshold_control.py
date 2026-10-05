"""Nested patient-grouped ACR threshold control; outer patients never select epochs or cutoffs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.threshold_control import (assert_disjoint, choose_threshold, decision_metrics,
                                     patient_folds, stop_partition, validate_events)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def outer_partitions(events, split_dir):
    result = []; seen = []
    for fold in range(5):
        split = pd.read_csv(split_dir / f'splits_{fold}.csv', dtype=str)
        train_ids = set(split.train.dropna()); test_ids = set(split.val.dropna())
        if train_ids | test_ids != set(events.event_id) or train_ids & test_ids:
            raise ValueError('Outer split must partition every event exactly once')
        train = events[events.event_id.isin(train_ids)].copy()
        test = events[events.event_id.isin(test_ids)].copy()
        assert_disjoint(train, test)
        if train.label.nunique() != 2:
            raise ValueError('Outer training lacks a class')
        seen.extend(test.event_id); result.append((train, test))
    if len(seen) != len(events) or len(set(seen)) != len(seen):
        raise ValueError('Outer test events must have exactly one fold')
    return result


def nested_plan(train, test, seed, fold):
    rows = []; parts = []
    for inner, (development, calibration) in enumerate(patient_folds(train, 3, seed + 100003*fold)):
        fit, stop = stop_partition(development, seed + 100003*fold + 1009*(inner+1))
        assert_disjoint(fit, stop, calibration, test)
        parts.append((fit, stop, calibration))
        for role, frame in [('fit', fit), ('stop', stop), ('calibration', calibration), ('outer_test', test)]:
            for row in frame.itertuples():
                rows.append(dict(inner=inner, role=role, event_id=row.event_id,
                                 case_id=row.case_id, label=int(row.label)))
    return parts, pd.DataFrame(rows)


def fit_model(train, stop, slides, args, seed, dest, fixed_epochs=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from models.stain_aware_event_mil import StainAwareEventMIL
    from train_smc_event_stain_mil import evaluate, seed_all
    from utils.event_mil import EventFeatureDataset, collate_event
    seed_all(seed); dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = StainAwareEventMIL(1536, 128, .25, use_stain_branches=True,
                              include_presence_masks=True, presence_mask_values='observed').to(device)
    def loader(frame, training):
        dataset = EventFeatureDataset(frame, slides, args.feature_dir, 2048, training)
        if training:
            counts = frame.label.value_counts()
            weights = frame.label.map({k: 1/v for k, v in counts.items()}).to_numpy()
            return DataLoader(dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights), replacement=True), collate_fn=collate_event)
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_event)
    train_loader = loader(train, True)
    stop_loader = loader(stop, False) if stop is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(); best = float('inf'); best_epoch = 0; stale = 0; logs = []
    checkpoint = dest / 'checkpoint.pt'
    for epoch in range(fixed_epochs or args.max_epochs):
        model.train(); losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model([(stain, features.to(device)) for stain, features in batch['slides']])
            loss = loss_fn(logits, torch.tensor([batch['label']], device=device))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        row = dict(epoch=epoch+1, train_loss=float(np.mean(losses)))
        if stop_loader is not None:
            val_loss, _ = evaluate(model, stop_loader, device); row['stop_loss'] = val_loss
            if not np.isfinite(val_loss):
                raise ValueError('Nonfinite stopping loss')
            if val_loss < best:
                best = val_loss; best_epoch = epoch+1; stale = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
        logs.append(row); print(f'[EPOCH] {dest.name} {row}', flush=True)
        if stop_loader is not None and epoch+1 >= args.min_epochs and stale >= args.patience:
            break
    pd.DataFrame(logs).to_csv(dest/'epochs.csv', index=False)
    if stop_loader is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        best_epoch = fixed_epochs; torch.save(model.state_dict(), checkpoint)
    dump(dest/'fit_record.json', dict(seed=seed, selected_epoch=best_epoch,
                                    selection='stopping_loss' if stop is not None else 'fixed_inner_median'))
    return model, best_epoch, loader


def run_fold(train, test, slides, args, seed, fold, dest):
    import torch
    from train_smc_event_stain_mil import evaluate
    parts, partition = nested_plan(train, test, seed, fold)
    partition.to_csv(dest/'partitions.csv', index=False)
    inner_frames = []; epochs = []
    for inner, (fit, stop, calibration) in enumerate(parts):
        print(f'[RUN] seed={seed} outer={fold} inner={inner}', flush=True)
        model, epoch, loader = fit_model(fit, stop, slides, args, seed + 100003*fold + 1009*(inner+1), dest/f'inner_{inner}')
        _, pred = evaluate(model, loader(calibration, False), torch.device(args.device))
        pred['inner'] = inner; inner_frames.append(pred); epochs.append(epoch)
        del model
    inner_oof = pd.concat(inner_frames, ignore_index=True)
    if inner_oof.event_id.duplicated().any() or set(inner_oof.event_id) != set(train.event_id):
        raise ValueError('Inner calibration coverage mismatch')
    inner_oof.to_csv(dest/'inner_oof_predictions.csv', index=False)
    thresholds = dict(fixed_0p5=.5, inner_f1=choose_threshold(inner_oof, 1), inner_f2=choose_threshold(inner_oof, 2))
    # All outer training patients are refitted for an epoch count chosen only inside that training set.
    refit_epochs = int(np.median(epochs))
    dump(dest/'thresholds.json', dict(thresholds=thresholds, inner_best_epochs=epochs,
                                    refit_epochs=refit_epochs, primary='inner_f2'))
    print(f'[REFIT] seed={seed} outer={fold} epochs={refit_epochs} thresholds={thresholds}', flush=True)
    model, _, loader = fit_model(train, None, slides, args, seed + 100003*fold, dest/'refit', fixed_epochs=refit_epochs)
    _, outer = evaluate(model, loader(test, False), torch.device(args.device))
    if not outer.probability.between(0, 1).all():
        raise ValueError('Invalid outer probabilities')
    outer['fold'] = fold
    for rule, threshold in thresholds.items():
        outer[f'threshold_{rule}'] = threshold
        outer[f'prediction_{rule}'] = (outer.probability >= threshold).astype(int)
    # Written last: this marks a completed outer fold. Interrupted folds are retrained.
    outer.to_csv(dest/'outer_predictions.csv', index=False)
    print(f'[OK] seed={seed} outer={fold}', flush=True)


def completed_frame(dest, test):
    frame = pd.read_csv(dest/'outer_predictions.csv', dtype={'event_id': str, 'case_id': str}, float_precision='round_trip')
    validate_events(frame)
    cols = ['case_id', 'label']
    if set(frame.event_id) != set(test.event_id) or not frame.set_index('event_id')[cols].sort_index().equals(test.set_index('event_id')[cols].sort_index()):
        raise ValueError('Completed outer patient/label coverage differs')
    thresholds = json.loads((dest/'thresholds.json').read_text())['thresholds']
    if not frame.probability.between(0, 1).all():
        raise ValueError('Invalid completed predictions')
    for rule, threshold in thresholds.items():
        if not frame[f'threshold_{rule}'].eq(threshold).all() or not frame[f'prediction_{rule}'].eq((frame.probability >= threshold).astype(int)).all():
            raise ValueError('Completed decision mismatch')
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['audit', 'run', 'summarize'], default='audit')
    p.add_argument('--manifest-root', type=Path, default=Path('/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930'))
    p.add_argument('--feature-dir', type=Path, default=Path('/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2/l0_0p25mpp_40x'))
    p.add_argument('--results-root', type=Path, default=PROJECT/'results/smc_event_threshold_control_20261005_exclude25')
    p.add_argument('--gpu', default='1'); p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 11, 21, 31, 41])
    p.add_argument('--max-epochs', type=int, default=50); p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=10)
    args = p.parse_args()
    if args.max_epochs < args.min_epochs or min(args.min_epochs, args.patience) < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Invalid epochs, patience or repeated seeds')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from utils.event_mil import load_event_tables, verify_feature_bags
    task = args.manifest_root.resolve()/'acr_high'; args.feature_dir = args.feature_dir.resolve()
    output = args.results_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    events, slides = load_event_tables(task/'events.csv', task/'event_slides.csv'); validate_events(events)
    paths = [task/'events.csv', task/'event_slides.csv']
    paths += [task/'splits'/f'seed{s}'/f'splits_{f}.csv' for s in args.seeds for f in range(5)]
    paths += [Path(__file__), PROJECT/'utils/threshold_control.py', PROJECT/'utils/event_mil.py',
              PROJECT/'models/stain_aware_event_mil.py', PROJECT/'train_smc_event_stain_mil.py']
    config = dict(protocol='nested3_stop_patient_refit_v1', seeds=args.seeds, feature_dir=str(args.feature_dir), device=args.device,
                  max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience,
                  primary='inner_f2', secondary='inner_f1', patch_cap=2048, mask='observed',
                  input_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    # Lightweight identity checks for large immutable feature bags, not content hashes.
    config['feature_inventory'] = {}
    for slide_id in sorted(set(slides.slide_id)):
        stat = (args.feature_dir/'pt_files'/f'{slide_id}.pt').stat()
        config['feature_inventory'][slide_id] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    config_path = output/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Protocol or inputs changed; use a new results root')
    dump(config_path, config)
    if args.action != 'summarize':
        verify_feature_bags(slides, args.feature_dir)
    audit = []; seed_records = []; fold_records = []; changes = []
    for seed in args.seeds:
        frames = []
        for fold, (train, test) in enumerate(outer_partitions(events, task/'splits'/f'seed{seed}')):
            parts, plan = nested_plan(train, test, seed, fold)
            dest = output/f'seed{seed}'/f'outer_{fold}'; dest.mkdir(parents=True, exist_ok=True)
            for inner, (fit, stop, calibration) in enumerate(parts):
                row = dict(seed=seed, outer=fold, inner=inner)
                for name, frame in [('fit', fit), ('stop', stop), ('calibration', calibration), ('test', test)]:
                    row.update({f'{name}_events': len(frame), f'{name}_positive': int(frame.label.sum()),
                                f'{name}_positive_patients': frame.loc[frame.label == 1, 'case_id'].nunique()})
                audit.append(row)
            if args.action == 'audit':
                plan.to_csv(dest/'partitions.csv', index=False); continue
            if not (dest/'outer_predictions.csv').exists():
                if args.action == 'summarize':
                    raise FileNotFoundError(f'Incomplete outer fold: {dest}')
                run_fold(train, test, slides, args, seed, fold, dest)
            else:
                print(f'[SKIP] seed={seed} outer={fold}', flush=True)
            frame = completed_frame(dest, test); frames.append(frame)
            for rule in ['fixed_0p5', 'inner_f1', 'inner_f2']:
                fold_records.append(dict(seed=seed, fold=fold, rule=rule,
                                         threshold=float(frame[f'threshold_{rule}'].iloc[0]),
                                         **decision_metrics(frame, frame[f'prediction_{rule}'])))
        if args.action != 'audit':
            oof = pd.concat(frames, ignore_index=True)
            oof.to_csv(output/f'seed{seed}'/'oof_predictions.csv', index=False)
            for rule in ['fixed_0p5', 'inner_f1', 'inner_f2']:
                seed_records.append(dict(seed=seed, rule=rule, **decision_metrics(oof, oof[f'prediction_{rule}'])))
                changed = oof[oof[f'prediction_{rule}'] != oof.prediction_fixed_0p5].copy()
                changed['seed'] = seed; changed['rule'] = rule; changes.append(changed)
    pd.DataFrame(audit).to_csv(output/'partition_audit.csv', index=False)
    if args.action == 'audit':
        print(f'[AUDIT OK] {len(events)} events / {int(events.label.sum())} positives; {len(args.seeds)*20} model fits planned', flush=True)
        print(pd.DataFrame(audit).to_string(index=False)); return
    pd.DataFrame(fold_records).to_csv(output/'fold_metrics.csv', index=False)
    record = pd.DataFrame(seed_records); record.to_csv(output/'per_seed_metrics.csv', index=False)
    pd.concat(changes, ignore_index=True).to_csv(output/'changed_calls.csv', index=False)
    numeric = ['tp', 'fp', 'fn', 'precision', 'sensitivity', 'specificity', 'f1', 'f2', 'mcc', 'auroc', 'pr_auc']
    summary = record.groupby('rule')[numeric].agg(['mean', 'std'])
    summary.to_csv(output/'seed_summary.csv'); print(summary.to_string())
    base = record[record.rule == 'fixed_0p5'].set_index('seed')
    deltas = []
    for row in record[record.rule != 'fixed_0p5'].itertuples():
        deltas.append(dict(seed=row.seed, rule=row.rule, **{k: getattr(row, k)-base.loc[row.seed, k] for k in numeric}))
    pd.DataFrame(deltas).to_csv(output/'paired_seed_deltas.csv', index=False)


if __name__ == '__main__':
    main()
