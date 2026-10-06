"""Paired sampling/loss control with independent patient stopping and outer evaluation."""
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
from utils.threshold_control import assert_disjoint, decision_metrics, stop_partition, validate_events
from run_smc_threshold_control import outer_partitions


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def fit_and_evaluate(fit, stop, test, slides, args, seed, fold, condition, dest):
    import torch
    from torch.utils.data import DataLoader
    from models.stain_aware_event_mil import StainAwareEventMIL
    from train_smc_event_stain_mil import seed_all, evaluate
    from utils.event_mil import EventFeatureDataset, collate_event
    from utils.imbalance_control import balanced_weights, make_sampler, model_hash, training_loss
    seed_all(seed+100003*fold); device = torch.device(args.device)
    model = StainAwareEventMIL(1536, 128, .25, use_stain_branches=True,
                              include_presence_masks=True, presence_mask_values='observed').to(device)
    initial_hash = model_hash(model)
    weights = balanced_weights(fit.label)
    def evaluation_loader(frame):
        return DataLoader(EventFeatureDataset(frame, slides, args.feature_dir, 2048, False),
                          batch_size=1, shuffle=False, collate_fn=collate_event)
    data = EventFeatureDataset(fit, slides, args.feature_dir, 2048, True)
    generator = torch.Generator().manual_seed(seed+100003*fold+17)
    sampler = make_sampler(fit.label, condition, generator)
    train_loader = DataLoader(data, batch_size=1, sampler=sampler, collate_fn=collate_event)
    stop_loader = evaluation_loader(stop)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    best = float('inf'); best_epoch = 0; stale = 0; logs = []; draws = []
    checkpoint = dest/'checkpoint.pt'
    config = dict(seed=seed, fold=fold, condition=condition, initialization_sha256=initial_hash,
                  class_weights=weights.tolist(), fit_events=len(fit), fit_positive=int(fit.label.sum()),
                  stopping_objective='unweighted CE', mask='observed', patch_cap=2048)
    write_json(dest/'fit_config.json', config)
    for epoch in range(args.max_epochs):
        model.train(); losses = []; selected = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model([(stain, feature.to(device)) for stain, feature in batch['slides']])
            label = torch.tensor([batch['label']], device=device)
            loss = training_loss(logits, label, condition, weights)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            selected.append((batch['event_id'], batch['label']))
        stop_loss, _ = evaluate(model, stop_loader, device)
        if not np.isfinite(stop_loss) or not np.isfinite(losses).all():
            raise ValueError('Nonfinite training or stopping loss')
        row = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), stop_loss=stop_loss,
                   draws=len(selected), positive_draws=sum(y for _, y in selected),
                   unique_events=len({event for event, _ in selected}))
        logs.append(row)
        draws.extend(dict(epoch=epoch+1, step=step, event_id=event, label=y)
                     for step, (event, y) in enumerate(selected))
        print(f'[EPOCH] seed={seed} fold={fold} {condition} {row}', flush=True)
        if stop_loss < best:
            best = stop_loss; best_epoch = epoch+1; stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if epoch+1 >= args.min_epochs and stale >= args.patience:
            break
    pd.DataFrame(logs).to_csv(dest/'epochs.csv', index=False)
    pd.DataFrame(draws).to_csv(dest/'training_draws.csv', index=False)
    write_json(dest/'selection.json', dict(best_epoch=best_epoch, best_stop_loss=best, epochs_run=len(logs)))
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    _, predictions = evaluate(model, evaluation_loader(test), device)
    predictions['fold'] = fold
    if not predictions.probability.between(0, 1).all():
        raise ValueError('Invalid outer probabilities')
    # Final file marks completion. No outer loss was used to select the checkpoint.
    predictions.to_csv(dest/'outer_predictions.csv', index=False)
    print(f'[OK] seed={seed} fold={fold} {condition}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['audit', 'run', 'summarize'], default='audit')
    p.add_argument('--gpu', default='1'); p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--manifest-root', type=Path, default=Path('/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930'))
    p.add_argument('--feature-dir', type=Path, default=Path('/home/jupyter/image_team/projects/SMC_MIL/data/features/uni_v2/l0_0p25mpp_40x'))
    p.add_argument('--results-root', type=Path, default=PROJECT/'results/smc_event_imbalance_control_20261006_exclude25')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 11, 21, 31, 41])
    p.add_argument('--max-epochs', type=int, default=50); p.add_argument('--min-epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=10)
    args = p.parse_args()
    if args.max_epochs < args.min_epochs or min(args.min_epochs, args.patience) < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Invalid training settings or repeated seeds')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    from utils.event_mil import load_event_tables, verify_feature_bags
    from utils.imbalance_control import CONDITIONS, balanced_weights
    task = args.manifest_root.resolve()/'acr_high'; args.feature_dir = args.feature_dir.resolve()
    out = args.results_root.resolve(); out.mkdir(parents=True, exist_ok=True)
    events, slides = load_event_tables(task/'events.csv', task/'event_slides.csv'); validate_events(events)
    sources = [task/'events.csv', task/'event_slides.csv']
    sources += [task/'splits'/f'seed{s}'/f'splits_{f}.csv' for s in args.seeds for f in range(5)]
    sources += [Path(__file__), PROJECT/'utils/imbalance_control.py', PROJECT/'utils/threshold_control.py',
                PROJECT/'utils/event_mil.py', PROJECT/'models/stain_aware_event_mil.py',
                PROJECT/'train_smc_event_stain_mil.py', PROJECT/'tools/run_smc_threshold_control.py']
    protocol = dict(version=1, seeds=args.seeds, conditions=CONDITIONS, device=args.device,
                    max_epochs=args.max_epochs, min_epochs=args.min_epochs, patience=args.patience,
                    feature_dir=str(args.feature_dir), threshold=.5, stopping='unweighted CE on separate stop patients',
                    class_weight='N_fit/(2*N_fit_class); unreduced CE times sample weight',
                    input_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
    protocol['conditions'] = list(CONDITIONS)
    protocol['feature_inventory'] = {}
    for slide_id in sorted(set(slides.slide_id)):
        stat = (args.feature_dir/'pt_files'/f'{slide_id}.pt').stat()
        protocol['feature_inventory'][slide_id] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    record = out/'protocol.json'
    if record.exists() and json.loads(record.read_text()) != protocol:
        raise ValueError('Inputs/code/settings changed; use a new results root')
    write_json(record, protocol)
    if args.action != 'summarize':
        verify_feature_bags(slides, args.feature_dir)
    audit = []; results = []
    for seed in args.seeds:
        frames = {condition: [] for condition in CONDITIONS}
        for fold, (outer_train, test) in enumerate(outer_partitions(events, task/'splits'/f'seed{seed}')):
            fit, stop = stop_partition(outer_train, seed+100003*fold+1009)
            assert_disjoint(fit, stop, test)
            w = balanced_weights(fit.label).tolist()
            row = dict(seed=seed, fold=fold, weight_negative=w[0], weight_positive=w[1])
            for role, frame in [('fit', fit), ('stop', stop), ('test', test)]:
                row.update({f'{role}_events': len(frame), f'{role}_positive': int(frame.label.sum()),
                            f'{role}_positive_patients': frame.loc[frame.label == 1, 'case_id'].nunique()})
            audit.append(row)
            partition = pd.concat([frame.assign(role=role) for role, frame in [('fit', fit), ('stop', stop), ('test', test)]])
            folder = out/f'seed{seed}'/f'fold_{fold}'; folder.mkdir(parents=True, exist_ok=True)
            partition[['event_id', 'case_id', 'label', 'role']].to_csv(folder/'partitions.csv', index=False)
            if args.action == 'audit':
                continue
            hashes = []
            for condition in CONDITIONS:
                dest = folder/condition; dest.mkdir(parents=True, exist_ok=True)
                if not (dest/'outer_predictions.csv').exists():
                    if args.action == 'summarize':
                        raise FileNotFoundError(f'Incomplete run: {dest}')
                    fit_and_evaluate(fit, stop, test, slides, args, seed, fold, condition, dest)
                else:
                    print(f'[SKIP] seed={seed} fold={fold} {condition}', flush=True)
                f = pd.read_csv(dest/'outer_predictions.csv', dtype={'event_id': str, 'case_id': str}, float_precision='round_trip')
                validate_events(f)
                if not f.set_index('event_id')[['case_id', 'label']].sort_index().equals(test.set_index('event_id')[['case_id', 'label']].sort_index()) or not f.fold.eq(fold).all() or not f.probability.between(0, 1).all():
                    raise ValueError('Invalid completed outer predictions')
                config = json.loads((dest/'fit_config.json').read_text())
                if config['condition'] != condition or config['seed'] != seed or config['fold'] != fold or config['class_weights'] != w:
                    raise ValueError('Invalid completed fit configuration')
                hashes.append(config['initialization_sha256']); frames[condition].append(f)
            if len(set(hashes)) != 1:
                raise ValueError('Paired model initializations differ')
        if args.action != 'audit':
            for condition in CONDITIONS:
                f = pd.concat(frames[condition], ignore_index=True)
                f.to_csv(out/f'seed{seed}'/f'{condition}_oof_predictions.csv', index=False)
                results.append(dict(seed=seed, condition=condition, **decision_metrics(f, f.probability >= .5)))
    pd.DataFrame(audit).to_csv(out/'partition_audit.csv', index=False)
    if args.action == 'audit':
        print(f'[AUDIT OK] events={len(events)} positives={int(events.label.sum())}; {len(args.seeds)*15} model fits')
        print(pd.DataFrame(audit).to_string(index=False)); return
    results = pd.DataFrame(results); results.to_csv(out/'per_seed_metrics.csv', index=False)
    numeric = ['auroc', 'pr_auc', 'tp', 'fp', 'fn', 'precision', 'sensitivity', 'specificity', 'f1', 'f2', 'mcc']
    summary = results.groupby('condition')[numeric].agg(['mean', 'std'])
    summary.to_csv(out/'seed_summary.csv'); print(summary.to_string())
    baseline = results[results.condition == 'balanced_sampler'].set_index('seed')
    differences = [dict(seed=r.seed, condition=r.condition,
                        **{x: getattr(r, x)-baseline.loc[r.seed, x] for x in numeric})
                   for r in results[results.condition != 'balanced_sampler'].itertuples()]
    pd.DataFrame(differences).to_csv(out/'paired_seed_deltas.csv', index=False)


if __name__ == '__main__':
    main()
