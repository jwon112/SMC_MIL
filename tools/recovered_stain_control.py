"""Agnostic counterpart of the frozen research-stain balanced-sampler fit.

Keep the original trainer unchanged: completed recovery protocols hash it.
The one-branch model retains the constant all-slides-present input used by the
earlier agnostic experiment. This input carries no stain identity.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def make_model(seed, fold, aware_config, device):
    from models.stain_aware_event_mil import StainAwareEventMIL
    from train_smc_event_stain_mil import seed_all
    from utils.imbalance_control import model_hash

    paired_seed = seed + 100003 * fold
    seed_all(paired_seed)
    aware = StainAwareEventMIL(1536, 128, .25, use_stain_branches=True,
                              include_presence_masks=True, presence_mask_values='observed')
    if model_hash(aware) != aware_config['initialization_sha256']:
        raise ValueError('Cannot reproduce saved aware initialization')
    shared_hash = model_hash(aware.patch_encoder)
    del aware
    seed_all(paired_seed)
    model = StainAwareEventMIL(1536, 128, .25, use_stain_branches=False,
                              include_presence_masks=True, presence_mask_values='observed')
    if model_hash(model.patch_encoder) != shared_hash:
        raise ValueError('Shared patch encoder initialization differs')
    return model.to(device), shared_hash


def assert_draws_match(actual, reference):
    """Different stopping epochs are allowed; every shared epoch must match."""
    columns = ['epoch', 'step', 'event_id', 'label']
    last = min(int(actual.epoch.max()), int(reference.epoch.max()))
    pd.testing.assert_frame_equal(
        actual.loc[actual.epoch <= last, columns].reset_index(drop=True),
        reference.loc[reference.epoch <= last, columns].reset_index(drop=True),
        check_dtype=False)
    return last


def fit_agnostic(fit, stop, test, slides, args, seed, fold, reference, dest):
    import torch
    from torch.utils.data import DataLoader
    from train_smc_event_stain_mil import evaluate
    from utils.event_mil import EventFeatureDataset, collate_event
    from utils.imbalance_control import balanced_weights, make_sampler, model_hash, training_loss

    aware_config = json.loads((reference / 'fit_config.json').read_text())
    device = torch.device(args.device)
    model, shared_hash = make_model(seed, fold, aware_config, device)
    weights = balanced_weights(fit.label)
    if aware_config['class_weights'] != weights.tolist():
        raise ValueError('Fit class weights differ from aware run')

    def evaluation_loader(frame):
        return DataLoader(EventFeatureDataset(frame, slides, args.feature_dir, 2048, False),
                          batch_size=1, shuffle=False, collate_fn=collate_event)

    generator = torch.Generator().manual_seed(seed + 100003 * fold + 17)
    train_loader = DataLoader(EventFeatureDataset(fit, slides, args.feature_dir, 2048, True),
                             batch_size=1, sampler=make_sampler(fit.label, 'balanced_sampler', generator),
                             collate_fn=collate_event)
    stop_loader = evaluation_loader(stop)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    config = dict(seed=seed, fold=fold, condition='balanced_sampler', mode='agnostic',
                  initialization_sha256=model_hash(model), shared_encoder_sha256=shared_hash,
                  use_stain_branches=False, include_presence_masks=True,
                  mask='constant all-slides-present; no stain identity',
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  class_weights=weights.tolist(), fit_events=len(fit), fit_positive=int(fit.label.sum()),
                  stopping_objective='unweighted CE', patch_cap=2048)
    (dest / 'fit_config.json').write_text(json.dumps(config, indent=2) + '\n')
    best = float('inf'); best_epoch = 0; stale = 0; logs = []; draws = []
    checkpoint = dest / 'checkpoint.pt'
    for epoch in range(args.max_epochs):
        model.train(); losses = []; selected = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits, _ = model([(stain, feature.to(device)) for stain, feature in batch['slides']])
            label = torch.tensor([batch['label']], device=device)
            loss = training_loss(logits, label, 'balanced_sampler', weights)
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
        print(f'[EPOCH] seed={seed} fold={fold} agnostic {row}', flush=True)
        if stop_loss < best:
            best = stop_loss; best_epoch = epoch+1; stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if epoch+1 >= args.min_epochs and stale >= args.patience:
            break
    pd.DataFrame(logs).to_csv(dest / 'epochs.csv', index=False)
    actual_draws = pd.DataFrame(draws)
    actual_draws.to_csv(dest / 'training_draws.csv', index=False)
    reference_draws = pd.read_csv(reference / 'training_draws.csv')
    shared_epochs = assert_draws_match(actual_draws, reference_draws)
    (dest / 'selection.json').write_text(json.dumps(dict(best_epoch=best_epoch,
        best_stop_loss=best, epochs_run=len(logs), identical_event_draw_epochs=shared_epochs), indent=2) + '\n')
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    _, predictions = evaluate(model, evaluation_loader(test), device)
    predictions['fold'] = fold
    if not predictions.probability.between(0, 1).all():
        raise ValueError('Invalid outer predictions')
    # The final prediction file marks completion only after exposure validation.
    predictions.to_csv(dest / 'outer_predictions.csv', index=False)
    print(f'[OK] seed={seed} fold={fold} agnostic', flush=True)
