"""Verify saved imbalance-control exposure, selection and predictions without feature bags."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.threshold_control import assert_disjoint, decision_metrics


def weighted_ap_draws(frame, weights, cases):
    """Vectorized AP under patient multiplicities; group score ties exactly."""
    order = np.argsort(-frame.probability.to_numpy(), kind='stable')
    p = frame.probability.to_numpy()[order]; y = frame.label.to_numpy()[order]
    lookup = {case: i for i, case in enumerate(cases)}
    ids = np.array([lookup[c] for c in frame.case_id.to_numpy()[order]])
    w = weights[:, ids]; positive = w*y
    total = positive.sum(axis=1)
    ends = np.r_[np.flatnonzero(p[:-1] != p[1:]), len(p)-1]
    cumulative = positive.cumsum(axis=1)[:, ends]
    denominator = w.cumsum(axis=1)[:, ends]
    precision = np.divide(cumulative, denominator, out=np.zeros_like(cumulative, dtype=float), where=denominator != 0)
    increments = np.diff(cumulative, axis=1, prepend=0)
    return np.divide((precision*increments).sum(axis=1), total,
                     out=np.full(len(weights), np.nan), where=total != 0)


def main():
    root = PROJECT/'results/smc_event_imbalance_control_20261006_exclude25'
    out = PROJECT/'results/smc_imbalance_review_20261007'; out.mkdir(parents=True, exist_ok=True)
    inputs = set()
    def read(path):
        inputs.add(path)
        return pd.read_csv(path, dtype={'event_id': str, 'case_id': str}, float_precision='round_trip')
    def read_json(path):
        inputs.add(path); return json.loads(path.read_text())
    protocol = read_json(root/'protocol.json'); conditions = protocol['conditions']; seeds = protocol['seeds']
    checks = {}
    for path, expected in protocol['input_sha256'].items():
        marker = '/SMC_MIL/'
        if marker not in path:
            continue
        local = PROJECT/path.split(marker, 1)[1]
        checks[str(local.relative_to(PROJECT))] = hashlib.sha256(local.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
    if len(checks) != 7 or not all(checks.values()):
        raise ValueError('Training code differs from protocol')
    results = []; exposure = []; all_predictions = []; base_table = None
    for seed in seeds:
        fold_frames = {c: [] for c in conditions}
        for fold in range(5):
            folder = root/f'seed{seed}'/f'fold_{fold}'
            partition = read(folder/'partitions.csv')
            groups = {role: partition[partition.role == role] for role in ['fit', 'stop', 'test']}
            if partition.event_id.duplicated().any() or len(partition) != 575:
                raise ValueError('Partition coverage mismatch')
            assert_disjoint(*groups.values())
            n = len(groups['fit']); counts = groups['fit'].label.value_counts()
            weights = [float(np.float32(n)/(2*np.float32(counts[k]))) for k in [0, 1]]
            hashes = []; natural_draws = []
            for condition in conditions:
                dest = folder/condition
                config = read_json(dest/'fit_config.json'); selection = read_json(dest/'selection.json')
                logs = read(dest/'epochs.csv'); draws = read(dest/'training_draws.csv')
                hashes.append(config['initialization_sha256'])
                if not np.allclose(config['class_weights'], weights) or config['fit_events'] != n or config['fit_positive'] != counts[1]:
                    raise ValueError('Fit-derived class weights mismatch')
                if config['condition'] != condition or config['fold'] != fold or config['seed'] != seed:
                    raise ValueError('Fit configuration identity mismatch')
                best_epoch = int(logs.loc[logs.stop_loss.idxmin(), 'epoch'])
                if best_epoch != selection['best_epoch'] or len(logs) != selection['epochs_run'] or not np.isclose(logs.stop_loss.min(), selection['best_stop_loss']):
                    raise ValueError('Stopping selection mismatch')
                labels = groups['fit'].set_index('event_id').label
                if not draws.label.eq(draws.event_id.map(labels)).all():
                    raise ValueError('Draw contains wrong label or non-fit event')
                if set(draws.epoch) != set(logs.epoch):
                    raise ValueError('Epoch coverage mismatch')
                for epoch, group in draws.groupby('epoch'):
                    actual = (len(group), int(group.label.sum()), group.event_id.nunique())
                    saved = logs[logs.epoch == epoch].iloc[0]
                    if actual != (saved.draws, saved.positive_draws, saved.unique_events) or len(group) != n or not np.array_equal(group.step.to_numpy(), np.arange(n)):
                        raise ValueError('Draw exposure summary mismatch')
                    if condition != 'balanced_sampler' and (actual[1] != counts[1] or actual[2] != n):
                        raise ValueError('Natural sampler must visit every fit event exactly once')
                if condition != 'balanced_sampler':
                    natural_draws.append(draws)
                prefix = draws[draws.epoch <= best_epoch]
                positives = prefix[prefix.label == 1]
                exposure.append(dict(seed=seed, fold=fold, condition=condition,
                                     fit_events=n, fit_positive=int(counts[1]),
                                     best_epoch=best_epoch, epochs_run=len(logs),
                                     mean_positive_draw_fraction=float((logs.positive_draws/logs.draws).mean()),
                                     mean_unique_event_fraction=float((logs.unique_events/logs.draws).mean()),
                                     best_prefix_positive_draws=len(positives),
                                     best_prefix_unique_positive_events=positives.event_id.nunique(),
                                     best_prefix_unique_negative_fraction=prefix.loc[prefix.label == 0, 'event_id'].nunique()/counts[0],
                                     mean_draws_per_positive_per_epoch=float(logs.positive_draws.mean()/counts[1]),
                                     positive_class_weight=weights[1]))
                f = read(dest/'outer_predictions.csv')
                if f.event_id.duplicated().any() or not f.probability.between(0, 1).all() or not f.fold.eq(fold).all():
                    raise ValueError('Invalid outer predictions')
                if not f.set_index('event_id')[['case_id', 'label']].sort_index().equals(groups['test'].set_index('event_id')[['case_id', 'label']].sort_index()):
                    raise ValueError('Outer patient-label mapping mismatch')
                fold_frames[condition].append(f)
            if len(set(hashes)) != 1:
                raise ValueError('Paired initial parameters differ')
            common_epoch = min(d.epoch.max() for d in natural_draws)
            if not natural_draws[0][natural_draws[0].epoch <= common_epoch].reset_index(drop=True).equals(natural_draws[1][natural_draws[1].epoch <= common_epoch].reset_index(drop=True)):
                raise ValueError('Natural sampler orders differ between losses')
        for condition in conditions:
            f = pd.concat(fold_frames[condition], ignore_index=True).sort_values('event_id').reset_index(drop=True)
            saved = read(root/f'seed{seed}'/f'{condition}_oof_predictions.csv').sort_values('event_id').reset_index(drop=True)
            if not f.equals(saved):
                raise ValueError('Combined predictions mismatch')
            mapping = f[['event_id', 'case_id', 'label']]
            if base_table is None:
                base_table = mapping
            if not mapping.equals(base_table) or len(f) != 575 or f.label.sum() != 14 or f.case_id.nunique() != 133 or (f.groupby('case_id').fold.nunique() != 1).any():
                raise ValueError('Unexpected cohort or crossing outer patient')
            results.append(dict(seed=seed, condition=condition, **decision_metrics(f, f.probability >= .5)))
            all_predictions.append(f.assign(seed=seed, condition=condition))
    metrics = pd.DataFrame(results)
    saved = read(root/'per_seed_metrics.csv').sort_values(['seed', 'condition']).reset_index(drop=True)
    verified = metrics.sort_values(['seed', 'condition']).reset_index(drop=True)[saved.columns]
    if not np.allclose(saved.select_dtypes('number'), verified.select_dtypes('number'), equal_nan=True):
        raise ValueError('Server metrics differ from recomputed values')
    metrics.to_csv(out/'verified_per_seed_metrics.csv', index=False)
    pd.DataFrame(exposure).to_csv(out/'exposure_and_selection.csv', index=False)
    combined = pd.concat(all_predictions, ignore_index=True)
    positive = combined[combined.label == 1].copy(); positive['detected'] = (positive.probability >= .5).astype(int)
    positive.groupby(['event_id', 'case_id', 'condition']).detected.sum().unstack('condition').reset_index().to_csv(out/'positive_event_review.csv', index=False)
    rows = []
    for omitted in [None]+seeds:
        for condition, f in metrics[metrics.seed != omitted].groupby('condition'):
            rows.append(dict(omitted_seed='none' if omitted is None else omitted, condition=condition,
                             mean_ap=f.pr_auc.mean(), median_ap=f.pr_auc.median(), mean_f1=f.f1.mean()))
    pd.DataFrame(rows).to_csv(out/'leave_one_seed_out_descriptive.csv', index=False)
    # Conditional patient bootstrap of seed-mean AP. All rules/seeds share each draw.
    cases = sorted(combined.case_id.unique()); rng = np.random.default_rng(20261007)
    weights = rng.multinomial(len(cases), np.full(len(cases), 1/len(cases)), size=2000).astype(float)
    ap = {c: [] for c in conditions}
    for condition in conditions:
        for seed in seeds:
            f = combined[(combined.condition == condition) & (combined.seed == seed)]
            result = weighted_ap_draws(f, weights, cases)
            unit = weighted_ap_draws(f, np.ones((1, len(cases))), cases)[0]
            if not np.isclose(unit, average_precision_score(f.label, f.probability)):
                raise ValueError('Vectorized AP differs from sklearn')
            lookup = {c: i for i, c in enumerate(cases)}
            for draw in range(3):
                sample_weight = np.array([weights[draw, lookup[c]] for c in f.case_id])
                if not np.isclose(result[draw], average_precision_score(f.label, f.probability, sample_weight=sample_weight), equal_nan=True):
                    raise ValueError('Weighted bootstrap AP differs from sklearn')
            ap[condition].append(result)
    intervals = []
    for condition in conditions[1:]:
        draws = (np.array(ap[condition])-np.array(ap['balanced_sampler'])).mean(axis=0)
        draws = draws[np.isfinite(draws)]
        low, high = np.quantile(draws, [.025, .975])
        delta = metrics[metrics.condition == condition].pr_auc.mean()-metrics[metrics.condition == 'balanced_sampler'].pr_auc.mean()
        intervals.append(dict(condition=condition, mean_ap_delta=delta,
                              conditional_ci_low=low, conditional_ci_high=high, valid_draws=len(draws)))
    pd.DataFrame(intervals).to_csv(out/'conditional_patient_bootstrap_ap.csv', index=False)
    manifest = dict(data_commit='efb95f6', verified_fits=75, verified_patient_partitions=25,
                    training_source_matches=checks, bootstrap_draws=2000, bootstrap_seed=20261007,
                    bootstrap_scope='patient clusters, fixed trained models and checkpoints; no retraining',
                    input_sha256_lf={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(inputs)},
                    script_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    (out/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, metric, title in zip(axes, ['pr_auc', 'tp', 'fp'], ['Average precision', 'True positives (14)', 'False positives (561)']):
        for seed in seeds:
            values = metrics[metrics.seed == seed].set_index('condition').loc[conditions, metric]
            ax.plot(range(3), values, marker='o', label=f'Seed {seed}', alpha=.85)
        ax.set_xticks(range(3), ['Balanced\nsampler', 'Natural\nCE', 'Natural\nweighted CE'])
        ax.set_title(title); ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Imbalance control: matched initialization and patient partitions')
    fig.tight_layout(); fig.savefig(out/'paired_seed_comparison.png', dpi=180); plt.close(fig)
    print('[VERIFIED] 75 fits: sources, partitions, draw labels/counts/order, initial hashes, checkpoint selections and metrics')
    print(metrics.to_string(index=False))
    print(pd.DataFrame(exposure).groupby('condition')[['best_epoch', 'mean_positive_draw_fraction', 'mean_unique_event_fraction', 'mean_draws_per_positive_per_epoch']].agg(['mean', 'median', 'min', 'max']).to_string())
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == '__main__':
    main()
