"""Audit completed nested threshold runs and compare patient calls with prior controls."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.threshold_control import assert_disjoint, decision_metrics


def main():
    root = PROJECT/'results/smc_event_threshold_control_20261005_exclude25'
    out = PROJECT/'results/smc_threshold_review_20261006'
    out.mkdir(parents=True, exist_ok=True)
    inputs = set()
    def read(path):
        inputs.add(path)
        return pd.read_csv(path, dtype={'event_id': str, 'case_id': str}, float_precision='round_trip')
    def read_json(path):
        inputs.add(path)
        return json.loads(path.read_text(encoding='utf-8'))
    protocol = read_json(root/'protocol.json')
    # Git may convert line endings locally. Match the Linux training sources using LF bytes.
    code_checks = {}
    for server_path, expected in protocol['input_sha256'].items():
        for prefix in ['tools/', 'utils/', 'models/']:
            if '/'+prefix in server_path:
                local = PROJECT/prefix/server_path.split('/'+prefix, 1)[1]
                break
        else:
            if not server_path.endswith('/train_smc_event_stain_mil.py'):
                continue
            local = PROJECT/'train_smc_event_stain_mil.py'
        raw = local.read_bytes().replace(b'\r\n', b'\n')
        code_checks[str(local.relative_to(PROJECT))] = hashlib.sha256(raw).hexdigest() == expected
    if not all(code_checks.values()):
        raise ValueError('Training source differs from server protocol')
    seed_metrics = []; fold_rows = []; patient_calls = []; transitions = []; previous = []
    all_seed_frames = []; inner_rows = []
    for seed in protocol['seeds']:
        frames = []
        for fold in range(5):
            dest = root/f'seed{seed}'/f'outer_{fold}'
            outer = read(dest/'outer_predictions.csv')
            inner = read(dest/'inner_oof_predictions.csv')
            part = read(dest/'partitions.csv'); cut = read_json(dest/'thresholds.json')
            if outer.event_id.duplicated().any() or inner.event_id.duplicated().any():
                raise ValueError('Duplicate predictions')
            if set(outer.event_id) & set(inner.event_id):
                raise ValueError('Outer/inner overlap')
            assert_disjoint(outer, inner)
            epochs = []
            for k in range(3):
                groups = {role: part[(part.inner == k) & (part.role == role)] for role in ['fit', 'stop', 'calibration', 'outer_test']}
                assert_disjoint(*groups.values())
                train = pd.concat([groups[x] for x in ['fit', 'stop', 'calibration']])
                for predicted, actual in [(train, inner), (groups['outer_test'], outer),
                                          (groups['calibration'], inner[inner.inner == k])]:
                    a = predicted.set_index('event_id')[['case_id', 'label']].sort_index()
                    b = actual.set_index('event_id')[['case_id', 'label']].sort_index()
                    if not a.equals(b):
                        raise ValueError('Partition/prediction patient-label mismatch')
                record = read_json(dest/f'inner_{k}'/'fit_record.json')
                log = read(dest/f'inner_{k}'/'epochs.csv')
                epoch = int(log.loc[log.stop_loss.idxmin(), 'epoch'])
                if record['selected_epoch'] != epoch or record['seed'] != seed+100003*fold+1009*(k+1):
                    raise ValueError('Checkpoint epoch/seed mismatch')
                epochs.append(epoch)
            if epochs != cut['inner_best_epochs'] or int(np.median(epochs)) != cut['refit_epochs']:
                raise ValueError('Refit epoch selection mismatch')
            refit = read(dest/'refit'/'epochs.csv')
            record = read_json(dest/'refit'/'fit_record.json')
            if len(refit) != cut['refit_epochs'] or record['selected_epoch'] != len(refit) or record['seed'] != seed+100003*fold:
                raise ValueError('Actual refit duration/seed mismatch')
            row = dict(seed=seed, fold=fold, refit_epochs=len(refit), outer_events=len(outer),
                       outer_positive=int(outer.label.sum()), inner_positive=int(inner.label.sum()))
            for rule, threshold in cut['thresholds'].items():
                if not outer[f'threshold_{rule}'].eq(threshold).all() or not outer[f'prediction_{rule}'].eq((outer.probability >= threshold).astype(int)).all():
                    raise ValueError('Saved calls do not match threshold')
                if rule != 'fixed_0p5':
                    beta = 1 if rule == 'inner_f1' else 2
                    values = np.r_[np.unique(inner.probability), np.nextafter(inner.probability.max(), np.inf)]
                    calls = inner.probability.to_numpy()[None, :] >= values[:, None]
                    positive = inner.label.to_numpy().astype(bool)
                    tp = (calls & positive).sum(axis=1); fp = (calls & ~positive).sum(axis=1)
                    fn = positive.sum()-tp
                    scores = (1+beta**2)*tp/((1+beta**2)*tp + beta**2*fn + fp)
                    selected = values[np.flatnonzero(scores == scores.max())[-1]]
                    if threshold != selected:
                        raise ValueError('Threshold does not maximize the prespecified inner score')
                    if not np.isclose(fbeta_score(inner.label, inner.probability >= threshold, beta=beta), scores.max()):
                        raise ValueError('Independent sklearn F-beta score differs')
                    inner_rows.append(dict(seed=seed, fold=fold, rule=rule, threshold=threshold,
                                           **decision_metrics(inner, inner.probability >= threshold)))
                row[f'threshold_{rule}'] = threshold
                result = decision_metrics(outer, outer[f'prediction_{rule}'])
                for metric in ['tp', 'fp', 'fn', 'auroc', 'pr_auc', 'f1', 'f2']:
                    row[f'{rule}_{metric}'] = result[metric]
            fold_rows.append(row); frames.append(outer)
        frame = pd.concat(frames, ignore_index=True)
        if frame.event_id.duplicated().any() or frame.case_id.nunique() != 133 or len(frame) != 575 or frame.label.sum() != 14:
            raise ValueError('Unexpected cohort')
        if (frame.groupby('case_id').fold.nunique() != 1).any():
            raise ValueError('Outer patient crosses folds')
        saved = read(root/f'seed{seed}'/'oof_predictions.csv')
        if not frame.set_index('event_id').sort_index().equals(saved.set_index('event_id').sort_index()):
            raise ValueError('Combined OOF differs from fold files')
        frame['seed'] = seed; all_seed_frames.append(frame)
        for rule in ['fixed_0p5', 'inner_f1', 'inner_f2']:
            seed_metrics.append(dict(seed=seed, rule=rule, **decision_metrics(frame, frame[f'prediction_{rule}'])))
            for label in [0, 1]:
                sub = frame[frame.label == label]
                old = sub.prediction_fixed_0p5; new = sub[f'prediction_{rule}']
                transitions.append(dict(seed=seed, rule=rule, label=label,
                                        negative_to_positive=int(((old == 0) & (new == 1)).sum()),
                                        positive_to_negative=int(((old == 1) & (new == 0)).sum())))
            for patient, group in frame.groupby('case_id'):
                patient_calls.append(dict(seed=seed, rule=rule, case_id=patient,
                                          label=int(group.label.max()), prediction=int(group[f'prediction_{rule}'].max()),
                                          events=len(group)))
        for family, relative in [('presence_aware', f'results/smc_event_presence_control_20261004_exclude25/acr_high_l0_0p25mpp_40x_seed{seed}'),
                                 ('patch2048', f'results/smc_event_patch_control_20261004_exclude25/cap2048_seed{seed}')]:
            old = read(PROJECT/relative/'oof_predictions.csv')
            a = old.set_index('event_id')[['case_id', 'label', 'fold']].sort_index()
            b = frame.set_index('event_id')[['case_id', 'label', 'fold']].sort_index()
            if not a.equals(b):
                raise ValueError('Prior/new outer partition mismatch')
            old_result = decision_metrics(old, old.probability >= .5)
            current = decision_metrics(frame, frame.prediction_fixed_0p5)
            previous.append(dict(seed=seed, prior_family=family,
                                 **{f'prior_{k}': v for k, v in old_result.items()},
                                 **{f'current_{k}': v for k, v in current.items()}))
    metrics = pd.DataFrame(seed_metrics)
    original = read(root/'per_seed_metrics.csv').sort_values(['seed', 'rule']).reset_index(drop=True)
    verified = metrics.sort_values(['seed', 'rule']).reset_index(drop=True)
    if not np.allclose(original.select_dtypes('number'), verified[original.columns].select_dtypes('number'), equal_nan=True):
        raise ValueError('Recomputed metrics differ from server summary')
    metrics.to_csv(out/'verified_per_seed_metrics.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_threshold_diagnostics.csv', index=False)
    pd.DataFrame(inner_rows).to_csv(out/'inner_calibration_metrics.csv', index=False)
    pd.DataFrame(transitions).to_csv(out/'call_transitions.csv', index=False)
    patients = pd.DataFrame(patient_calls); patients.to_csv(out/'patient_any_positive_calls.csv', index=False)
    patient_results = []
    for (seed, rule), group in patients.groupby(['seed', 'rule']):
        pred = group.prediction.astype(bool); y = group.label
        patient_results.append(dict(seed=seed, rule=rule, patients=len(group), positive_patients=int(y.sum()),
                                    tp=int(((y == 1) & pred).sum()), fp=int(((y == 0) & pred).sum()),
                                    fn=int(((y == 1) & ~pred).sum())))
    pd.DataFrame(patient_results).to_csv(out/'patient_call_counts.csv', index=False)
    pd.DataFrame(previous).to_csv(out/'prior_protocol_comparison.csv', index=False)
    combined = pd.concat(all_seed_frames)
    # Ordinary patient-cluster resampling, shared across every seed and rule.
    # This conditions on already fitted models/cutoffs; there is no retraining.
    case_ids = sorted(combined.case_id.unique()); seeds = protocol['seeds']
    rng = np.random.default_rng(20261006)
    weights = rng.multinomial(len(case_ids), np.full(len(case_ids), 1/len(case_ids)), size=2000)
    totals = {}
    for rule in ['fixed_0p5', 'inner_f1', 'inner_f2']:
        counts = []
        for seed in seeds:
            f = combined[combined.seed == seed].copy()
            pred = f[f'prediction_{rule}'].astype(bool); y = f.label
            for name, value in [('tp', (y == 1) & pred), ('fp', (y == 0) & pred),
                                ('tn', (y == 0) & ~pred), ('fn', (y == 1) & ~pred)]:
                f[name] = value.astype(int)
            matrix = f.groupby('case_id')[['tp', 'fp', 'tn', 'fn']].sum().reindex(case_ids).to_numpy()
            expected = metrics[(metrics.seed == seed) & (metrics.rule == rule)][['tp', 'fp', 'tn', 'fn']].iloc[0].to_numpy()
            if not np.array_equal(matrix.sum(axis=0), expected):
                raise ValueError('Bootstrap patient counts do not reproduce seed metrics')
            counts.append(matrix)
        totals[rule] = np.einsum('bp,spc->bsc', weights, np.array(counts))
    intervals = []
    def values(counts):
        tp, fp, tn, fn = np.moveaxis(counts, -1, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return dict(f1=2*tp/(2*tp+fp+fn), f2=5*tp/(5*tp+4*fn+fp),
                        sensitivity=tp/(tp+fn), specificity=tn/(tn+fp), tp=tp, fp=fp)
    base = values(totals['fixed_0p5'])
    for rule in ['inner_f1', 'inner_f2']:
        new = values(totals[rule])
        for metric in base:
            draws = (new[metric]-base[metric]).mean(axis=1)
            draws = draws[np.isfinite(draws)]
            point = metrics[metrics.rule == rule][metric].mean()-metrics[metrics.rule == 'fixed_0p5'][metric].mean()
            lo, hi = np.quantile(draws, [.025, .975])
            intervals.append(dict(rule=rule, metric=metric, seed_mean_delta=point,
                                  conditional_ci_low=lo, conditional_ci_high=hi, valid_draws=len(draws)))
    pd.DataFrame(intervals).to_csv(out/'conditional_patient_bootstrap.csv', index=False)
    rows = []
    for event, group in combined[combined.label == 1].groupby('event_id'):
        row = dict(event_id=event, case_id=group.case_id.iloc[0])
        for rule in ['fixed_0p5', 'inner_f1', 'inner_f2']:
            row[f'{rule}_detected_seeds'] = int(group[f'prediction_{rule}'].sum())
        row['f2_rescue_seeds'] = int(((group.prediction_fixed_0p5 == 0) & (group.prediction_inner_f2 == 1)).sum())
        row['f2_lost_seeds'] = int(((group.prediction_fixed_0p5 == 1) & (group.prediction_inner_f2 == 0)).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'positive_event_review.csv', index=False)
    negative = combined[combined.label == 0]
    fp = negative.groupby(['event_id', 'case_id'])[['prediction_fixed_0p5', 'prediction_inner_f2']].sum().reset_index()
    fp = fp.rename(columns={'prediction_fixed_0p5': 'fixed_fp_seeds', 'prediction_inner_f2': 'f2_fp_seeds'})
    fp[fp.f2_fp_seeds > 0].sort_values(['f2_fp_seeds', 'fixed_fp_seeds'], ascending=False).to_csv(out/'negative_event_review.csv', index=False)
    manifest = dict(data_commit='1c85034', verified_outer_folds=25, verified_inner_folds=75,
                    cohort_events=575, positive_events=14, patients=133, positive_patients=11,
                    training_source_matches=code_checks,
                    bootstrap=dict(draws=2000, seed=20261006, unit='patient', scope='fixed trained models and thresholds; shared draws across seeds'),
                    input_sha256_lf={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(inputs)},
                    script_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    (out/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('[VERIFIED] 25 outer folds, 75 inner folds; partition, epoch, threshold, call and metric checks passed')
    print(metrics.to_string(index=False))
    print(pd.DataFrame(patient_results).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    # Publication-style standalone figure: no uncertainty implied by connecting seed points.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    rules = ['fixed_0p5', 'inner_f1', 'inner_f2']
    for ax, metric, title in zip(axes, ['tp', 'fp', 'f2'], ['True positives (14 events)', 'False positives (561 events)', 'F2 score']):
        for seed in seeds:
            series = metrics[metrics.seed == seed].set_index('rule').loc[rules, metric]
            ax.plot(range(3), series, marker='o', alpha=.8, label=f'Seed {seed}')
        ax.set_xticks(range(3), ['Fixed 0.5', 'Inner F1', 'Inner F2']); ax.set_title(title)
        ax.grid(axis='y', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    axes[2].legend(fontsize=8, loc='best')
    fig.suptitle('Nested threshold control: identical predictions, different decisions')
    fig.tight_layout(); fig.savefig(out/'paired_seed_tradeoffs.png', dpi=180); plt.close(fig)
    print(pd.DataFrame(intervals).to_string(index=False))


if __name__ == '__main__':
    main()
