"""Audit matched stain-comparison cohorts and persistent positive misses, without training.

Use the server event manifest if available. Otherwise reconstruct the local copy
from reviewed stain records and accept it only if both protocol hashes match.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
COMBINATIONS = ['HE', 'HE+IHC', 'HE+other', 'HE+IHC+other', 'other']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT/'results/smc_event_imbalance_control_20261006_exclude25')
    parser.add_argument('--manifest-dir', type=Path)
    parser.add_argument('--label-csv', type=Path, default=PROJECT/'dataset_csv/smc_acr_binary_0r1r_vs_2r3r.csv')
    parser.add_argument('--stain-reference-csv', type=Path)
    parser.add_argument('--decisions-csv', type=Path)
    parser.add_argument('--pending-csv', type=Path)
    parser.add_argument('--prior-audit-csv', type=Path, default=PROJECT/'results/smc_dataset_audit/event_multiplicity_audit.csv')
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'results/smc_stain_feasibility_20261007')
    args = parser.parse_args()
    inputs = {}

    def read(path, **kwargs):
        path = Path(path)
        raw = path.read_bytes()
        inputs[str(path.resolve())] = sha(raw.replace(b'\r\n', b'\n'))
        return pd.read_csv(path, dtype={'slide_id': str, 'event_id': str, 'case_id': str}, **kwargs)

    protocol_path = args.results_root/'protocol.json'
    raw = protocol_path.read_bytes()
    inputs[str(protocol_path.resolve())] = sha(raw.replace(b'\r\n', b'\n'))
    protocol = json.loads(raw)
    expected = {Path(k).name: v for k, v in protocol['input_sha256'].items() if k.endswith(('/events.csv', '/event_slides.csv'))}
    inventory = set(protocol['feature_inventory'])
    labels = read(args.label_csv).fillna('')
    if labels.slide_id.duplicated().any():
        raise ValueError('Duplicate slide IDs in label source')
    labels = labels[labels.slide_id.isin(inventory)].copy()
    if set(labels.slide_id) != inventory:
        raise ValueError('Local labels do not cover the exact experiment feature inventory')
    if (labels.groupby('event_id').label.nunique() != 1).any() or (labels.groupby('event_id').case_id.nunique() != 1).any():
        raise ValueError('Inconsistent event labels or patient IDs')

    manifest_dir = args.manifest_dir or Path('/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930/acr_high')
    if manifest_dir.is_dir():
        events = read(manifest_dir/'events.csv').fillna('')
        slides = read(manifest_dir/'event_slides.csv')
        mode = 'server_manifest'
    elif args.manifest_dir:
        raise FileNotFoundError(manifest_dir)
    else:
        outer = PROJECT.parents[1]
        reference_path = args.stain_reference_csv or outer/'stainnet_분류결과/stainnet_predictions.csv'
        review_dir = outer/'data/image_team/labels/derived/wsi_stain_provisional_20260930_upload'
        decisions_path = args.decisions_csv or review_dir/'stain_decisions_provisional_215.csv'
        pending_path = args.pending_csv or review_dir/'pending_review_25.csv'
        reference = read(reference_path).fillna('')
        decisions = read(decisions_path).fillna('')
        pending = read(pending_path)
        if reference.slide_id.duplicated().any() or decisions.slide_id.duplicated().any():
            raise ValueError('Duplicate reviewed slide records')
        if set(pending.slide_id) & inventory:
            raise ValueError('Pending review slides entered the experiment')
        if set(decisions.slide_id) & set(pending.slide_id):
            raise ValueError('Decisions overlap pending review')
        source = reference.set_index('slide_id')
        baseline = source.manual_label.replace('', pd.NA).fillna(source.filename_label).replace('', 'unknown')
        slides = labels[['event_id', 'slide_id']].copy()
        slides['stain_group'] = slides.slide_id.map(decisions.set_index('slide_id').stain_group).fillna(slides.slide_id.map(baseline)).replace({'special_other': 'other'})
        slides = slides.sort_values(['event_id', 'slide_id']).reset_index(drop=True)
        events = labels[['event_id', 'case_id', 'label', 'label_text', 'biopsy_date']].drop_duplicates('event_id').sort_values('event_id').reset_index(drop=True)
        mode = 'local_review_reconstruction'

    # Compare canonical CSV bytes with the actual training inputs, not just row counts.
    hashes = {}
    for name, table in [('events.csv', events), ('event_slides.csv', slides)]:
        hashes[name] = sha(table.to_csv(index=False, lineterminator='\n').encode('utf-8'))
        if hashes[name] != expected[name]:
            raise ValueError(f'Experiment manifest hash differs: {name}')
    if slides.slide_id.duplicated().any() or set(slides.slide_id) != inventory:
        raise ValueError('Invalid experiment slide membership')
    if not set(slides.stain_group) <= {'HE', 'IHC', 'other'}:
        raise ValueError('Unknown stain group in experiment manifest')
    metadata = labels[['slide_id', 'source_dataset', 'slide_rel_path']]
    slides = slides.merge(metadata, on='slide_id', validate='one_to_one')
    counts = pd.crosstab(slides.event_id, slides.stain_group).reindex(columns=['HE', 'IHC', 'other'], fill_value=0)
    counts.columns = ['HE_slides', 'IHC_slides', 'other_slides']
    table = events.merge(counts, on='event_id', validate='one_to_one')
    table['stain_combination'] = table.apply(lambda r: '+'.join(s for s in ['HE', 'IHC', 'other'] if r[f'{s}_slides'] > 0), axis=1)
    table['slides'] = table[['HE_slides', 'IHC_slides', 'other_slides']].sum(axis=1)
    source = slides.groupby('event_id').source_dataset.agg(lambda x: '+'.join(sorted(set(x))))
    table = table.merge(source, on='event_id', validate='one_to_one')
    expected_mapping = table[['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True)

    predictions = []
    for seed in protocol['seeds']:
        reference_fold = None
        for condition in protocol['conditions']:
            frame = read(args.results_root/f'seed{seed}'/f'{condition}_oof_predictions.csv')
            mapping = frame[['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True)
            if frame.event_id.duplicated().any() or not mapping.equals(expected_mapping):
                raise ValueError('Prediction cohort does not match training manifest')
            if not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
                raise ValueError('Invalid prediction probabilities')
            fold = frame.set_index('event_id').fold.sort_index()
            if reference_fold is not None and not fold.equals(reference_fold):
                raise ValueError('Conditions use different outer folds')
            if frame.groupby('case_id').fold.nunique().max() != 1:
                raise ValueError('Patient spans outer folds')
            reference_fold = fold
            predictions.append(frame.assign(seed=seed, condition=condition, detected=frame.probability.ge(protocol['threshold']).astype(int)))
    predictions = pd.concat(predictions, ignore_index=True)
    positive = predictions[predictions.label.eq(1)]
    detections = positive.groupby(['event_id', 'condition']).detected.sum().unstack().add_suffix('_detections')
    table = table.merge(detections, on='event_id', how='left', validate='one_to_one')
    detection_cols = list(detections.columns)
    table['persistent_miss'] = table.label.eq(1) & table[detection_cols].eq(0).all(axis=1)
    miss_ids = set(table.loc[table.persistent_miss, 'event_id'])

    def summary(frame, name):
        pos = frame[frame.label.eq(1)]
        missed = frame[frame.persistent_miss]
        return dict(group=name, events=len(frame), patients=frame.case_id.nunique(), positive_events=len(pos),
                    positive_patients=pos.case_id.nunique(), slides=int(frame.slides.sum()),
                    positive_fraction=len(pos)/len(frame) if len(frame) else None,
                    persistent_miss_events=len(missed), persistent_miss_patients=missed.case_id.nunique())

    composition = pd.DataFrame([summary(table[table.stain_combination.eq(c)], c) for c in COMBINATIONS])
    sources = pd.DataFrame([summary(f, name) for name, f in table.groupby('source_dataset')])
    cohorts = {
        'all_current': table.event_id.isin(table.event_id),
        'HE_available': table.HE_slides.gt(0),
        'HE_and_IHC_available': table.HE_slides.gt(0) & table.IHC_slides.gt(0),
        'HE_and_other_available': table.HE_slides.gt(0) & table.other_slides.gt(0),
        'HE_IHC_other_available': table[['HE_slides', 'IHC_slides', 'other_slides']].gt(0).all(axis=1),
    }
    feasibility = pd.DataFrame([summary(table[mask], name) for name, mask in cohorts.items()])
    partitions = []
    for seed in protocol['seeds']:
        for fold in range(5):
            partition = read(args.results_root/f'seed{seed}'/f'fold_{fold}'/'partitions.csv')
            mapping = partition[['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True)
            if not mapping.equals(expected_mapping) or set(partition.role) != {'fit', 'stop', 'test'}:
                raise ValueError('Invalid saved fit/stop/test partition')
            if partition.groupby('case_id').role.nunique().max() != 1:
                raise ValueError('Patient spans fit/stop/test')
            for name, mask in cohorts.items():
                eligible = set(table.loc[mask, 'event_id'])
                for role in ['fit', 'stop', 'test']:
                    f = partition[partition.event_id.isin(eligible) & partition.role.eq(role)]
                    pos = f[f.label.eq(1)]
                    partitions.append(dict(cohort=name, seed=seed, fold=fold, role=role, events=len(f),
                                           patients=f.case_id.nunique(), positive_events=len(pos), positive_patients=pos.case_id.nunique()))
    partitions = pd.DataFrame(partitions)
    stability = partitions.groupby(['cohort', 'role']).agg(
        partitions=('fold', 'size'), min_positive_events=('positive_events', 'min'),
        max_positive_events=('positive_events', 'max'), min_positive_patients=('positive_patients', 'min'),
        max_positive_patients=('positive_patients', 'max'),
        zero_positive_partitions=('positive_events', lambda x: int(x.eq(0).sum())),
    ).reset_index()

    qc_cols = ['tissue_ratio_mean', 'sharpness_score_mean', 'tissue_luminance_mean_mean', 'tissue_saturation_mean_mean', 'grid_periodicity_score_mean']
    if args.prior_audit_csv.exists():
        prior = read(args.prior_audit_csv)
        prior = prior[prior.task.eq('acr_high')]
        if prior.event_id.duplicated().any():
            raise ValueError('Duplicate prior ACR audit events')
        prior = prior[['event_id', 'slides_per_event'] + qc_cols].rename(columns={'slides_per_event': 'prior_audit_slides'})
        table = table.merge(prior, on='event_id', how='left', validate='one_to_one')
        table['qc_same_slide_count'] = table.slides.eq(table.prior_audit_slides)
        # Historical event averages are usable only if no slides were removed.
        table.loc[~table.qc_same_slide_count, qc_cols] = np.nan

    missed = table[table.persistent_miss].copy()
    if len(missed) != 7 or missed.case_id.nunique() != 5:
        raise ValueError('Persistent-miss set differs from reviewed experiment')
    slide_review = slides[slides.event_id.isin(miss_ids)].merge(missed[['event_id', 'case_id', 'stain_combination']], on='event_id', validate='many_to_one')
    probability_summary = positive.groupby(['event_id', 'condition']).probability.agg(['min', 'median', 'max']).reset_index()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in {
        'stain_composition_summary.csv': composition, 'comparison_cohort_summary.csv': feasibility,
        'source_summary.csv': sources, 'event_inventory.csv': table,
        'persistent_miss_events.csv': missed, 'persistent_miss_slides.csv': slide_review,
        'positive_probability_summary.csv': probability_summary,
        'subset_partition_counts.csv': partitions, 'subset_partition_summary.csv': stability,
    }.items():
        frame.to_csv(out/filename, index=False, lineterminator='\n')
    manifest = dict(analysis_script_sha256_lf=sha(Path(__file__).read_bytes().replace(b'\r\n', b'\n')),
                    manifest_mode=mode, verified_training_manifest_sha256=hashes,
                    cohort=summary(table, 'all_current'), conditions=protocol['conditions'], seeds=protocol['seeds'],
                    threshold=protocol['threshold'], patient_counts_overlap_between_groups=True,
                    positive_patient_definition='At least one positive event within the reported group',
                    qc_scope='Historical event averages retained only when slide counts are unchanged; no new image review',
                    persistent_miss_scope='Below fixed threshold in every condition and seed; not evidence of label error or unlearnability',
                    input_sha256_lf=inputs)
    (out/'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(composition.to_string(index=False))
    print('\nComparison cohorts:\n'+feasibility.to_string(index=False))
    print('\nSource groups:\n'+sources.to_string(index=False))
    print('\nPositive patient ranges by partition:\n'+stability.to_string(index=False))
    print(f'\nVerified {len(events)} events / {len(slides)} slides; persistent misses {len(missed)} events / {missed.case_id.nunique()} patients -> {out}')


if __name__ == '__main__':
    main()
