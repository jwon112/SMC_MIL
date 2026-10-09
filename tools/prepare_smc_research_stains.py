"""Prepare versioned gold-only research stain labels and frozen patient partitions.

No source labels, previous results, or clinical rejection grades are modified.
The JSON decision specification has review numbers, not patient identifiers.
Generated tables contain identifiers and must remain in an approved data location.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.threshold_control import assert_disjoint, validate_events


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def save_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
    if path.exists() and path.read_bytes().replace(b'\r\n', b'\n') != raw:
        raise ValueError(f'Existing output differs; use a new output root: {path}')
    path.write_bytes(raw)


def extend_partitions(reference, events, recovered, seeds):
    """Preserve all reference roles; assign each new patient one test and one stop fold."""
    result = {}; audits = []; sources = []
    base_ids = None
    for seed in seeds:
        old = []
        for fold in range(5):
            path = reference/f'seed{seed}'/f'fold_{fold}'/'partitions.csv'; sources.append(path)
            part = pd.read_csv(path, dtype={'event_id': str, 'case_id': str})
            validate_events(part)
            if set(part.role) != {'fit', 'stop', 'test'}:
                raise ValueError('Unexpected reference partition roles')
            assert_disjoint(*(part[part.role.eq(role)] for role in ('fit', 'stop', 'test')))
            ids = set(part.event_id)
            if base_ids is None: base_ids = ids
            if ids != base_ids: raise ValueError('Reference event membership changed across folds')
            expected = events[events.event_id.isin(ids)][['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True)
            pd.testing.assert_frame_equal(part[['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True), expected)
            old.append(part)
        if sorted(x for p in old for x in p.loc[p.role.eq('test'), 'event_id']) != sorted(base_ids):
            raise ValueError('Reference does not test every event exactly once')
        new = events[~events.event_id.isin(base_ids)].copy()
        if set(new.event_id) != set(recovered.event_id):
            raise ValueError('Unexpected recovered event membership')
        if set(new.case_id) & set(old[0].case_id):
            raise ValueError('Recovered patient already exists; inherit that patient roles before extending')
        rng = np.random.default_rng(seed+20261009)
        new_cases = sorted(new.case_id.unique()); rng.shuffle(new_cases)
        loads = np.array([p.loc[p.role.eq('test') & p.label.eq(1), 'case_id'].nunique() for p in old])
        assignments = {}
        for case in new_cases:
            test_fold = int(rng.choice(np.flatnonzero(loads == loads.min())))
            loads[test_fold] += 1
            stop_fold = int(rng.choice([f for f in range(5) if f != test_fold]))
            assignments[case] = test_fold, stop_fold
        tested = []
        for fold, part in enumerate(old):
            extra = new[['event_id', 'case_id', 'label']].copy()
            extra['role'] = extra.case_id.map(lambda c: 'test' if assignments[c][0] == fold else 'stop' if assignments[c][1] == fold else 'fit')
            combined = pd.concat([part, extra], ignore_index=True).sort_values('event_id').reset_index(drop=True)
            validate_events(combined)
            roles = [combined[combined.role.eq(role)] for role in ('fit', 'stop', 'test')]
            assert_disjoint(*roles)
            if any(frame.label.nunique() != 2 for frame in roles[:2]):
                raise ValueError('Fit or stop lacks a class')
            pd.testing.assert_frame_equal(combined[combined.event_id.isin(base_ids)].reset_index(drop=True), part.sort_values('event_id').reset_index(drop=True))
            tested.extend(roles[2].event_id)
            result[seed, fold] = combined
            for role, frame in zip(('fit', 'stop', 'test'), roles):
                audits.append(dict(seed=seed, fold=fold, role=role, events=len(frame), positive_events=int(frame.label.sum()),
                    patients=frame.case_id.nunique(), recovered_events=int(frame.event_id.isin(new.event_id).sum())))
        if sorted(tested) != sorted(events.event_id):
            raise ValueError('Recovered cohort does not have exactly-once outer evaluation')
    return result, pd.DataFrame(audits), sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label-csv', type=Path, default=PROJECT/'dataset_csv/smc_acr_binary_0r1r_vs_2r3r.csv')
    p.add_argument('--pending-csv', type=Path, default=Path('/home/jupyter/data/image_team/labels/derived/wsi_stain_provisional_20260930_upload/pending_review_25.csv'))
    p.add_argument('--spec', type=Path, default=Path(__file__).with_name('research_stain_gold7_20261009.json'))
    p.add_argument('--baseline-manifest-dir', type=Path, default=Path('/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930/acr_high'))
    p.add_argument('--reference-root', type=Path, default=PROJECT/'results/smc_event_imbalance_control_20261006_exclude25')
    p.add_argument('--stain-reference-csv', type=Path, help='Local fallback if the server baseline manifest is unavailable')
    p.add_argument('--previous-decisions-csv', type=Path, help='The previous 215 decisions, required for local fallback')
    p.add_argument('--output-root', type=Path, default=PROJECT/'results/smc_stain_research_20261009')
    a = p.parse_args(); out = a.output_root.resolve()
    sources = [a.label_csv, a.pending_csv, a.spec, a.reference_root/'protocol.json', Path(__file__), PROJECT/'utils/threshold_control.py']
    spec = json.loads(a.spec.read_text(encoding='utf-8'))
    decisions = pd.DataFrame(spec['decisions'])
    if len(decisions) != 7 or decisions.slide_number.duplicated().any() or not set(decisions.stain_group) <= {'HE','IHC','other'}:
        raise ValueError('Expected seven unique known-stain decisions')
    pending = pd.read_csv(a.pending_csv, dtype={'slide_id': str})
    labels = pd.read_csv(a.label_csv, dtype={'slide_id': str, 'case_id': str, 'event_id': str}).fillna('')
    if labels.slide_id.duplicated().any() or len(pending) != 25 or pending.slide_id.duplicated().any():
        raise ValueError('Invalid gold or pending membership')
    decisions = pending[['slide_id','slide_number']].merge(decisions, on='slide_number', validate='one_to_one')
    expected_decisions = set(pending.slide_id) & set(labels.slide_id)
    if len(decisions) != 7 or set(decisions.slide_id) != expected_decisions:
        raise ValueError('Decisions must cover exactly the seven held gold slides')
    decisions = decisions.merge(labels[['slide_id','event_id','case_id','label','label_text']], on='slide_id', validate='one_to_one')
    protocol = json.loads((a.reference_root/'protocol.json').read_text(encoding='utf-8'))
    inventory = set(protocol['feature_inventory'])
    base_labels = labels[labels.slide_id.isin(inventory)]
    if a.baseline_manifest_dir.is_dir():
        paths = [a.baseline_manifest_dir/'events.csv', a.baseline_manifest_dir/'event_slides.csv']; sources += paths
        base_events = pd.read_csv(paths[0], dtype={'event_id': str, 'case_id': str}).fillna('')
        base_slides = pd.read_csv(paths[1], dtype={'event_id': str, 'slide_id': str})
    else:
        if not a.stain_reference_csv or not a.previous_decisions_csv:
            raise FileNotFoundError('Baseline manifest missing; supply both local fallback inputs')
        sources += [a.stain_reference_csv, a.previous_decisions_csv]
        ref = pd.read_csv(a.stain_reference_csv, dtype=str).fillna('').set_index('slide_id')
        prev = pd.read_csv(a.previous_decisions_csv, dtype=str).fillna('').set_index('slide_id')
        if ref.index.duplicated().any() or prev.index.duplicated().any(): raise ValueError('Duplicate stain reference IDs')
        stain = ref.manual_label.replace('',pd.NA).fillna(ref.filename_label).replace('', 'unknown')
        base_slides = base_labels[['event_id','slide_id']].copy()
        base_slides['stain_group'] = base_slides.slide_id.map(prev.stain_group).fillna(base_slides.slide_id.map(stain)).replace({'special_other':'other'})
        base_slides = base_slides.sort_values(['event_id','slide_id']).reset_index(drop=True)
        base_events = base_labels[['event_id','case_id','label','label_text','biopsy_date']].drop_duplicates('event_id').sort_values('event_id').reset_index(drop=True)
    expected = {Path(k).name: v for k,v in protocol['input_sha256'].items() if k.endswith(('/events.csv','/event_slides.csv'))}
    for name, frame in [('events.csv',base_events),('event_slides.csv',base_slides)]:
        if hashlib.sha256(frame.to_csv(index=False,lineterminator='\n').encode()).hexdigest() != expected[name]:
            raise ValueError(f'Baseline does not match actual previous experiment: {name}')
    if set(base_slides.slide_id) != inventory or inventory & set(pending.slide_id):
        raise ValueError('Baseline inventory or pending exclusion differs')
    if set(labels.slide_id) != inventory | expected_decisions: raise ValueError('Unexpected source gold membership')
    events = labels[list(base_events.columns)].drop_duplicates('event_id').sort_values('event_id').reset_index(drop=True)
    validate_events(events)
    if (labels.groupby('event_id').label.nunique() != 1).any() or (labels.groupby('event_id').case_id.nunique() != 1).any():
        raise ValueError('Inconsistent clinical event mapping')
    recovered = events[~events.event_id.isin(base_events.event_id)]
    if len(base_events)!=575 or int(base_events.label.sum())!=14 or len(events)!=578 or int(events.label.sum())!=17 or len(recovered)!=3:
        raise ValueError('Unexpected baseline/recovered cohort counts')
    if not recovered.label.eq(1).all() or not decisions[decisions.event_id.isin(recovered.event_id)].stain_group.eq('HE').all():
        raise ValueError('Recovered event/stain mapping differs from the reviewed three HE slides')
    if not set(base_slides.stain_group) <= {'HE','IHC','other'}: raise ValueError('Unknown baseline stain')
    parts, audit, part_sources = extend_partitions(a.reference_root, events, recovered, protocol['seeds']); sources += part_sources
    for (seed,fold), frame in parts.items(): save_csv(out/'partitions'/f'seed{seed}'/f'fold_{fold}'/'partitions.csv', frame)
    save_csv(out/'partition_audit.csv', audit)
    for cohort, added in [('restored7',decisions),('restored3',decisions[decisions.event_id.isin(recovered.event_id)])]:
        slides = pd.concat([base_slides,added[['event_id','slide_id','stain_group']]],ignore_index=True).sort_values(['event_id','slide_id']).reset_index(drop=True)
        if slides.slide_id.duplicated().any() or set(slides.event_id)!=set(events.event_id): raise ValueError('Invalid restored slide membership')
        dest=out/cohort/'acr_high'
        save_csv(dest/'events.csv',events); save_csv(dest/'event_slides.csv',slides)
        for (seed,fold), part in parts.items():
            split = {name: part.loc[part.role.eq('test') if name=='val' else ~part.role.eq('test'),'event_id'].sort_values().reset_index(drop=True) for name in ['train','val']}
            save_csv(dest/'splits'/f'seed{seed}'/f'splits_{fold}.csv',pd.DataFrame(split))
    save_csv(out/'baseline575'/'acr_high'/'events.csv',base_events)
    save_csv(out/'baseline575'/'acr_high'/'event_slides.csv',base_slides)
    decisions['review_status']=spec['status']; decisions['stain_version']=spec['version']
    save_csv(out/'stain_decisions_gold7.csv',decisions.sort_values('slide_number'))
    full = labels.merge(pd.concat([base_slides,decisions[['event_id','slide_id','stain_group']]])[['slide_id','stain_group']],on='slide_id',validate='one_to_one')
    full['stain_version']=spec['version']; full['review_status']=np.where(full.slide_id.isin(decisions.slide_id),spec['status'],'inherited_previous_provisional')
    save_csv(out/'gold_stain_labels.csv',full.sort_values(['event_id','slide_id']))
    save_csv(out/'pending_non_gold_18.csv',pending[~pending.slide_id.isin(decisions.slide_id)])
    manifest=dict(version=spec['version'],status=spec['status'],seeds=protocol['seeds'],events=578,positive_events=17,
        patients=int(events.case_id.nunique()),positive_patients=int(events.loc[events.label.eq(1),'case_id'].nunique()),
        cohorts={'restored7':1269,'restored3':1265},primary_cohort='restored7',reference_condition='balanced_sampler',
        reference_results_root=str(a.reference_root.resolve()),baseline_roles_preserved=True,
        new_patient_rule='Seeded tie-breaking among least-positive-patient outer folds; one other fold for stop; remaining folds for fit',
        source_sha256_lf={str(x.resolve()):digest(x) for x in sources},
        output_sha256_lf={str(x.relative_to(out)).replace('\\','/'):digest(x) for x in out.rglob('*.csv')})
    target=out/'manifest.json'; encoded=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
    if target.exists() and target.read_text(encoding='utf-8')!=encoded: raise ValueError('Preparation inputs changed; use new output root')
    target.write_text(encoded,encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['version','events','positive_events','patients','positive_patients','cohorts','baseline_roles_preserved']},indent=2))


if __name__=='__main__':main()
