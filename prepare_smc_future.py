#!/usr/bin/env python3
"""Build WSI future labels without changing the gold classification cohort.

next_biopsy and explicitly requested next_within_* need only the two workbooks.
within_* means ANY recorded rejection in (anchor, anchor+horizon]; negatives
require an explicitly curated outcome-ascertainment interval, NOT last biopsy.
All outputs are new, pseudonymous research data; no original workbook is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

OUTCOMES = ('acr_high', 'amr', 'significant')
ENDPOINTS = ('next_biopsy', 'within_30d', 'within_60d', 'within_90d',
             'next_within_30d', 'next_within_60d', 'next_within_90d')
ACR = {'0R': 0, '1R': 0, '2R': 1, '3R': 1}
AMR = {'PAMR0': 0, 'PAMR1': 1, 'PAMR1(I+)': 1, 'PAMR1(H+)': 1,
       'PAMR2': 1, 'PAMR3': 1}


def text(value):
    return '' if value is None or pd.isna(value) else str(value).strip()


def hospital_id(value):
    value = re.sub(r'\.0$', '', text(value))
    match = re.fullmatch(r'(\d+)(?:-\d+)?', value)
    return str(int(match[1])) if match else ''


def day(value):
    if value is None or pd.isna(value) or text(value) == '':
        return None
    if isinstance(value, (datetime, date)):
        return value.date() if isinstance(value, datetime) else value
    if isinstance(value, (int, float)) and 20000 <= value <= 65000:
        return date(1899, 12, 30) + timedelta(days=int(value))
    parsed = pd.to_datetime(text(value), errors='coerce')
    return None if pd.isna(parsed) else parsed.date()


def norm_event(value):
    return re.sub(r'[^A-Z0-9]', '', text(value).upper())


def token(prefix, value):
    return prefix + hashlib.sha256(str(value).encode()).hexdigest()[:20]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grade(value):
    return re.sub(r'\s+', '', text(value)).upper()


def labels(acr, amr):
    a, m = ACR.get(grade(acr)), AMR.get(grade(amr))
    combined = 1 if a == 1 or m == 1 else (0 if a == m == 0 else None)
    return {'acr_high': a, 'amr': m, 'significant': combined}


def build_timeline(ehr):
    required = {'환자번호', 'SMC_id', 'TPL NO', 'TPL-date', '생검일자', 'ACR 등급', 'AMR 등급'}
    if not required <= set(ehr.columns):
        raise ValueError(f'EHR missing columns: {sorted(required - set(ehr.columns))}')
    counts = Counter()
    grouped = defaultdict(list)
    identities = defaultdict(set)
    hospital_ids_by_person = defaultdict(set)
    records = ehr.to_dict('records')
    uncertain_people, uncertain_hospital_ids = set(), set()
    for row in records:
        pid, hid = text(row['환자번호']), hospital_id(row['SMC_id'])
        if pid and hid:
            identities[hid].add(pid)
            hospital_ids_by_person[pid].add(hid)
        if not pid or day(row['TPL-date']) is None or day(row['생검일자']) is None or not text(row['TPL NO']):
            if pid:
                uncertain_people.add(pid)
            if hid:
                uncertain_hospital_ids.add(hid)
    for row in records:
        pid, hid = text(row['환자번호']), hospital_id(row['SMC_id'])
        tpl, when = day(row['TPL-date']), day(row['생검일자'])
        # A missing hospital ID on a future biopsy must not delete that biopsy
        # and incorrectly connect its predecessor to a later observed result.
        if not hid and pid and len(hospital_ids_by_person[pid]) == 1:
            hid = next(iter(hospital_ids_by_person[pid]))
            counts['ehr_hospital_id_resolved_within_same_pseudonym'] += 1
        if not pid or not hid or tpl is None or when is None or not text(row['TPL NO']):
            counts['ehr_missing_identity_or_date'] += 1
            continue
        if pid in uncertain_people or hid in uncertain_hospital_ids:
            counts['ehr_quarantined_uncertain_timeline'] += 1
            continue
        if when < tpl:
            counts['ehr_pretransplant'] += 1
            continue
        grouped[(pid, hid, text(row['TPL NO']), tpl, when)].append(row)
    # Never silently merge different EHR persons under a stripped hospital ID.
    ambiguous_ids = {hid for hid, pids in identities.items() if len(pids) > 1}
    ambiguous_people = {pid for pid, hids in hospital_ids_by_person.items() if len(hids) > 1}
    episodes = defaultdict(list)
    for (pid, hid, number, tpl, when), rows in grouped.items():
        if hid in ambiguous_ids or pid in ambiguous_people:
            counts['ehr_ambiguous_hospital_id_date_groups'] += 1
            continue
        counts['ehr_duplicate_rows_collapsed'] += len(rows) - 1
        acrs = {grade(r['ACR 등급']) for r in rows}
        amrs = {grade(r['AMR 등급']) for r in rows}
        lab = labels(next(iter(acrs)) if len(acrs) == 1 else '',
                     next(iter(amrs)) if len(amrs) == 1 else '')
        if len(acrs) > 1 or len(amrs) > 1:
            counts['ehr_same_day_grade_conflict'] += 1
        episode = (pid, number, tpl)
        episodes[episode].append(dict(hid=hid, date=when, tpl=tpl, labels=lab,
                                     acr=acrs, amr=amrs, episode=episode,
                                     case_id=token('future_patient_', pid)))
    by_person = defaultdict(set)
    for pid, _, tpl in episodes:
        by_person[pid].add(tpl)
    lookup = defaultdict(list)
    for episode, events in episodes.items():
        next_tpl = min((d for d in by_person[episode[0]] if d > episode[2]), default=None)
        events.sort(key=lambda e: e['date'])
        for e in events:
            e['next_transplant_date'] = next_tpl
            if next_tpl and e['date'] >= next_tpl:
                counts['ehr_old_episode_after_retransplant'] += 1
                continue
            lookup[(e['hid'], e['date'])].append(e)
        episodes[episode] = [e for e in events if not next_tpl or e['date'] < next_tpl]
    counts['ehr_rows'] = len(ehr)
    return episodes, lookup, counts


def link_wsi(wsi, lookup):
    required = {'ID', '생검일자', 'ACR 등급', 'AMR 등급', '병리ID'}
    if not required <= set(wsi.columns):
        raise ValueError(f'WSI workbook missing columns: {sorted(required - set(wsi.columns))}')
    counts, linked, rejected = Counter(), {}, {}
    event_counts = Counter(norm_event(v) for v in wsi['병리ID'])
    for row in wsi.to_dict('records'):
        event = norm_event(row['병리ID'])
        when, hid = day(row['생검일자']), hospital_id(row['ID'])
        candidates = lookup.get((hid, when), [])
        reason = None
        if not event or event_counts[event] != 1:
            reason = 'wsi_duplicate_or_missing_pathology_id'
        elif len(candidates) != 1:
            reason = 'wsi_no_exact_ehr_match' if not candidates else 'wsi_ambiguous_episode'
        else:
            anchor = candidates[0]
            for source, key in [('ACR 등급', 'acr'), ('AMR 등급', 'amr')]:
                value = grade(row[source])
                if value and anchor[key] != {value}:
                    reason = 'wsi_ehr_current_grade_disagreement'
        if reason:
            counts[reason] += 1
            rejected[event] = reason
        else:
            linked[event] = candidates[0]
    counts['wsi_rows'] = len(wsi)
    counts['wsi_exact_linked_rows'] = len(linked)
    return linked, rejected, counts


def future_target(anchor, events, outcome, endpoint, coverage=None):
    future = [e for e in events if e['date'] > anchor['date']]
    if endpoint == 'next_biopsy' or endpoint.startswith('next_within_'):
        if not future:
            return None, 'no_next_observed_biopsy', None
        nxt = future[0]  # Do NOT skip the immediate next biopsy if grade is missing.
        if endpoint.startswith('next_within_'):
            horizon = int(endpoint.removeprefix('next_within_').removesuffix('d'))
            if (nxt['date'] - anchor['date']).days > horizon:
                return None, 'next_biopsy_outside_window', nxt['date']
        value = nxt['labels'][outcome]
        return value, 'observed_next' if value is not None else 'next_grade_unknown', nxt['date']
    horizon = int(endpoint.removeprefix('within_').removesuffix('d'))
    end = anchor['date'] + timedelta(days=horizon)
    observed = [e for e in future if e['date'] <= end]
    positive = next((e for e in observed if e['labels'][outcome] == 1), None)
    if positive:
        return 1, 'observed_positive_in_window', positive['date']
    if anchor.get('next_transplant_date') and anchor['next_transplant_date'] <= end:
        return None, 'retransplant_before_horizon', None
    if any(e['labels'][outcome] is None for e in observed):
        return None, 'unknown_grade_in_window', None
    if coverage and coverage['start'] <= anchor['date'] and coverage['end'] >= end:
        return 0, 'curated_complete_capture_no_positive', end
    return None, 'outcome_ascertainment_not_confirmed', None


def read_coverage(path):
    if path is None:
        return {}
    frame = pd.read_csv(path, dtype=str).fillna('')
    required = {'case_id', 'transplant_date', 'ascertainment_start_date',
                'ascertainment_end_date', 'complete_rejection_capture', 'review_source'}
    if not required <= set(frame.columns):
        raise ValueError(f'Coverage missing columns: {sorted(required-set(frame.columns))}')
    result = {}
    for r in frame.to_dict('records'):
        if r['complete_rejection_capture'] != '1':
            continue
        start, end, tpl = (day(r[k]) for k in ('ascertainment_start_date', 'ascertainment_end_date', 'transplant_date'))
        if not r['case_id'] or not r['review_source'] or None in (start, end, tpl) or start > end:
            raise ValueError('Invalid confirmed ascertainment interval')
        key = (r['case_id'], tpl)
        if key in result:
            raise ValueError('Duplicate coverage patient/episode')
        result[key] = {'start': start, 'end': end}
    return result


def make_splits(data, folds, seed):
    from sklearn.model_selection import StratifiedKFold
    patients = data.groupby('case_id', sort=True)['label'].max()
    if set(patients.unique()) != {0, 1} or patients.value_counts().min() < folds:
        raise ValueError(f'Need >= {folds} positive and {folds} negative-only patients')
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    output = []
    for tr, va in splitter.split(patients.index, patients.values):
        train = data[data.case_id.isin(patients.index[tr])]
        val = data[data.case_id.isin(patients.index[va])]
        assert not set(train.case_id) & set(val.case_id)
        if train.label.nunique() != 2 or val.label.nunique() != 2:
            raise ValueError('A fold is single-class; reduce folds or review cohort')
        output.append((train, val))
    return output


def write_csv(path, rows, columns=None):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def prepare(args):
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f'Preserve existing cohort; choose a new --output-dir: {out}')
    ehr = pd.read_excel(args.ehr_xlsx, sheet_name='biopsy_features', dtype=object)
    wsi = pd.read_excel(args.wsi_xlsx, sheet_name='Sheet1', dtype=object)
    episodes, lookup, counts = build_timeline(ehr)
    linked, rejected, link_counts = link_wsi(wsi, lookup)
    counts.update(link_counts)
    seeds = [int(s) for s in args.seeds.split(',')]
    if len(set(seeds)) != len(seeds) or min(seeds) < 0 or args.folds < 2:
        raise ValueError('Invalid seeds/fold count')
    endpoints = args.endpoints.split(',')
    if not set(endpoints) <= set(ENDPOINTS):
        raise ValueError('Unknown endpoint')
    coverage = read_coverage(args.coverage_csv)
    sources = [args.ehr_xlsx, args.wsi_xlsx, Path(__file__)]
    if args.coverage_csv:
        sources.append(args.coverage_csv)
    gold = None
    if not args.audit_only:
        sources.append(args.gold_csv)
        gold = pd.read_csv(args.gold_csv, dtype=str).fillna('')
        required = {'slide_id', 'case_id', 'event_id', 'source_dataset', 'biopsy_date'}
        if not required <= set(gold.columns) or gold.slide_id.duplicated().any():
            raise ValueError('Gold CSV requires unique slide_id, case_id, event_id, source_dataset, biopsy_date')
        if not (args.baseline_dir / 'pt_files').is_dir():
            raise FileNotFoundError(args.baseline_dir / 'pt_files')
    out.mkdir(parents=True)
    audit, anchors = [], []
    if gold is None:
        anchors = [(event, '', anchor, {}) for event, anchor in linked.items()]
    else:
        for row in gold.to_dict('records'):
            sid, event = row['slide_id'], norm_event(row['event_id'])
            if Path(sid).name != sid or not sid:
                raise ValueError('Unsafe slide_id')
            anchor = linked.get(event)
            reason = rejected.get(event, 'no_wsi_workbook_match') if anchor is None else ''
            if anchor is not None and day(row['biopsy_date']) != anchor['date']:
                reason = 'gold_anchor_date_mismatch'
            feature = args.baseline_dir / 'pt_files' / f'{sid}.pt'
            if not feature.is_file() or feature.stat().st_size == 0:
                reason = 'missing_original_feature'
            if reason:
                counts[reason] += 1
                audit.append({'slide_id': sid, 'event_id': event, 'status': reason})
                continue
            anchors.append((event, sid, anchor, row))
    summary = {'schema': 1, 'audit_only': args.audit_only, 'source_sha256':
               {str(p.resolve()): sha(p) for p in sources}, 'source_counts': dict(counts),
               'baseline_dir': str(args.baseline_dir.resolve()), 'seeds': seeds, 'folds': args.folds,
               'definitions': {'acr_high': 'ACR 2R/3R', 'amr': 'pAMR1,1(I+),1(H+),2,3',
                               'significant': 'ACR>=2R OR AMR-positive; not ACR>=1R',
                               'within': 'ANY recorded outcome in (anchor,anchor+H]; negative requires curated capture',
                               'next_within': 'conditional next observed biopsy within H days'},
               'evaluation': 'exploratory patient-grouped CV; val used for checkpoint selection, not independent test',
               'tasks': {}}
    task_rows = defaultdict(list)
    for event, sid, anchor, source in anchors:
        for endpoint in endpoints:
            for outcome in OUTCOMES:
                task = f'{endpoint}_{outcome}'
                cov = coverage.get((anchor['case_id'], anchor['tpl']))
                value, status, target_date = future_target(anchor, episodes[anchor['episode']], outcome, endpoint, cov)
                row = {'case_id': anchor['case_id'], 'slide_id': sid, 'label': value,
                       'event_id': event, 'anchor_date': anchor['date'].isoformat(),
                       'transplant_date': anchor['tpl'].isoformat(),
                       'target_date': target_date.isoformat() if target_date else '',
                       'source_dataset': source.get('source_dataset', ''), 'status': status,
                       'endpoint': endpoint, 'outcome': outcome}
                task_rows[task].append(row)
    for task, rows in task_rows.items():
        known = [r for r in rows if r['label'] is not None]
        report = {'anchor_slides': len(rows), 'known_slides': len(known),
                  'unknown_slides': len(rows) - len(known), 'status': dict(Counter(r['status'] for r in rows)),
                  'patients': len({r['case_id'] for r in known}),
                  'events': len({(r['case_id'], r['anchor_date']) for r in known}),
                  'positive_events': len({(r['case_id'], r['anchor_date']) for r in known if r['label'] == 1}),
                  'positive_patients': len({r['case_id'] for r in known if r['label'] == 1}),
                  'ready': False}
        summary['tasks'][task] = report
        if args.audit_only:
            continue
        folder = out / task
        folder.mkdir()
        write_csv(folder / 'label_audit.csv', rows)
        if not known:
            report['blocked_reason'] = 'No known future labels'
            continue
        data = pd.DataFrame(known)
        data['label'] = data.label.astype(int)
        # Model CSV has only current WSI identifiers and future label. No target
        # dates, future intervals or current/future grade fields enter the model.
        data[['case_id', 'slide_id', 'label', 'event_id', 'source_dataset']].to_csv(folder / 'cohort.csv', index=False)
        data.to_csv(folder / 'provenance.csv', index=False)
        if task.startswith('within_') and not (data.label == 0).any():
            report['blocked_reason'] = 'No confirmed negatives: provide reviewed outcome-ascertainment coverage; no-event records are unknown'
            continue
        try:
            prepared = {seed: make_splits(data, args.folds, seed) for seed in seeds}
        except ValueError as exc:
            report['blocked_reason'] = str(exc)
            continue
        fold_reports = []
        for seed, partitions in prepared.items():
            root = folder / f'seed_{seed}'
            root.mkdir()
            for fold, (tr, va) in enumerate(partitions):
                pd.DataFrame({name: pd.Series(values, dtype=object) for name, values in
                              {'train': tr.slide_id.tolist(), 'val': va.slide_id.tolist(), 'test': []}.items()}).to_csv(
                                  root / f'splits_{fold}.csv', index=False)
                for name, subset in [('train', tr), ('val', va)]:
                    fold_reports.append(dict(seed=seed, fold=fold, split=name, slides=len(subset),
                                             patients=subset.case_id.nunique(), positive_slides=int(subset.label.sum()),
                                             positive_patients=subset[subset.label == 1].case_id.nunique()))
        write_csv(folder / 'fold_summary.csv', fold_reports)
        report['ready'] = True
    if not args.audit_only:
        write_csv(out / 'linkage_exclusions.csv', audit, ['slide_id', 'event_id', 'status'])
        # Blank review fields by design. Never manufacture follow-up from last biopsy.
        review = {(a['case_id'], a['tpl']): a for _, _, a, _ in anchors}
        write_csv(out / 'coverage_review_template.csv', [dict(case_id=k[0], transplant_date=k[1].isoformat(),
                  ascertainment_start_date='', ascertainment_end_date='', complete_rejection_capture='',
                  review_source='') for k in review])
    summary['anchor_unit'] = 'WSI workbook rows, not feature-verified slides' if args.audit_only else 'feature-verified WSI slides'
    summary['generated_sha256'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*.csv'))}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ehr-xlsx', type=Path, default=Path('/home/jupyter/data/image_team/labels/raw/biopsy_ehr_3722.xlsx'))
    p.add_argument('--wsi-xlsx', type=Path, default=Path('/home/jupyter/data/image_team/labels/raw/WSI_LABEL_ID_MATCH_20260814.xlsx'))
    p.add_argument('--gold-csv', type=Path, default=Path('dataset_csv/smc_acr_binary_0r_vs_1r2r3r.csv'))
    p.add_argument('--baseline-dir', type=Path, default=Path('data/features/uni_v2/l0_0p25mpp_40x'))
    p.add_argument('--output-dir', type=Path, default=Path('cohorts/smc_future_v1'))
    p.add_argument('--endpoints', default='next_biopsy,within_30d,within_60d,within_90d')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seeds', default='1,11,21,31,41')
    p.add_argument('--coverage-csv', type=Path)
    p.add_argument('--audit-only', action='store_true', help='Workbook linkage/counts only; no feature check or train-ready CSVs')
    return p


if __name__ == '__main__':
    prepare(parser().parse_args())
