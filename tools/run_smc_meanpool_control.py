"""CPU mean-pooling/logistic baseline on the exact saved imbalance-control partitions."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.threshold_control import assert_disjoint, decision_metrics, validate_events

C_VALUES = (0.001, 0.01, 0.1, 1.0)
CAP = 2048
DIMENSION = 1536


def digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def blob_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def atomic_csv(path, frame):
    temp = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temp, index=False, lineterminator='\n')
    os.replace(temp, path)


def check_mapping(actual, expected):
    validate_events(actual)
    a = actual[['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True)
    b = expected[['event_id', 'case_id', 'label']].sort_values('event_id').reset_index(drop=True)
    if not a.equals(b):
        raise ValueError('Event/patient/label mapping differs from reference')


def load_inputs(args):
    root = args.reference_root.resolve()
    reference_protocol = json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    manifest = args.manifest_dir.resolve()
    events_path = manifest/'events.csv'; slides_path = manifest/'event_slides.csv'
    expected = {Path(k).name: v for k,v in reference_protocol['input_sha256'].items()
                if k.endswith(('/events.csv', '/event_slides.csv'))}
    for p in (events_path, slides_path):
        if digest(p) != expected[p.name]:
            raise ValueError(f'Training manifest changed: {p.name}')
    events = pd.read_csv(events_path, dtype={'event_id': str, 'case_id': str})
    slides = pd.read_csv(slides_path, dtype={'event_id': str, 'slide_id': str})
    validate_events(events)
    if slides.slide_id.duplicated().any() or set(slides.event_id) != set(events.event_id):
        raise ValueError('Invalid slide membership')
    if set(slides.slide_id) != set(reference_protocol['feature_inventory']):
        raise ValueError('Feature inventory and slide manifest differ')
    if not set(slides.stain_group) <= {'HE', 'IHC', 'other'}:
        raise ValueError('Unknown stain group')
    sources = [root/'protocol.json', events_path, slides_path, Path(__file__), PROJECT/'utils/threshold_control.py']
    partitions = {}
    for seed in reference_protocol['seeds']:
        test_events = []
        for fold in range(5):
            path = root/f'seed{seed}'/f'fold_{fold}'/'partitions.csv'
            sources.append(path)
            p = pd.read_csv(path, dtype={'event_id': str, 'case_id': str})
            check_mapping(p, events)
            if set(p.role) != {'fit', 'stop', 'test'}:
                raise ValueError('Invalid partition roles')
            roles = {role: p[p.role.eq(role)].sort_values('event_id').reset_index(drop=True)
                     for role in ['fit', 'stop', 'test']}
            assert_disjoint(*roles.values())
            if any(roles[r].label.nunique() != 2 for r in ['fit', 'stop']):
                raise ValueError('Fit/stop partition needs both classes')
            partitions[seed, fold] = roles
            test_events.extend(roles['test'].event_id)
        if sorted(test_events) != sorted(events.event_id):
            raise ValueError('Outer folds must test every event exactly once')
        for condition in reference_protocol['conditions']:
            path = root/f'seed{seed}'/f'{condition}_oof_predictions.csv'
            sources.append(path)
            f = pd.read_csv(path, dtype={'event_id': str, 'case_id': str})
            check_mapping(f, events)
            if not f.probability.between(0, 1).all() or not np.isfinite(f.probability).all():
                raise ValueError('Invalid reference probabilities')
            for fold in range(5):
                check_mapping(f[f.fold.eq(fold)], partitions[seed, fold]['test'])
    feature_dir = (args.feature_dir or Path(reference_protocol['feature_dir'])).resolve()
    protocol = dict(version=1, method='meanpool_logreg', primary_comparator='natural_ce',
                    seeds=reference_protocol['seeds'], reference_conditions=reference_protocol['conditions'],
                    feature_dir=str(feature_dir), dimension=DIMENSION, patch_cap=CAP,
                    pooling='torch float32 linspace patch selection; patch mean then equal slide mean; no stain branches/masks',
                    scaling='StandardScaler fit patients only', c_values=list(C_VALUES),
                    selection='minimum unweighted stop log loss; exact ties prefer smaller C; no fit+stop refit',
                    classifier='L2 logistic regression, lbfgs, class_weight=None, max_iter=2000, tol=1e-6',
                    threshold=0.5, sklearn_version=sklearn.__version__, numpy_version=np.__version__, torch_version=version('torch'),
                    reference_feature_inventory=reference_protocol['feature_inventory'],
                    input_sha256_lf={str(p.resolve()):digest(p) for p in sources})
    return events.sort_values('event_id').reset_index(drop=True), slides, partitions, protocol


def check_features(protocol):
    directory = Path(protocol['feature_dir'])/'pt_files'
    for slide, expected in protocol['reference_feature_inventory'].items():
        path = directory/f'{slide}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'Feature file missing: {path}; run on server or set --feature-dir')
        stat = path.stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Feature file metadata differs from reference: {slide}')


def pool_slide(features):
    import torch
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or not len(features) or features.shape[1] != DIMENSION:
        raise ValueError('Expected nonempty patch features with dimension 1536')
    features = features.float()
    if len(features) > CAP:
        features = features[torch.linspace(0, len(features)-1, CAP).long()]
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite selected patch features')
    return features.mean(dim=0)


def embeddings(events, slides, protocol, out):
    import torch
    cache = out/'event_embeddings.npz'; record = out/'embedding_cache.json'
    protocol_hash = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode('utf-8')).hexdigest()
    if cache.exists() and record.exists():
        saved = json.loads(record.read_text(encoding='utf-8'))
        if saved['sha256'] != blob_digest(cache) or saved['protocol_sha256'] != protocol_hash:
            raise ValueError('Embedding cache changed')
        with np.load(cache, allow_pickle=False) as data:
            ids = data['event_ids']; matrix = data['features'].copy()
        if ids.tolist() != events.event_id.tolist() or matrix.shape != (len(events), DIMENSION) or not np.isfinite(matrix).all():
            raise ValueError('Embedding cache membership/shape mismatch')
        print('[CACHE] Verified event embeddings', flush=True)
        return matrix
    grouped = slides.groupby('event_id', sort=False)
    vectors = []; inventory = []; directory = Path(protocol['feature_dir'])/'pt_files'
    for index, event in enumerate(events.event_id, 1):
        pooled = []
        for row in grouped.get_group(event).itertuples():
            path = directory/f'{row.slide_id}.pt'
            features = torch.load(path, map_location='cpu', weights_only=True)
            pooled.append(pool_slide(features))
            inventory.append(dict(event_id=event, slide_id=row.slide_id, patches=len(features), selected_patches=min(len(features), CAP)))
        vectors.append(torch.stack(pooled).mean(dim=0).numpy())
        if index == 1 or index % 25 == 0 or index == len(events):
            print(f'[POOL] events={index}/{len(events)} slides={len(inventory)}', flush=True)
    matrix = np.stack(vectors)
    temp = cache.with_suffix('.npz.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, event_ids=events.event_id.to_numpy(dtype=str), features=matrix)
    os.replace(temp, cache)
    atomic_csv(out/'embedding_audit.csv', pd.DataFrame(inventory))
    atomic_json(record, dict(sha256=blob_digest(cache), protocol_sha256=protocol_hash, events=len(events), dimension=DIMENSION,
                            supervised_fit=False, scope='Fixed per-event feature averages; no label fitting or cohort-wide normalization'))
    return matrix


def fit_candidates(fit_x, fit_y, stop_x, stop_y):
    scaler = StandardScaler().fit(fit_x)
    fit_scaled = scaler.transform(fit_x); stop_scaled = scaler.transform(stop_x)
    rows = []; best = None
    for c in C_VALUES:
        model = LogisticRegression(C=c, solver='lbfgs', class_weight=None, max_iter=2000, tol=1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(fit_scaled, fit_y)
        loss = float(log_loss(stop_y, model.predict_proba(stop_scaled), labels=[0, 1]))
        if not np.isfinite(loss):
            raise ValueError('Nonfinite stop log loss')
        rows.append(dict(C=c, stop_log_loss=loss, iterations=int(model.n_iter_.max())))
        if best is None or (loss, c) < best[:2]:
            best = (loss, c, model)
    return scaler, best[2], best[1], pd.DataFrame(rows)


def summarize(events, partitions, protocol, out, reference_root):
    rows = []; all_frames = []
    for seed in protocol['seeds']:
        pieces = []
        for fold in range(5):
            path = out/f'seed{seed}'/f'fold_{fold}'/'outer_predictions.csv'
            frame = pd.read_csv(path, dtype={'event_id': str, 'case_id': str})
            check_mapping(frame, partitions[seed, fold]['test'])
            if not frame.fold.eq(fold).all() or not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
                raise ValueError('Invalid completed predictions')
            pieces.append(frame)
        simple = pd.concat(pieces).sort_values('event_id').reset_index(drop=True)
        check_mapping(simple, events)
        atomic_csv(out/f'seed{seed}'/'meanpool_logreg_oof_predictions.csv', simple)
        for condition in ['meanpool_logreg']+protocol['reference_conditions']:
            frame = simple if condition == 'meanpool_logreg' else pd.read_csv(reference_root/f'seed{seed}'/f'{condition}_oof_predictions.csv')
            rows.append(dict(seed=seed, condition=condition, **decision_metrics(frame, frame.probability.ge(.5))))
            all_frames.append(frame.assign(seed=seed, condition=condition))
    metrics = pd.DataFrame(rows)
    atomic_csv(out/'per_seed_metrics.csv', metrics)
    columns = ['auroc','pr_auc','tp','fp','fn','precision','sensitivity','specificity','f1','f2','mcc']
    summary = metrics.groupby('condition')[columns].agg(['mean','std'])
    summary.to_csv(out/'seed_summary.csv')
    base = metrics[metrics.condition.eq('natural_ce')].set_index('seed')
    simple = metrics[metrics.condition.eq('meanpool_logreg')].set_index('seed')
    atomic_csv(out/'paired_deltas_vs_natural_ce.csv', (simple[columns]-base[columns]).reset_index())
    ensembles = []
    combined = pd.concat(all_frames, ignore_index=True)
    for condition, frame in combined.groupby('condition'):
        ensemble = frame.groupby(['event_id','case_id','label'],as_index=False).probability.mean()
        atomic_csv(out/f'{condition}_ensemble_predictions.csv', ensemble)
        ensembles.append(dict(condition=condition, **decision_metrics(ensemble,ensemble.probability.ge(.5))))
    atomic_csv(out/'ensemble_metrics.csv',pd.DataFrame(ensembles))
    print(summary.to_string(), flush=True)
    print('[COMPLETE] 5 folds per seed; primary comparison meanpool_logreg vs natural_ce; ensemble reported separately', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['audit','run','summarize'], default='audit')
    parser.add_argument('--reference-root', type=Path, default=PROJECT/'results/smc_event_imbalance_control_20261006_exclude25')
    parser.add_argument('--manifest-dir', type=Path, default=Path('/home/jupyter/data/image_team/labels/derived/event_stain_mil_gold_provisional_20260930/acr_high'))
    parser.add_argument('--feature-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'results/smc_event_meanpool_control_20261007_exclude25')
    args = parser.parse_args()
    events, slides, partitions, protocol = load_inputs(args)
    out = args.output_dir.resolve(); record = out/'protocol.json'
    if record.exists() and json.loads(record.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Inputs/code/environment/settings changed; use a new output directory')
    if not record.exists() and (any(out.glob('seed*')) or (out/'event_embeddings.npz').exists()):
        raise ValueError('Run artifacts exist without protocol; use a new output directory')
    if args.action != 'summarize':
        check_features(protocol)
    if args.action == 'audit':
        print(json.dumps(dict(events=len(events), positive_events=int(events.label.sum()), patients=events.case_id.nunique(),
                              positive_patients=events.loc[events.label.eq(1),'case_id'].nunique(), slides=len(slides),
                              partitions=len(partitions), candidate_fits=len(partitions)*len(C_VALUES), device='CPU',
                              c_values=list(C_VALUES), primary_comparator='natural_ce', feature_dir=protocol['feature_dir']),indent=2))
        return
    if args.action == 'summarize' and not record.exists():
        raise FileNotFoundError('Run protocol missing')
    out.mkdir(parents=True,exist_ok=True)
    if args.action == 'run':
        import torch
        torch.set_num_threads(2)
        atomic_json(record,protocol)
        matrix = embeddings(events,slides,protocol,out)
        lookup = {event:index for index,event in enumerate(events.event_id)}
        for (seed,fold),roles in partitions.items():
            dest = out/f'seed{seed}'/f'fold_{fold}'; dest.mkdir(parents=True,exist_ok=True)
            final = dest/'outer_predictions.csv'
            if final.exists():
                print(f'[SKIP] seed={seed} fold={fold}',flush=True)
                continue
            positions = {role:[lookup[e] for e in frame.event_id] for role,frame in roles.items()}
            scaler,model,c,candidates = fit_candidates(matrix[positions['fit']],roles['fit'].label.to_numpy(),
                                                      matrix[positions['stop']],roles['stop'].label.to_numpy())
            test = roles['test'][['event_id','case_id','label']].copy()
            test['probability'] = model.predict_proba(scaler.transform(matrix[positions['test']]))[:,1]
            test['fold'] = fold
            atomic_csv(dest/'candidates.csv',candidates)
            atomic_csv(dest/'partitions.csv',pd.concat([f.assign(role=role) for role,f in roles.items()])[['event_id','case_id','label','role']])
            atomic_json(dest/'selection.json',dict(seed=seed,fold=fold,selected_C=c,stop_log_loss=float(candidates.loc[candidates.C.eq(c),'stop_log_loss'].iloc[0]),
                                                 fit_events=len(roles['fit']),stop_events=len(roles['stop']),test_events=len(test),
                                                 coefficients=model.coef_.size+model.intercept_.size,no_refit=True))
            np.savez_compressed(dest/'model.npz',coef=model.coef_,intercept=model.intercept_,scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
            atomic_csv(final,test)
            print(f'[OK] seed={seed} fold={fold} C={c}',flush=True)
    summarize(events,partitions,protocol,out,args.reference_root)


if __name__ == '__main__':
    main()
