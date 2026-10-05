"""Patient-only nested partitions and prespecified threshold selection."""
import numpy as np
import pandas as pd


def validate_events(events):
    if events[['event_id', 'case_id', 'label']].isna().any().any():
        raise ValueError('Missing event, patient or label')
    if events.event_id.duplicated().any() or not events.label.isin([0, 1]).all():
        raise ValueError('Duplicate events or nonbinary labels')


def patient_folds(events, n_splits, seed):
    """Balance positive patients first, then distribute negative patients by event count.

    A patient with any positive event is a positive patient. Mixed-label patients
    stay together. This is a predetermined split rule, not a performance search.
    """
    validate_events(events)
    patients = events.groupby('case_id').label.agg(['max', 'size'])
    if n_splits < 2 or (patients['max'] == 1).sum() < n_splits or (patients['max'] == 0).sum() < n_splits:
        raise ValueError('Too few positive/negative patients for requested inner folds')
    rng = np.random.default_rng(seed)
    assignments = {}; totals = np.zeros(n_splits, dtype=int)
    for label in [1, 0]:
        ids = patients.index[patients['max'] == label].to_numpy().copy()
        rng.shuffle(ids)
        for i, case in enumerate(ids):
            fold = i % n_splits if label == 1 or i < n_splits else int(np.argmin(totals))
            assignments[case] = fold
            totals[fold] += int(patients.loc[case, 'size'])
    assigned = events.case_id.map(assignments).to_numpy()
    return [(events.loc[assigned != i].copy(), events.loc[assigned == i].copy()) for i in range(n_splits)]


def stop_partition(events, seed):
    patients = events.groupby('case_id').label.max()
    n = min(5, int((patients == 1).sum()), int((patients == 0).sum()))
    if n < 2:
        raise ValueError('Need at least two positive and two negative patients for early stopping')
    train, stop = patient_folds(events, n, seed)[0]
    assert_disjoint(train, stop)
    return train, stop


def assert_disjoint(*frames):
    for i, a in enumerate(frames):
        for b in frames[i + 1:]:
            if set(a.case_id) & set(b.case_id) or set(a.event_id) & set(b.event_id):
                raise ValueError('Patient/event leakage between partitions')


def choose_threshold(frame, beta):
    validate_events(frame)
    if not frame.probability.between(0, 1).all() or frame.label.nunique() != 2:
        raise ValueError('Threshold calibration needs valid probabilities and both classes')
    if beta <= 0:
        raise ValueError('beta must be positive')
    y = frame.label.to_numpy(int); p = frame.probability.to_numpy(float)
    candidates = np.r_[np.unique(p), np.nextafter(p.max(), np.inf)]
    options = []
    for threshold in candidates:
        pred = p >= threshold
        tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        value = (1 + beta**2)*tp / ((1 + beta**2)*tp + beta**2*fn + fp)
        options.append((value, float(threshold)))
    # Ties prefer fewer positive calls via the higher threshold.
    return max(options)[1]


def decision_metrics(frame, prediction):
    from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
    y = frame.label.to_numpy(int); p = frame.probability.to_numpy(float)
    pred = np.asarray(prediction, dtype=bool)
    if len(pred) != len(y):
        raise ValueError('Prediction length mismatch')
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    tn = int(((y == 0) & ~pred).sum()); fn = int(((y == 1) & ~pred).sum())
    return dict(events=len(y), positive_events=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
                precision=tp/(tp+fp) if tp+fp else float('nan'),
                sensitivity=tp/(tp+fn) if tp+fn else float('nan'),
                specificity=tn/(tn+fp) if tn+fp else float('nan'),
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                f2=5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,
                mcc=float(matthews_corrcoef(y, pred)),
                auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan'),
                pr_auc=float(average_precision_score(y, p)) if y.sum() else float('nan'))
