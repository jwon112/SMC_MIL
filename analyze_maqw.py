"""
M-AQW 결과 CSV 분석: 분포 타입별 클러스터링 및 τ/k 박스플롯·산점도 시각화.
단일 지표(laplacian만) 및 다중 지표(laplacian + stain_saturation + contrast) 대응.
출력은 out_dir 아래 combined/, indicators/<이름>/ 등 폴더로 정리.
Usage:
  python analyze_maqw.py --csv maqw_test_details.csv [maqw_val_details.csv ...] --out_dir ./maqw_plots
  python analyze_maqw.py --results_dir ./results/exp_s1 --out_dir ./maqw_plots
"""
import os
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Multi-indicator M-AQW: CSV column suffix _0, _1, _2 -> indicator name
INDICATOR_NAMES = ['laplacian', 'stain_saturation', 'contrast']


def load_maqw_csv(path):
    """Load M-AQW details CSV; parse q_hist10, w_hist10 into arrays."""
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            for key in list(row.keys()):
                if key in ('q_hist10', 'w_hist10') and row[key]:
                    row[key] = np.array([float(x) for x in row[key].split(',')], dtype=np.float64)
            rows.append(row)
    return rows


def load_all_csvs(csv_paths):
    """Load one or more CSVs; merge rows with a 'source' column (filename stem)."""
    all_rows = []
    for p in csv_paths:
        if not os.path.isfile(p):
            print('Warning: file not found:', p)
            continue
        rows = load_maqw_csv(p)
        stem = os.path.splitext(os.path.basename(p))[0]  # e.g. maqw_test_details
        for r in rows:
            r['_source'] = stem
        all_rows.extend(rows)
    return all_rows


def cluster_by_q_hist(rows, n_clusters=3, seed=42):
    """
    Cluster rows by 10-dim q_hist10; add 'cluster' and 'cluster_name'.
    Names: low_q_dominant, mid, high_q_dominant by mean bin index (weighted by density).
    """
    if not HAS_SKLEARN or not rows:
        return rows
    X = np.stack([np.asarray(r['q_hist10'], dtype=np.float64) for r in rows if r.get('q_hist10') is not None])
    if len(X) != len(rows):
        return rows
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)
    # mean bin index per cluster (0..9): low index = mass at low q (blur), high = high q (sharp/artifact)
    bin_centers = np.linspace(0.05, 0.95, 10)
    for i, r in enumerate(rows):
        r['cluster'] = int(labels[i])
        r['cluster_mean_bin'] = float(np.sum(bin_centers * r['q_hist10']))
    # name clusters by global mean bin index of cluster centroid
    cluster_mean_bin = []
    for k in range(n_clusters):
        idx = [i for i, r in enumerate(rows) if r['cluster'] == k]
        cluster_mean_bin.append(np.mean([rows[i]['cluster_mean_bin'] for i in idx]))
    order = np.argsort(cluster_mean_bin)  # 0=lowest mean bin -> "low_q", etc.
    name_map = {}
    for k in range(n_clusters):
        name_map[order[k]] = ['low_q_dominant', 'mid', 'high_q_dominant'][k] if n_clusters == 3 else f'cluster_{k}'
    for r in rows:
        r['cluster_name'] = name_map[r['cluster']]
    return rows


def ensure_numeric(rows, keys):
    for r in rows:
        for k in keys:
            if k in r and r[k] is not None:
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
    return rows


def plot_boxplots(rows, out_dir, prefix='maqw'):
    """Boxplots of tau_L, tau_R, k_L, k_R by cluster_name."""
    if not rows or 'cluster_name' not in rows[0]:
        return
    keys = ['tau_L', 'tau_R', 'k_L', 'k_R']
    rows = ensure_numeric(rows, keys)
    clusters = sorted(set(r['cluster_name'] for r in rows))
    data = {k: [r[k] for r in rows if k in r and r[k] is not None] for k in keys}
    if not data:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, k in enumerate(keys):
        ax = axes[i]
        by_cluster = defaultdict(list)
        for r in rows:
            if k in r and r[k] is not None:
                by_cluster[r['cluster_name']].append(r[k])
        positions = []
        box_data = []
        for c in clusters:
            if c in by_cluster:
                box_data.append(by_cluster[c])
                positions.append(len(positions))
        if box_data:
            bp = ax.boxplot(box_data, positions=positions, tick_labels=clusters, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
        ax.set_title(k)
        ax.set_ylabel(k)
    plt.suptitle('M-AQW parameters by quality distribution cluster')
    plt.tight_layout()
    path = os.path.join(out_dir, f'{prefix}_boxplots.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def plot_scatter_tau(rows, out_dir, prefix='maqw', color_by='cluster_name'):
    """Scatter tau_L vs tau_R; color by cluster or by correct/incorrect."""
    rows = ensure_numeric(rows, ['tau_L', 'tau_R', 'label', 'Y_hat'])
    if not rows or 'tau_L' not in rows[0]:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    tau_L = [r['tau_L'] for r in rows]
    tau_R = [r['tau_R'] for r in rows]
    if color_by == 'cluster_name' and 'cluster_name' in rows[0]:
        clusters = sorted(set(r['cluster_name'] for r in rows))
        try:
            cmap = plt.colormaps['tab10'].resampled(max(len(clusters), 1))
        except AttributeError:
            cmap = plt.cm.get_cmap('tab10', max(len(clusters), 1))
        for i, c in enumerate(clusters):
            idx = [j for j, r in enumerate(rows) if r.get('cluster_name') == c]
            if idx:
                ci = i / max(len(clusters) - 1, 1) if len(clusters) > 1 else 0.0
                ax.scatter([tau_L[j] for j in idx], [tau_R[j] for j in idx], label=c, alpha=0.7, s=40, color=cmap(ci))
        ax.legend(loc='best', fontsize=8)
    else:
        # color by correct (1) vs incorrect (0)
        correct = [1 if r.get('label') == r.get('Y_hat') else 0 for r in rows]
        sc = ax.scatter(tau_L, tau_R, c=correct, cmap='RdYlGn', alpha=0.7, s=40, vmin=-0.5, vmax=1.5)
        cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1])
        cbar.set_label('correct (1=yes)')
    ax.set_xlabel('tau_L')
    ax.set_ylabel('tau_R')
    ax.set_title('tau_L vs tau_R (by {})'.format(color_by))
    plt.tight_layout()
    path = os.path.join(out_dir, f'{prefix}_scatter_tau_L_R_{color_by}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def plot_w_summary(rows, out_dir, prefix='maqw'):
    """Boxplot w_mean, w_lt_0p1, w_gt_0p9 by cluster."""
    if not rows or 'cluster_name' not in rows[0]:
        return
    keys = ['w_mean', 'w_lt_0p1', 'w_gt_0p9']
    rows = ensure_numeric(rows, keys)
    clusters = sorted(set(r['cluster_name'] for r in rows))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    by_cluster = defaultdict(list)
    for r in rows:
        if 'w_mean' in r and r['w_mean'] is not None:
            by_cluster[r['cluster_name']].append(r['w_mean'])
    box_data = [by_cluster[c] for c in clusters if c in by_cluster]
    if box_data:
        ax.boxplot(box_data, tick_labels=clusters, patch_artist=True)
        ax.set_ylabel('w_mean')
        ax.set_title('Mean weight by cluster')
    plt.tight_layout()
    path = os.path.join(out_dir, f'{prefix}_w_mean_by_cluster.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def is_multi_indicator(rows):
    """True if CSV has per-indicator columns (tau_L_0, tau_L_1, ...)."""
    if not rows:
        return False
    return 'tau_L_0' in rows[0] or 'tau_L_1' in rows[0]


def build_rows_for_indicator(rows, indicator_index):
    """Build virtual rows with tau_L, tau_R, k_L, k_R from tau_L_<i>, etc.; copy cluster_name, label, Y_hat, w_*."""
    param_keys = ['tau_L', 'tau_R', 'k_L', 'k_R']
    suffix = f'_{indicator_index}'
    out = []
    for r in rows:
        nr = {k: v for k, v in r.items() if k in ('cluster_name', 'label', 'Y_hat', 'w_mean', 'w_lt_0p1', 'w_gt_0p9')}
        for p in param_keys:
            key = p + suffix
            if key in r and r[key] is not None:
                nr[p] = r[key]
        if all(p in nr for p in param_keys):
            out.append(nr)
    return out if out else rows


def write_cluster_summary(rows, out_path, param_keys=None):
    """Write cluster summary CSV for given rows and param keys."""
    if not rows or 'cluster_name' not in rows[0]:
        return
    if param_keys is None:
        param_keys = ('tau_L', 'tau_R', 'k_L', 'k_R', 'q_mean', 'w_mean', 'w_lt_0p1', 'w_gt_0p9')
    clusters = sorted(set(r['cluster_name'] for r in rows))
    summary_rows = []
    for c in clusters:
        sub = [r for r in rows if r['cluster_name'] == c]
        if not sub:
            continue
        sr = {'cluster_name': c, 'count': len(sub)}
        for k in param_keys:
            vals = [r[k] for r in sub if k in r and r[k] is not None]
            if vals:
                sr[f'{k}_mean'] = float(np.mean(vals))
                sr[f'{k}_std'] = float(np.std(vals))
        summary_rows.append(sr)
    if summary_rows:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print('Saved', out_path)


def plot_representative_W_from_summary(summary_csv, out_dir, prefix='maqw'):
    """
    Read cluster_summary CSV (from write_cluster_summary) and plot a representative W(q) curve
    using the global mean of tau_L, tau_R, k_L, k_R.
    """
    if not os.path.isfile(summary_csv):
        return
    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    # 전체 클러스터 평균으로 대표 파라미터 계산
    if not all(col in df.columns for col in ['tau_L_mean', 'tau_R_mean', 'k_L_mean', 'k_R_mean']):
        return
    tau_L = df['tau_L_mean'].mean()
    tau_R = df['tau_R_mean'].mean()
    k_L = df['k_L_mean'].mean()
    k_R = df['k_R_mean'].mean()

    q = np.linspace(0.0, 1.0, 500)

    # M-AQW와 동일한 수식 (plateau: min(2*w_left, 2*w_right, 1.0))
    w_left = 1.0 / (1.0 + np.exp(-k_L * (q - tau_L)))
    w_right = 1.0 / (1.0 + np.exp(-k_R * (tau_R - q)))
    W = np.minimum(2.0 * w_left, 2.0 * w_right)
    W = np.clip(W, 0.0, 1.0)

    plt.figure(figsize=(6, 4))
    plt.plot(q, W, label='avg W(q)')
    plt.xlabel('q (normalized quality)')
    plt.ylabel('W(q)')
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.title('Representative M-AQW weight curve')
    plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{prefix}_representative_W_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def plot_mean_q_w_hist(rows, out_dir, prefix='maqw'):
    """
    Plot mean q_hist10 and w_hist10 across rows.
    Requires that load_maqw_csv has parsed q_hist10 and w_hist10 into numpy arrays.
    """
    if not rows:
        return
    q_hists = [np.asarray(r['q_hist10'], dtype=np.float64)
               for r in rows if isinstance(r.get('q_hist10'), np.ndarray)]
    w_hists = [np.asarray(r['w_hist10'], dtype=np.float64)
               for r in rows if isinstance(r.get('w_hist10'), np.ndarray)]
    if not q_hists or not w_hists:
        return

    mean_q = np.mean(q_hists, axis=0)
    mean_w = np.mean(w_hists, axis=0)

    # 10-bin histogram over [0,1] (same as _slide_stats_and_histogram)
    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2.0

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(centers, mean_q, width=0.09)
    plt.xlabel('q (normalized quality)')
    plt.ylabel('density')
    plt.title('Mean q distribution')

    plt.subplot(1, 2, 2)
    plt.bar(centers, mean_w, width=0.09, color='orange')
    plt.xlabel('w')
    plt.ylabel('density')
    plt.title('Mean w distribution')

    plt.tight_layout()
    path = os.path.join(out_dir, f'{prefix}_mean_q_w_hist.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', path)


def main():
    parser = argparse.ArgumentParser(description='Analyze M-AQW CSV outputs: cluster and plot.')
    parser.add_argument('--csv', nargs='+', default=[], help='Paths to maqw_*_details.csv')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='If set, look for maqw_val_details.csv and maqw_test_details.csv here')
    parser.add_argument('--out_dir', type=str, default='./maqw_plots', help='Output directory (combined + indicators/*)')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters (default 3)')
    parser.add_argument('--no_cluster', action='store_true', help='Skip clustering; only plot scatter (no cluster_name)')
    args = parser.parse_args()

    csv_paths = list(args.csv)
    if args.results_dir:
        for name in ('maqw_val_details.csv', 'maqw_test_details.csv'):
            p = os.path.join(args.results_dir, name)
            if os.path.isfile(p) and p not in csv_paths:
                csv_paths.append(p)
    if not csv_paths:
        print('No CSV files given. Use --csv or --results_dir.')
        return

    rows = load_all_csvs(csv_paths)
    if not rows:
        print('No rows loaded.')
        return

    multi = is_multi_indicator(rows)
    numeric_keys = ['tau_L', 'tau_R', 'k_L', 'k_R', 'q_mean', 'q_std', 'w_mean', 'w_lt_0p1', 'w_gt_0p9', 'label', 'Y_hat']
    if multi:
        for i in range(len(INDICATOR_NAMES)):
            numeric_keys.extend([f'tau_L_{i}', f'k_L_{i}', f'tau_R_{i}', f'k_R_{i}'])
    rows = ensure_numeric(rows, numeric_keys)
    if not args.no_cluster and HAS_SKLEARN:
        rows = cluster_by_q_hist(rows, n_clusters=args.n_clusters)
    else:
        if args.no_cluster:
            for r in rows:
                r['cluster_name'] = 'all'
        elif not HAS_SKLEARN:
            print('sklearn not found; skipping clustering. Install scikit-learn for cluster plots.')
            for r in rows:
                r['cluster_name'] = 'all'

    # 사용자가 직접 out_dir를 결과 폴더(예: ./results/exp_xxx)로 줄 때,
    # 그 안에 하위 폴더(maqw_plots)를 만들어 정리하도록 함.
    if args.out_dir == './maqw_plots':
        out_root = args.out_dir
    else:
        out_root = os.path.join(args.out_dir, 'maqw_plots')

    os.makedirs(out_root, exist_ok=True)
    prefix = 'maqw'
    if args.results_dir:
        prefix = os.path.basename(args.results_dir.rstrip(os.sep)) or 'maqw'

    # Combined (mean params for multi, or single-indicator)
    combined_dir = os.path.join(out_root, 'combined')
    os.makedirs(combined_dir, exist_ok=True)
    plot_boxplots(rows, combined_dir, prefix=prefix)
    plot_scatter_tau(rows, combined_dir, prefix=prefix, color_by='cluster_name')
    plot_scatter_tau(rows, combined_dir, prefix=prefix, color_by='correct')
    plot_w_summary(rows, combined_dir, prefix=prefix)
    summary_csv_path = os.path.join(combined_dir, f'{prefix}_cluster_summary.csv')
    write_cluster_summary(rows, summary_csv_path)
    # Representative W(q) curve from mean tau/k, and empirical q/w histograms
    plot_representative_W_from_summary(summary_csv_path, combined_dir, prefix=prefix)
    plot_mean_q_w_hist(rows, combined_dir, prefix=prefix)

    # Per-indicator (multi only)
    if multi:
        for i, name in enumerate(INDICATOR_NAMES):
            if f'tau_L_{i}' not in rows[0]:
                continue
            ind_dir = os.path.join(out_root, 'indicators', name)
            os.makedirs(ind_dir, exist_ok=True)
            rows_i = build_rows_for_indicator(rows, i)
            if not rows_i:
                continue
            rows_i = ensure_numeric(rows_i, ['tau_L', 'tau_R', 'k_L', 'k_R'])
            plot_boxplots(rows_i, ind_dir, prefix=name)
            plot_scatter_tau(rows_i, ind_dir, prefix=name, color_by='cluster_name')
            plot_scatter_tau(rows_i, ind_dir, prefix=name, color_by='correct')
            plot_w_summary(rows_i, ind_dir, prefix=name)
            write_cluster_summary(rows_i, os.path.join(ind_dir, f'{name}_cluster_summary.csv'), param_keys=('tau_L', 'tau_R', 'k_L', 'k_R', 'w_mean', 'w_lt_0p1', 'w_gt_0p9'))

    print('Done. Outputs in', out_root, '(combined/ and indicators/<name>/)')


if __name__ == '__main__':
    main()
