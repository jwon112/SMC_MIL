"""
Compute correlation matrix for patch-level quality metrics across feature H5 files.
Use this to check multicollinearity before using multiple indicators in M-AQW.
Usage:
  python compute_metric_correlations.py --feat_dir ./data/features/conch_v1_5/maqw
  python compute_metric_correlations.py --feat_dir ./feats --csv_path slides.csv --max_patches_per_slide 5000 --out_csv corr.csv
"""
import os
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

METRIC_KEYS = ['laplacian_scores', 'stain_saturation', 'color_entropy', 'contrast']


def get_slide_ids_from_feat_dir(feat_dir):
    """List slide IDs from feat_dir/h5_files/*.h5."""
    h5_dir = os.path.join(feat_dir, 'h5_files')
    if not os.path.isdir(h5_dir):
        return []
    return sorted([os.path.splitext(f)[0] for f in os.listdir(h5_dir) if f.endswith('.h5')])


def get_slide_ids_from_csv(csv_path):
    """List slide IDs from CSV (slide_id column or first column)."""
    df = pd.read_csv(csv_path)
    if 'slide_id' in df.columns:
        return df['slide_id'].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def load_metrics_from_h5(path, keys=None, max_patches=None, rng=None):
    """
    Load metric arrays from one H5. Returns (N, n_keys) array or None if any key missing.
    """
    if keys is None:
        keys = METRIC_KEYS
    with h5py.File(path, 'r') as f:
        for k in keys:
            if k not in f:
                return None
        n = f[keys[0]].shape[0]
        if n == 0:
            return None
        idx = np.arange(n)
        if max_patches is not None and n > max_patches:
            if rng is None:
                rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_patches, replace=False)
        rows = []
        for k in keys:
            rows.append(f[k][:][idx])
        return np.column_stack(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Compute correlation matrix of patch metrics (laplacian, stain_saturation, color_entropy, contrast) across H5 files.'
    )
    parser.add_argument('--feat_dir', type=str, required=True,
                        help='Feature directory containing h5_files/')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Optional: only use slide IDs from this CSV')
    parser.add_argument('--max_patches_per_slide', type=int, default=2000,
                        help='Max patches per slide to use (sample if larger; default 2000)')
    parser.add_argument('--max_slides', type=int, default=None,
                        help='Optional: cap number of slides to process')
    parser.add_argument('--out_csv', type=str, default=None,
                        help='Save correlation matrix to this CSV')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.csv_path and os.path.isfile(args.csv_path):
        slide_ids = get_slide_ids_from_csv(args.csv_path)
        h5_dir = os.path.join(args.feat_dir, 'h5_files')
        slide_ids = [s for s in slide_ids if os.path.isfile(os.path.join(h5_dir, s + '.h5'))]
    else:
        slide_ids = get_slide_ids_from_feat_dir(args.feat_dir)

    if not slide_ids:
        print('No slides found. Check --feat_dir and --csv_path.')
        return

    if args.max_slides is not None:
        slide_ids = slide_ids[: args.max_slides]

    rng = np.random.default_rng(args.seed)
    all_rows = []
    missing = []
    for slide_id in tqdm(slide_ids, desc='Loading'):
        path = os.path.join(args.feat_dir, 'h5_files', slide_id + '.h5')
        arr = load_metrics_from_h5(
            path,
            keys=METRIC_KEYS,
            max_patches=args.max_patches_per_slide,
            rng=rng,
        )
        if arr is None:
            missing.append(slide_id)
            continue
        all_rows.append(arr)

    if not all_rows:
        print('No valid metric data (all slides missing one of {}).'.format(METRIC_KEYS))
        if missing:
            print('Slides skipped (first 5):', missing[:5])
        return

    data = np.vstack(all_rows)
    n_patches = data.shape[0]
    n_slides_used = len(all_rows)
    print('Loaded {} patches from {} slides.'.format(n_patches, n_slides_used))
    if missing:
        print('Skipped {} slides (missing metrics).'.format(len(missing)))

    # Pearson correlation
    corr = np.corrcoef(data.T)
    df_corr = pd.DataFrame(corr, index=METRIC_KEYS, columns=METRIC_KEYS)

    print('\n' + '=' * 60)
    print('Pearson correlation matrix (patch-level)')
    print('=' * 60)
    print(df_corr.to_string(float_format='%.4f'))
    print()

    # Summary: high correlations (|r| > 0.5 or 0.7)
    print('Pairs with |r| > 0.7 (consider dropping one):')
    for i in range(len(METRIC_KEYS)):
        for j in range(i + 1, len(METRIC_KEYS)):
            r = corr[i, j]
            if abs(r) > 0.7:
                print('  {} vs {}: r = {:.4f}'.format(METRIC_KEYS[i], METRIC_KEYS[j], r))
    print('Pairs with 0.5 < |r| <= 0.7:')
    for i in range(len(METRIC_KEYS)):
        for j in range(i + 1, len(METRIC_KEYS)):
            r = corr[i, j]
            if 0.5 < abs(r) <= 0.7:
                print('  {} vs {}: r = {:.4f}'.format(METRIC_KEYS[i], METRIC_KEYS[j], r))

    if args.out_csv:
        df_corr.to_csv(args.out_csv)
        print('\nSaved correlation matrix to', args.out_csv)

    # Optional: heatmap if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(METRIC_KEYS)))
        ax.set_yticks(range(len(METRIC_KEYS)))
        ax.set_xticklabels(METRIC_KEYS, rotation=45, ha='right')
        ax.set_yticklabels(METRIC_KEYS)
        for i in range(len(METRIC_KEYS)):
            for j in range(len(METRIC_KEYS)):
                ax.text(j, i, '{:.2f}'.format(corr[i, j]), ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax, label='Pearson r')
        plt.tight_layout()
        out_plot = (args.out_csv.replace('.csv', '_heatmap.png') if args.out_csv
                    else 'metric_correlation_heatmap.png')
        plt.savefig(out_plot, dpi=150, bbox_inches='tight')
        plt.close()
        print('Saved heatmap to', out_plot)
    except Exception as e:
        pass  # no matplotlib or save failed


if __name__ == '__main__':
    main()
