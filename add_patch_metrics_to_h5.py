"""
Add patch-level metrics (stain saturation, color entropy, contrast; optionally Laplacian)
to existing feature H5 files. Metrics are computed from WSI patch images, not from features.
Usage:
  python add_patch_metrics_to_h5.py --feat_dir ./feats --data_h5_dir ./patches_root --data_slide_dir ./wsi --csv_path slides.csv
  python add_patch_metrics_to_h5.py --feat_dir ./feats --data_h5_dir ./patches_root --data_slide_dir ./wsi  # use all H5 in feat_dir/h5_files
"""
import os
import argparse
import h5py
import numpy as np
import openslide
from tqdm import tqdm

from utils.patch_metrics import compute_patch_metrics


def get_slide_ids_from_feat_dir(feat_dir):
    """List slide IDs from existing feature H5 files (feat_dir/h5_files/*.h5)."""
    h5_dir = os.path.join(feat_dir, 'h5_files')
    if not os.path.isdir(h5_dir):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(h5_dir) if f.endswith('.h5')]


def get_slide_ids_from_csv(csv_path, slide_ext='.svs'):
    """List slide IDs from CSV (column slide_id or first column)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if 'slide_id' in df.columns:
        return df['slide_id'].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def add_metrics_to_slide(
    slide_id,
    feat_dir,
    data_h5_dir,
    data_slide_dir,
    slide_ext='.svs',
    add_laplacian=False,
    downsample=2,
    skip_existing=True,
):
    """
    For one slide: read feature H5 coords, read patches from WSI, compute metrics, write to same H5.
    Returns (success: bool, n_patches: int or 0, message: str).
    """
    feat_h5_path = os.path.join(feat_dir, 'h5_files', slide_id + '.h5')
    patch_h5_path = os.path.join(data_h5_dir, 'patches', slide_id + '.h5')
    wsi_path = os.path.join(data_slide_dir, slide_id + slide_ext)

    if not os.path.isfile(feat_h5_path):
        return False, 0, 'Feature H5 not found: {}'.format(feat_h5_path)
    if not os.path.isfile(patch_h5_path):
        return False, 0, 'Patch H5 not found: {}'.format(patch_h5_path)
    if not os.path.isfile(wsi_path):
        return False, 0, 'WSI not found: {}'.format(wsi_path)

    with h5py.File(feat_h5_path, 'r') as f:
        coords = f['coords'][:]
        n = len(coords)
        if n == 0:
            return False, 0, 'No coords in feature H5'
        # Optionally skip if all metrics already present
        if skip_existing:
            if 'stain_saturation' in f and 'color_entropy' in f and 'contrast' in f:
                if not add_laplacian or 'laplacian_scores' in f:
                    return True, n, 'already has metrics'

    with h5py.File(patch_h5_path, 'r') as f:
        if 'coords' not in f:
            return False, 0, 'Patch H5 has no coords'
        patch_level = int(f['coords'].attrs.get('patch_level', 0))
        patch_size = int(f['coords'].attrs.get('patch_size', 256))

    wsi = openslide.open_slide(wsi_path)
    stain_sat = []
    color_ent = []
    contrast_list = []
    laplacian_list = [] if add_laplacian else None

    try:
        for i in range(n):
            coord = tuple(int(x) for x in coords[i])
            pil = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
            metrics = compute_patch_metrics(pil, downsample=downsample, include_laplacian=add_laplacian)
            stain_sat.append(metrics['stain_saturation'])
            color_ent.append(metrics['color_entropy'])
            contrast_list.append(metrics['contrast'])
            if add_laplacian:
                laplacian_list.append(metrics['laplacian'])
    finally:
        wsi.close()

    stain_sat = np.array(stain_sat, dtype=np.float32)
    color_ent = np.array(color_ent, dtype=np.float32)
    contrast_arr = np.array(contrast_list, dtype=np.float32)

    with h5py.File(feat_h5_path, 'r+') as f:
        def _write_or_overwrite(name, data):
            if name in f:
                del f[name]
            f.create_dataset(name, data=data, dtype=np.float32)

        _write_or_overwrite('stain_saturation', stain_sat)
        _write_or_overwrite('color_entropy', color_ent)
        _write_or_overwrite('contrast', contrast_arr)
        if add_laplacian and laplacian_list is not None:
            _write_or_overwrite('laplacian_scores', np.array(laplacian_list, dtype=np.float32))

    return True, n, 'ok'


def main():
    import torch
    parser = argparse.ArgumentParser(
        description='Add stain_saturation, color_entropy, contrast (and optionally Laplacian) to existing feature H5 from WSI patches.'
    )
    parser.add_argument('--feat_dir', type=str, required=True, help='Feature directory (contains h5_files/)')
    parser.add_argument('--data_h5_dir', type=str, required=True, help='Root dir containing patches/<slide_id>.h5 with coords attrs')
    parser.add_argument('--data_slide_dir', type=str, required=True, help='Directory of WSI files')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='CSV with slide_id column; if not set, use all H5 in feat_dir/h5_files')
    parser.add_argument('--add_laplacian', action='store_true', help='Also compute and save laplacian_scores if missing')
    parser.add_argument('--downsample', type=int, default=2, help='Downsample factor for metric computation (default 2)')
    parser.add_argument('--no_skip_existing', action='store_true', help='Recompute and overwrite even if metrics already exist')
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        torch.distributed.init_process_group(backend='gloo')

    if args.csv_path and os.path.isfile(args.csv_path):
        slide_ids = get_slide_ids_from_csv(args.csv_path, args.slide_ext)
    else:
        slide_ids = get_slide_ids_from_feat_dir(args.feat_dir)

    # Multi-GPU: this process handles slides where index % world_size == rank
    slide_ids = [slide_ids[i] for i in range(rank, len(slide_ids), world_size)]

    if not slide_ids:
        if rank == 0:
            print('No slides to process (check --feat_dir and --csv_path).')
        if world_size > 1:
            torch.distributed.destroy_process_group()
        return

    if rank == 0 or world_size == 1:
        print('Processing {} slides (this process: {}). feat_dir={}'.format(
            len(slide_ids) if world_size == 1 else 'total see above', len(slide_ids), args.feat_dir))
    ok, fail, total_patches = 0, 0, 0
    for slide_id in tqdm(slide_ids, desc='slides', position=rank, leave=(rank == 0)):
        success, n, msg = add_metrics_to_slide(
            slide_id,
            args.feat_dir,
            args.data_h5_dir,
            args.data_slide_dir,
            slide_ext=args.slide_ext,
            add_laplacian=args.add_laplacian,
            downsample=args.downsample,
            skip_existing=not args.no_skip_existing,
        )
        if success:
            ok += 1
            total_patches += n
        else:
            fail += 1
            tqdm.write('{}: {}'.format(slide_id, msg))

    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank == 0 or world_size == 1:
        print('Done. OK={}, Failed={}, Total patches with metrics={}'.format(ok, fail, total_patches))


if __name__ == '__main__':
    main()
