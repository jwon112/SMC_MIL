"""
Create a noisy patch set from clean patch images. Output has the same structure as the input
(same slide_id, same patch order) so dataset_csv can be used as-is and pairing is by (slide_id, patch_idx).
Noisy patches can then be passed through your feature extractor and saved to e.g. ./data/features/.../noisy.
Usage:
  # Patch H5 with 'imgs' stored (patch_h5_dir/patches/slide_id.h5):
  python create_noisy_patches.py --patch_h5_dir ./data/patches --output_dir ./data/patches_noisy --noise_type gaussian --sigma 15

  # Patch H5 with only 'coords' (read from WSI on the fly):
  python create_noisy_patches.py --patch_h5_dir ./data/patches --output_dir ./data/patches_noisy --data_slide_dir ./wsi --slide_ext .svs --noise_type gaussian --sigma 15

  # Restrict to slides in CSV (e.g. train split):
  python create_noisy_patches.py ... --csv_path dataset_csv/splits_0.csv  # use 'train' column or first column for slide list
"""
from __future__ import print_function

import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def add_gaussian_noise(imgs, sigma, rng=None):
    """imgs: uint8 [N,H,W,C]. Add Gaussian noise and clip."""
    if rng is None:
        rng = np.random.default_rng()
    imgs = imgs.astype(np.float32)
    noise = rng.normal(0, sigma, imgs.shape).astype(np.float32)
    out = np.clip(imgs + noise, 0, 255).astype(np.uint8)
    return out


def add_gaussian_blur(imgs, kernel_size, sigma_x, rng=None):
    """imgs: uint8 [N,H,W,C]. Apply Gaussian blur per image."""
    if not HAS_CV2:
        raise ImportError("opencv-python required for gaussian_blur. pip install opencv-python")
    out = np.zeros_like(imgs)
    for i in range(len(imgs)):
        out[i] = cv2.GaussianBlur(imgs[i], (kernel_size, kernel_size), sigma_x)
    return out


def add_noise(imgs, noise_type, sigma=15, blur_kernel=5, blur_sigma=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if noise_type == 'gaussian':
        return add_gaussian_noise(imgs, sigma, rng)
    if noise_type == 'gaussian_blur':
        if not HAS_CV2:
            raise ImportError("opencv-python required for gaussian_blur")
        blurred = add_gaussian_blur(imgs, blur_kernel, blur_sigma, rng)
        return add_gaussian_noise(blurred, sigma, rng)
    raise ValueError("noise_type must be 'gaussian' or 'gaussian_blur'")


def get_patch_h5_subdir(patch_h5_dir):
    for sub in ['patches', 'h5_files']:
        p = os.path.join(patch_h5_dir, sub)
        if os.path.isdir(p):
            return p
    return patch_h5_dir


def get_slide_ids(patch_h5_subdir, csv_path=None):
    if csv_path and os.path.isfile(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        if 'train' in df.columns:
            # boolean-style: index is slide_id, column 'train' True/False
            if df.index.dtype == object or df.index.dtype.name == 'string':
                return df.index[df['train']].astype(str).tolist()
            if 'slide_id' in df.columns:
                return df['slide_id'].dropna().astype(str).unique().tolist()
            return df.iloc[:, 0].dropna().astype(str).unique().tolist()
        if 'slide_id' in df.columns:
            ids = df['slide_id'].dropna().astype(str).unique().tolist()
        else:
            ids = df.iloc[:, 0].dropna().astype(str).unique().tolist()
        # keep only those that have a patch h5
        existing = {os.path.splitext(f)[0] for f in os.listdir(patch_h5_subdir) if f.endswith('.h5')}
        return [s for s in ids if s in existing]
    return [os.path.splitext(f)[0] for f in os.listdir(patch_h5_subdir) if f.endswith('.h5')]


def process_slide_from_imgs(slide_id, patch_h5_subdir, output_dir, noise_type, sigma, blur_kernel, blur_sigma, seed):
    in_path = os.path.join(patch_h5_subdir, slide_id + '.h5')
    out_sub = os.path.join(output_dir, 'patches')
    os.makedirs(out_sub, exist_ok=True)
    out_path = os.path.join(out_sub, slide_id + '.h5')

    if not os.path.isfile(in_path):
        return False, 0, 'Patch H5 not found'

    rng = np.random.default_rng(seed)
    with h5py.File(in_path, 'r') as f:
        if 'imgs' not in f:
            return False, 0, "No 'imgs' in H5 (use --data_slide_dir to read from WSI)"
        imgs = f['imgs'][:]
        coords = f['coords'][:] if 'coords' in f else None
        attrs = {}
        if coords is not None and 'coords' in f and hasattr(f['coords'], 'attrs'):
            attrs = dict(f['coords'].attrs)

    noised = add_noise(imgs, noise_type, sigma=sigma, blur_kernel=blur_kernel, blur_sigma=blur_sigma, rng=rng)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('imgs', data=noised, compression='gzip')
        if coords is not None:
            d = f.create_dataset('coords', data=coords)
            for k, v in attrs.items():
                d.attrs[k] = v
    return True, len(noised), 'ok'


def process_slide_from_wsi(slide_id, patch_h5_subdir, output_dir, data_slide_dir, slide_ext,
                          noise_type, sigma, blur_kernel, blur_sigma, seed):
    import openslide
    in_path = os.path.join(patch_h5_subdir, slide_id + '.h5')
    wsi_path = os.path.join(data_slide_dir, slide_id + slide_ext)
    out_sub = os.path.join(output_dir, 'patches')
    os.makedirs(out_sub, exist_ok=True)
    out_path = os.path.join(out_sub, slide_id + '.h5')

    if not os.path.isfile(in_path):
        return False, 0, 'Patch H5 not found'
    if not os.path.isfile(wsi_path):
        return False, 0, 'WSI not found'

    with h5py.File(in_path, 'r') as f:
        if 'coords' not in f:
            return False, 0, "No 'coords' in H5"
        coords = f['coords'][:]
        patch_level = int(f['coords'].attrs.get('patch_level', 0))
        patch_size = int(f['coords'].attrs.get('patch_size', 256))
        attrs = dict(f['coords'].attrs)

    wsi = openslide.open_slide(wsi_path)
    n = len(coords)
    imgs = []
    for i in range(n):
        coord = tuple(int(x) for x in coords[i])
        pil = wsi.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
        imgs.append(np.array(pil))
    imgs = np.array(imgs)

    rng = np.random.default_rng(seed)
    noised = add_noise(imgs, noise_type, sigma=sigma, blur_kernel=blur_kernel, blur_sigma=blur_sigma, rng=rng)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('imgs', data=noised, compression='gzip')
        d = f.create_dataset('coords', data=coords)
        for k, v in attrs.items():
            d.attrs[k] = v
    return True, n, 'ok'


def main():
    parser = argparse.ArgumentParser(description='Create noisy patch set (paired with clean by slide_id and patch order)')
    parser.add_argument('--patch_h5_dir', type=str, required=True,
                        help='Root dir containing patches (patches/ or h5_files/ with slide_id.h5)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output root; noisy patches written to output_dir/patches/slide_id.h5')
    parser.add_argument('--noise_type', type=str, choices=['gaussian', 'gaussian_blur'], default='gaussian',
                        help='Noise to apply (default: gaussian)')
    parser.add_argument('--sigma', type=float, default=15,
                        help='Gaussian noise std (default 15)')
    parser.add_argument('--blur_kernel', type=int, default=5,
                        help='Kernel size for gaussian_blur (default 5)')
    parser.add_argument('--blur_sigma', type=float, default=2,
                        help='Blur sigma for gaussian_blur (default 2)')
    parser.add_argument('--data_slide_dir', type=str, default=None,
                        help='If patch H5 has only coords, WSI directory to read patches from')
    parser.add_argument('--slide_ext', type=str, default='.svs',
                        help='WSI file extension (default .svs)')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Optional CSV to restrict slides (e.g. train split); same CSV can be used for training')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    patch_h5_sub = get_patch_h5_subdir(args.patch_h5_dir)
    slide_ids = get_slide_ids(patch_h5_sub, args.csv_path)
    if not slide_ids:
        print('No slides to process.')
        return

    use_wsi = args.data_slide_dir is not None and os.path.isdir(args.data_slide_dir)
    if use_wsi:
        print('Mode: read patches from WSI (coords-only patch H5)')
    else:
        print('Mode: read patches from H5 imgs')

    ok, fail, total = 0, 0, 0
    for slide_id in tqdm(slide_ids, desc='slides'):
        if use_wsi:
            success, n, msg = process_slide_from_wsi(
                slide_id, patch_h5_sub, args.output_dir,
                args.data_slide_dir, args.slide_ext,
                args.noise_type, args.sigma, args.blur_kernel, args.blur_sigma, args.seed
            )
        else:
            success, n, msg = process_slide_from_imgs(
                slide_id, patch_h5_sub, args.output_dir,
                args.noise_type, args.sigma, args.blur_kernel, args.blur_sigma, args.seed
            )
        if success:
            ok += 1
            total += n
        else:
            fail += 1
            if fail <= 3:
                print('Skip {}: {}'.format(slide_id, msg))

    print('Done. OK={}, Failed={}, Total noisy patches={}'.format(ok, fail, total))
    print('Noisy patches saved under {}. Run your feature extractor on this and save to e.g. ./data/features/.../noisy'.format(os.path.join(args.output_dir, 'patches')))


if __name__ == '__main__':
    main()
