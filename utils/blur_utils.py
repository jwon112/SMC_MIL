import os
import argparse
import h5py
import numpy as np
import pandas as pd
from PIL import Image
import openslide
import cv2
from tqdm import tqdm

def blur_score_laplacian(pil_img, downsample=2):
    """Calculate blur score using Laplacian variance.
    
    Args:
        pil_img: PIL Image
        downsample: Downsample factor for faster computation
        
    Returns:
        float: Blur score (higher = sharper)
    """
    img = np.array(pil_img.convert("RGB"))
    if downsample and downsample > 1:
        img = cv2.resize(img, (img.shape[1]//downsample, img.shape[0]//downsample))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def sample_blur_scores(
    slide_path, coords_h5_path, patch_level=0, patch_size=256,
    max_patches=2000, seed=42
):
    """Sample blur scores from a slide.
    
    Args:
        slide_path: Path to WSI file
        coords_h5_path: Path to H5 file with coordinates
        patch_level: Patch level
        patch_size: Patch size
        max_patches: Maximum number of patches to sample
        seed: Random seed
        
    Returns:
        list: List of blur scores
    """
    rng = np.random.default_rng(seed)
    slide = openslide.OpenSlide(slide_path)

    with h5py.File(coords_h5_path, "r") as f:
        coords = f["coords"][:]  # (N, 2) top-left

    n = len(coords)
    if n == 0:
        return []

    idx = rng.choice(n, size=min(max_patches, n), replace=False)
    scores = []
    for i in idx:
        x, y = coords[i]
        region = slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert("RGB")
        scores.append(blur_score_laplacian(region, downsample=2))
    slide.close()
    return scores

def profile_blur_scores(
    csv_path, data_slide_dir, slide_ext,
    data_h5_dir, patch_level=0, patch_size=256,
    max_patches_per_slide=2000, out_csv="blur_scores.csv"
):
    """Profile blur scores across multiple slides.
    
    Args:
        csv_path: Path to CSV with slide IDs
        data_slide_dir: Directory containing WSI files
        slide_ext: Slide file extension
        data_h5_dir: Directory containing H5 coordinate files
        patch_level: Patch level
        patch_size: Patch size
        max_patches_per_slide: Maximum patches to sample per slide
        out_csv: Output CSV path
    """
    df = pd.read_csv(csv_path)
    # CLAM process_list_autogen.csv는 보통 "slide_id" 혹은 "slide" 컬럼을 가짐
    slide_ids = df["slide_id"].astype(str).tolist() if "slide_id" in df.columns else df["slide"].astype(str).tolist()

    rows = []
    for sid in tqdm(slide_ids):
        slide_path = os.path.join(data_slide_dir, sid + slide_ext)
        h5_path = os.path.join(data_h5_dir, "patches", sid + ".h5")
        if not (os.path.exists(slide_path) and os.path.exists(h5_path)):
            continue
        scores = sample_blur_scores(
            slide_path, h5_path, patch_level=patch_level, patch_size=patch_size,
            max_patches=max_patches_per_slide
        )
        for s in scores:
            rows.append({"slide_id": sid, "blur_score": s})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print("\n" + "="*60)
    print("Blur Score Statistics")
    print("="*60)
    print(out["blur_score"].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    print("\nRecommended blur threshold values:")
    print(f"  5th percentile: {out['blur_score'].quantile(0.05):.2f}")
    print(f"  10th percentile: {out['blur_score'].quantile(0.10):.2f}")
    print(f"  25th percentile: {out['blur_score'].quantile(0.25):.2f}")
    print("\nUse these values with --blur_thr in extract_features_fp.py")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile blur scores across multiple slides')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with slide IDs')
    parser.add_argument('--data_slide_dir', type=str, required=True,
                        help='Directory containing WSI files')
    parser.add_argument('--data_h5_dir', type=str, required=True,
                        help='Directory containing H5 coordinate files')
    parser.add_argument('--slide_ext', type=str, default='.svs',
                        help='Slide file extension (default: .svs)')
    parser.add_argument('--patch_level', type=int, default=0,
                        help='Patch level (default: 0)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size (default: 256)')
    parser.add_argument('--max_patches_per_slide', type=int, default=2000,
                        help='Maximum patches to sample per slide (default: 2000)')
    parser.add_argument('--out_csv', type=str, default='blur_scores.csv',
                        help='Output CSV path (default: blur_scores.csv)')
    
    args = parser.parse_args()
    
    profile_blur_scores(
        csv_path=args.csv_path,
        data_slide_dir=args.data_slide_dir,
        slide_ext=args.slide_ext,
        data_h5_dir=args.data_h5_dir,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        max_patches_per_slide=args.max_patches_per_slide,
        out_csv=args.out_csv
    )
